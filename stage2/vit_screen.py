"""Frozen-feature screen of transformer backbones, before choosing one to fine-tune.

Same protocol as the section-3 baseline (notebooks 02/03): pooled features of a frozen
pretrained backbone, one standardised logistic regression per label, out-of-fold on the
saved folds. Two extra views of the named pets:

* nested per-label thresholds (``stage2.thresholds.nested_thresholds``), as in the CV records;
* a retrieval score: cosine similarity to the nearest *training-fold* photo of that pet.
  It needs no classifier and shows how well the raw features recognise one individual animal.

Embeddings are cached under ``artifacts/vit_screen/<key>/`` (Git-ignored), results go to
``experiments/vit_screen/<key>/``.
"""

from __future__ import annotations

import contextlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from stage2.experiment import LABEL_COLUMNS, ImageDataset, score_probabilities
from stage2.thresholds import nested_thresholds

NAMED_PETS = ("Moris", "Motya", "Biatrix")


@dataclass(frozen=True)
class Candidate:
    key: str
    model_name: str
    input_hw: tuple[int, int]  # (height, width); photos are 375 high and 450 wide
    family: str


# Patch-14 models get 448x378: the whole 450x375 photo at ~native scale, ~5 px of padding.
# Patch-16 SigLIP gets 368x448 (multiples of 16). Swin's 12x12 windows need its 384x384 square.
CANDIDATES = (
    Candidate("convnext_b_ref", "convnext_base.fb_in22k_ft_in1k_384", (384, 384), "CNN reference (current pipeline input)"),
    Candidate("dinov2_b", "vit_base_patch14_reg4_dinov2.lvd142m", (378, 448), "ViT-B/14, self-supervised (DINOv2 + registers)"),
    Candidate("dinov2_l", "vit_large_patch14_reg4_dinov2.lvd142m", (378, 448), "ViT-L/14, self-supervised (DINOv2 + registers)"),
    Candidate("eva02_b", "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k", (378, 448), "ViT-B/14, MIM + IN-22k/1k fine-tune (EVA-02)"),
    Candidate("siglip_b", "vit_base_patch16_siglip_384.webli", (368, 448), "ViT-B/16, image-text contrastive (SigLIP)"),
    Candidate("swin_b", "swin_base_patch4_window12_384.ms_in22k_ft_in1k", (384, 384), "hierarchical windowed transformer (Swin)"),
    Candidate("dinov2_s", "vit_small_patch14_reg4_dinov2.lvd142m", (378, 448), "ViT-S/14, cheap DINOv2 variant"),
)


def get_candidate(key: str) -> Candidate:
    for candidate in CANDIDATES:
        if candidate.key == key:
            return candidate
    raise KeyError(key)


def build_feature_model(candidate: Candidate) -> torch.nn.Module:
    """Pretrained backbone with the classifier removed; returns its default pooled features."""
    import timm

    # ConvNeXt takes any size and Swin uses its native 384; the plain ViTs resample their position embeddings.
    fixed_size = candidate.model_name.startswith(("convnext", "swin"))
    kwargs = {} if fixed_size else {"img_size": candidate.input_hw}
    return timm.create_model(candidate.model_name, pretrained=True, num_classes=0, **kwargs).eval()


def _autocast(device: str, half: bool):
    if half and device.startswith(("cuda", "mps")):
        return torch.autocast(device.split(":")[0], dtype=torch.float16)
    return contextlib.nullcontext()


@torch.inference_mode()
def embed_rows(
    model: torch.nn.Module, rows: list[dict], image_dir: Path, candidate: Candidate, device: str,
    batch_size: int = 16, num_workers: int = 0, half: bool = False, log: Callable[[str], None] = print,
) -> np.ndarray:
    """One float32 feature row per input row, in row order."""
    import timm

    config = timm.data.resolve_model_data_config(model)
    dataset = ImageDataset(rows, image_dir, candidate.input_hw, config["mean"], config["std"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.to(device)
    parts, started = [], time.time()
    for index, (_, images, _) in enumerate(loader):
        with _autocast(device, half):
            parts.append(model(images.to(device)).float().cpu().numpy())
        if index % 100 == 0:
            done = min((index + 1) * batch_size, len(rows))
            log(f"  {candidate.key}: {done}/{len(rows)} images, {time.time() - started:.0f}s")
    features = np.concatenate(parts)
    if not np.isfinite(features).all():
        raise ValueError(f"Non-finite features from {candidate.key}; rerun without half precision.")
    return features


def cached_embeddings(
    candidate: Candidate, split: str, rows: list[dict], image_dir: Path, cache_root: Path, device: str,
    model_factory: Callable[[], torch.nn.Module], **kwargs,
) -> np.ndarray:
    """Load ``<cache>/<key>/<split>.npy`` if it matches these image IDs, else embed and save it."""
    directory = cache_root / candidate.key
    array_path, meta_path = directory / f"{split}.npy", directory / f"{split}.json"
    image_ids = [row["image_id"] for row in rows]
    if array_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["image_ids"] == image_ids and meta["candidate"] == asdict(candidate):
            return np.load(array_path)
    started = time.time()
    features = embed_rows(model_factory(), rows, image_dir, candidate, device, **kwargs)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(array_path, features)
    meta_path.write_text(json.dumps({
        "candidate": asdict(candidate), "image_ids": image_ids, "shape": list(features.shape),
        "device": device, "seconds": round(time.time() - started, 1),
    }))
    return features


def _logreg(seed: int):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))


def logreg_oof_and_test(
    train_features: np.ndarray, targets: np.ndarray, folds: np.ndarray, test_features: np.ndarray | None = None, seed: int = 42,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Notebook-02 protocol: one LogReg per label and fold for OOF; one per label on all train for test."""
    oof = np.zeros(targets.shape)
    test = None if test_features is None else np.zeros((len(test_features), targets.shape[1]))
    for label in range(targets.shape[1]):
        for fold in np.unique(folds):
            held_out = folds == fold
            model = _logreg(seed).fit(train_features[~held_out], targets[~held_out, label])
            oof[held_out, label] = model.predict_proba(train_features[held_out])[:, 1]
        if test is not None:
            test[:, label] = _logreg(seed).fit(train_features, targets[:, label]).predict_proba(test_features)[:, 1]
    return oof, test


def nearest_pet_similarity(features: np.ndarray, targets: np.ndarray, folds: np.ndarray) -> dict[str, np.ndarray]:
    """Per named pet: each image's highest cosine similarity to a photo of that pet in the *other* folds."""
    unit = features / np.linalg.norm(features, axis=1, keepdims=True)
    scores = {}
    for pet in NAMED_PETS:
        column = LABEL_COLUMNS.index(pet)
        score = np.zeros(len(unit))
        for fold in np.unique(folds):
            held_out = folds == fold
            references = unit[(~held_out) & (targets[:, column] == 1)]
            score[held_out] = (unit[held_out] @ references.T).max(axis=1)
        scores[pet] = score
    return scores


def summarize(targets: np.ndarray, oof: np.ndarray, folds: np.ndarray, retrieval: dict[str, np.ndarray]) -> dict:
    at_025 = score_probabilities(targets, oof, 0.25)
    nested = nested_thresholds(targets, oof, folds)
    return {
        "oof_macro_f1_at_025": at_025["macro_f1"],
        "oof_per_label_f1_at_025": at_025["per_label_f1"],
        "oof_macro_ap": at_025["macro_ap"],
        "oof_per_label_ap": at_025["per_label_ap"],
        "fair_macro_f1_per_label_thresholds": nested["fair_macro_f1"],
        "fair_per_label_f1": nested["fair_per_label_f1"],
        "retrieval_ap": {
            pet: float(average_precision_score(targets[:, LABEL_COLUMNS.index(pet)], retrieval[pet])) for pet in NAMED_PETS
        },
    }


def summary_row(key: str, summary: dict) -> dict:
    """One flat row for the comparison table."""
    row = {
        "model": key,
        "F1@0.25": summary["oof_macro_f1_at_025"],
        "F1 fair": summary["fair_macro_f1_per_label_thresholds"],
        "macro AP": summary["oof_macro_ap"],
    }
    for pet in NAMED_PETS:
        row[f"AP {pet}"] = summary["oof_per_label_ap"][pet]
    for pet in NAMED_PETS:
        row[f"kNN AP {pet}"] = summary["retrieval_ap"][pet]
    return row


SCREEN_ORDER = ("dinov2_s", "siglip_b", "convnext_b_ref", "swin_b", "dinov2_b", "eva02_b", "dinov2_l")  # cheapest first


def run_screen(
    root: Path, keys=SCREEN_ORDER, device: str = "mps", half: bool = True, batch_size: int = 16,
    include_test: bool = False, log: Callable[[str], None] = print,
) -> list[dict]:
    """Embed (cached), score and save each candidate; returns one summary row per finished model."""
    import csv

    from stage2.cv import write_matrix

    train_rows = list(csv.DictReader((root / "data" / "train.csv").open()))
    fold_by_id = {row["image_id"]: int(row["fold"]) for row in csv.DictReader((root / "folds.csv").open())}
    image_ids = [row["image_id"] for row in train_rows]
    folds = np.array([fold_by_id[image_id] for image_id in image_ids])
    targets = np.array([[int(row[label]) for label in LABEL_COLUMNS] for row in train_rows])
    test_rows = list(csv.DictReader((root / "data" / "test.csv").open())) if include_test else None
    cache_root, results_root = root / "artifacts" / "vit_screen", root / "experiments" / "vit_screen"

    rows = []
    for key in keys:
        candidate = get_candidate(key)
        log(f"== {key}: {candidate.model_name} at {candidate.input_hw[1]}x{candidate.input_hw[0]}")
        factory = lambda: build_feature_model(candidate)  # noqa: E731 - loaded only when the cache misses
        options = dict(batch_size=batch_size, half=half, log=log)
        train_features = cached_embeddings(candidate, "train", train_rows, root / "data" / "train", cache_root, device, factory, **options)
        test_features = None
        if include_test:
            test_features = cached_embeddings(candidate, "test", test_rows, root / "data" / "test", cache_root, device, factory, **options)
        oof, test = logreg_oof_and_test(train_features, targets, folds, test_features)
        summary = summarize(targets, oof, folds, nearest_pet_similarity(train_features, targets, folds))
        directory = results_root / key
        directory.mkdir(parents=True, exist_ok=True)
        write_matrix(directory / "oof.csv", image_ids, oof, folds)
        if test is not None:
            write_matrix(directory / "test.csv", [row["image_id"] for row in test_rows], test)
        record = {"candidate": asdict(candidate), "protocol": "frozen pooled features, StandardScaler + LogisticRegression per label, 5-fold OOF (folds.csv)",
                  "precision": "fp16 autocast" if half else "fp32", "device": device, **summary}
        (directory / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
        rows.append(summary_row(key, summary))
        log(json.dumps(rows[-1]))
    return rows


def format_table(rows: list[dict]) -> str:
    """Plain-text comparison table (the project environment has no pandas)."""
    if not rows:
        return "(no finished models)"
    columns = list(rows[0])
    cells = [[str(row[c]) if isinstance(row[c], str) else f"{row[c]:.4f}" for c in columns] for row in rows]
    widths = [max(len(c), *(len(r[i]) for r in cells)) for i, c in enumerate(columns)]
    line = lambda values: "  ".join(v.rjust(w) for v, w in zip(values, widths))  # noqa: E731
    return "\n".join([line(columns), *(line(r) for r in cells)])
