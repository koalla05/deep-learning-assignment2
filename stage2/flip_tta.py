"""Horizontal-flip test-time augmentation for a finished five-fold model.

For every fold, the selected checkpoint (``fold<k>/checkpoints/best.ckpt``) predicts the
mirrored images of its held-out fold and of the test set. The original predictions are
not recomputed or touched; the mirrored logits go to new files next to them::

    fold<k>/oof_flip_logits.csv     mirrored held-out images (with their fold)
    fold<k>/test_flip_logits.csv    mirrored test images
    fold<k>/flip.json               record: checkpoint hash, file hashes, sanity check, timing

``fold.json`` and every earlier file stay valid. A fold with a valid ``flip.json`` is skipped.

Before the mirrored pass, the checkpoint re-predicts the first few held-out batches without
the flip and compares them with the saved ``oof_logits.csv``; a large difference means the
wrong weights or preprocessing, and the fold stops before anything is written.

Command line (on the machine that holds the checkpoints)::

    python -m stage2.flip_tta --model-dir /content/drive/MyDrive/AI301-stage2/cv/siglip_b --folds 0,1,2,3,4
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch

from stage2.cv import (
    Recipe, Runtime, _autocast, _loader, build_network, fold_dir, git_commit, load_fold_record, sha256,
    sha256_matches, sigmoid, write_json_once, write_matrix,
)
from stage2.experiment import LABEL_COLUMNS, read_csv, split_rows

FLIP_FILES = {"oof_flip_logits": "oof_flip_logits.csv", "test_flip_logits": "test_flip_logits.csv"}
CHECK_BATCHES = 4  # held-out batches re-predicted without the flip as a sanity check
CHECK_TOLERANCE = 0.05  # largest allowed probability difference to the saved predictions (fp16 vs bf16 differ slightly)


def recipe_from_record(record: dict) -> Recipe:
    """Rebuild the Recipe from a fold.json, so the script cannot drift from the training run."""
    fields = dict(record["recipe"])
    if fields.get("image_hw") is not None:
        fields["image_hw"] = tuple(fields["image_hw"])
    fields["model_kwargs"] = tuple(tuple(pair) for pair in fields.get("model_kwargs") or ())
    return Recipe(**fields)


def read_matrix(path: Path, image_ids: Sequence[str]) -> np.ndarray:
    """Read a logit (or probability) CSV written by write_matrix, aligned to the given IDs."""
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or reader.fieldnames[-len(LABEL_COLUMNS):] != list(LABEL_COLUMNS):
            raise ValueError(f"{path} must end with the columns {list(LABEL_COLUMNS)}.")
        by_id = {row["image_id"]: [float(row[label]) for label in LABEL_COLUMNS] for row in reader}
    if set(by_id) != set(image_ids) or len(by_id) != len(image_ids):
        raise ValueError(f"{path} does not cover exactly the expected image IDs.")
    values = np.asarray([by_id[image_id] for image_id in image_ids])
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains nonfinite values.")
    return values


@torch.inference_mode()
def predict(network: torch.nn.Module, loader, runtime: Runtime, flip: bool, max_batches: int | None = None) -> tuple[list[str], np.ndarray]:
    network.to(runtime.device).eval()
    image_ids, parts = [], []
    for index, (ids, images, _) in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        images = images.to(runtime.device)
        if flip:
            images = images.flip(-1)  # (batch, channels, height, width): mirror left-right
        with _autocast(runtime):
            logits = network(images)
        image_ids.extend(ids)
        parts.append(logits.float().cpu().numpy())
    logits = np.concatenate(parts)
    if not np.isfinite(logits).all():
        raise ValueError("The checkpoint produced nonfinite logits.")
    return image_ids, logits


def flip_status(model_dir: Path, fold: int) -> str:
    """'completed' (flip.json and matching files), 'broken' (files changed), or 'not started'."""
    path = fold_dir(model_dir, fold) / "flip.json"
    if not path.is_file():
        return "not started"
    record = json.loads(path.read_text())
    ok = all(sha256_matches(fold_dir(model_dir, fold) / entry["path"], entry["sha256"]) for entry in record["files"].values())
    return "completed" if ok else "broken"


def flip_fold(model_dir: Path, root: Path, fold: int, runtime: Runtime, log: Callable[[str], None] = print) -> dict:
    """Mirrored predictions of one fold's selected checkpoint; returns the flip.json record."""
    directory = fold_dir(model_dir, fold)
    status = flip_status(model_dir, fold)
    if status == "completed":
        log(f"Fold {fold}: flip predictions already saved.")
        return json.loads((directory / "flip.json").read_text())
    if status == "broken":
        raise RuntimeError(f"{directory / 'flip.json'} exists but its files changed. Inspect them; nothing was overwritten.")

    record = load_fold_record(model_dir, fold)  # verifies the hashes of the original predictions
    if record.get("smoke_test"):
        raise ValueError(f"Fold {fold} is a smoke run, not a result.")
    done = json.loads((directory / "training_done.json").read_text())
    best_path = directory / done["best_checkpoint"]
    if not best_path.is_file():
        raise FileNotFoundError(f"Missing {best_path}. Point --model-dir at the folder that holds the checkpoints.")
    if sha256(best_path) != done["best_checkpoint_sha256"]:
        raise ValueError(f"{best_path} differs from the checkpoint recorded when training ended.")

    recipe = recipe_from_record(record)
    norm = {"mean": tuple(record["data"]["mean"]), "std": tuple(record["data"]["std"])}
    network = build_network(recipe, pretrained=False)  # the checkpoint holds every weight; no download
    state = torch.load(best_path, map_location="cpu", weights_only=True)["state_dict"]
    network.load_state_dict({name.removeprefix("network."): value for name, value in state.items()}, strict=True)

    _, validation_rows = split_rows(root / "data", root / "folds.csv", fold)
    test_rows = read_csv(root / "data" / "test.csv", ("image_id",))
    validation_ids = [row["image_id"] for row in validation_rows]
    validation_loader = _loader(validation_rows, root / "data" / "train", recipe, runtime, norm, train=False)
    test_loader = _loader(test_rows, root / "data" / "test", recipe, runtime, norm, train=False)

    started = time.perf_counter()
    # Sanity check: the same weights and preprocessing must reproduce the saved predictions.
    check_ids, check_logits = predict(network, validation_loader, runtime, flip=False, max_batches=CHECK_BATCHES)
    saved = read_matrix(directory / record["files"]["oof_logits"]["path"], validation_ids)
    position = {image_id: index for index, image_id in enumerate(validation_ids)}
    difference = float(np.abs(sigmoid(check_logits) - sigmoid(saved[[position[i] for i in check_ids]])).max())
    if difference > CHECK_TOLERANCE:
        raise ValueError(f"Fold {fold}: re-predicted probabilities differ from oof_logits.csv by {difference:.4f}; "
                         "wrong checkpoint or preprocessing. Nothing was written.")
    log(f"Fold {fold}: sanity check passed on {len(check_ids)} images (max probability difference {difference:.4f}).")

    oof_ids, oof_logits = predict(network, validation_loader, runtime, flip=True)
    if oof_ids != validation_ids:
        raise ValueError("The validation loader changed the image order.")
    test_ids, test_logits = predict(network, test_loader, runtime, flip=True)
    if test_ids != [row["image_id"] for row in test_rows]:
        raise ValueError("The test loader changed the image order.")
    seconds = time.perf_counter() - started

    write_matrix(directory / FLIP_FILES["oof_flip_logits"], oof_ids, oof_logits, fold)
    write_matrix(directory / FLIP_FILES["test_flip_logits"], test_ids, test_logits)
    flip_record = {
        "status": "completed", "model": recipe.key, "fold": fold,
        "method": "selected checkpoint on horizontally mirrored images (tensor flip of the letterboxed canvas)",
        "best_checkpoint_sha256": done["best_checkpoint_sha256"],
        "fold_record_sha256": sha256(directory / "fold.json"),
        "sanity_check": {"images": len(check_ids), "max_probability_difference": difference, "tolerance": CHECK_TOLERANCE},
        "runtime": asdict(runtime), "prediction_seconds": seconds,
        "files": {name: {"path": path, "sha256": sha256(directory / path)} for name, path in FLIP_FILES.items()},
        "git_commit": git_commit(root),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_json_once(directory / "flip.json", flip_record)
    log(f"Fold {fold}: flip predictions saved in {seconds / 60:.1f} min.")
    return flip_record


def collect_flip(model_dir: Path, image_ids: Sequence[str], folds: np.ndarray, test_ids: Sequence[str]) -> dict:
    """Original and mirrored logits of all five folds, aligned to the training and test IDs.

    Returns oof / oof_flip (images x 5 logits, each image from its held-out fold's model) and
    test / test_flip (fold -> test x 5 logits).
    """
    shape = (len(image_ids), len(LABEL_COLUMNS))
    result = {"oof": np.full(shape, np.nan), "oof_flip": np.full(shape, np.nan), "test": {}, "test_flip": {}}
    for fold in range(5):
        if flip_status(model_dir, fold) != "completed":
            raise ValueError(f"Fold {fold} has no valid flip.json yet.")
        directory = fold_dir(model_dir, fold)
        record = load_fold_record(model_dir, fold)
        rows = np.flatnonzero(folds == fold)
        held_out = [image_ids[index] for index in rows]
        result["oof"][rows] = read_matrix(directory / record["files"]["oof_logits"]["path"], held_out)
        result["oof_flip"][rows] = read_matrix(directory / FLIP_FILES["oof_flip_logits"], held_out)
        result["test"][fold] = read_matrix(directory / record["files"]["test_logits"]["path"], test_ids)
        result["test_flip"][fold] = read_matrix(directory / FLIP_FILES["test_flip_logits"], test_ids)
    if np.isnan(result["oof"]).any() or np.isnan(result["oof_flip"]).any():
        raise ValueError("Some training images have no out-of-fold prediction.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, required=True, help="CV folder of one model, e.g. .../cv/siglip_b")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="repository root (data/, folds.csv)")
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("STAGE2_FLIP_BATCH_SIZE", "32")))
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("STAGE2_NUM_WORKERS", "4")))
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    ampere = device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
    precision = os.environ.get("STAGE2_PRECISION", ("bf16-mixed" if ampere else "16-mixed") if device == "cuda" else "32-true")
    runtime = Runtime(batch_size=args.batch_size, precision=precision, num_workers=args.num_workers, device=device)
    print(f"device {device} | {precision} | batch {args.batch_size} | workers {args.num_workers}")
    for fold in (int(value) for value in args.folds.split(",")):
        flip_fold(args.model_dir, args.root.resolve(), fold, runtime)


if __name__ == "__main__":
    main()
