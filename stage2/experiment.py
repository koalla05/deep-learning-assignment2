"""Shared data and result contracts for the Stage 2 image experiments."""

from __future__ import annotations

import csv
import json
import math
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
from torchvision.transforms import Normalize, ToTensor

LABEL_COLUMNS = ("Cat", "Dog", "Moris", "Motya", "Biatrix")


def preflight_images(rows: Sequence[dict[str, str]], image_dir: Path) -> None:
    """Decode required images before an expensive run starts."""
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    for row in rows:
        path = image_dir / f"{row['image_id']}.jpg"
        if not path.is_file():
            raise FileNotFoundError(f"Missing image: {path}")
        try:
            with Image.open(path) as image:
                image.load()
        except (OSError, ValueError) as error:
            raise ValueError(f"Cannot decode image: {path}") from error


def read_csv(path: Path, expected_columns: Sequence[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Put the competition data under data/.")
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != list(expected_columns):
            raise ValueError(f"{path} has columns {reader.fieldnames}; expected {list(expected_columns)}.")
        return list(reader)


def split_rows(data_dir: Path, folds_csv: Path, validation_fold: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Join the fixed folds to targets by image ID, never by row position."""
    if validation_fold not in range(5):
        raise ValueError("validation_fold must be 0–4.")
    rows = read_csv(data_dir / "train.csv", ("image_id", *LABEL_COLUMNS))
    folds = read_csv(folds_csv, ("image_id", "fold"))
    image_ids = [row["image_id"] for row in rows]
    fold_ids = [row["image_id"] for row in folds]
    if len(image_ids) != len(set(image_ids)) or len(fold_ids) != len(set(fold_ids)):
        raise ValueError("Training or fold image IDs are duplicated.")
    if set(image_ids) != set(fold_ids):
        raise ValueError("train.csv and folds.csv contain different image IDs.")
    if any(row[label] not in {"0", "1"} for row in rows for label in LABEL_COLUMNS):
        raise ValueError("All five training labels must be binary and nonmissing.")
    try:
        fold_by_id = {row["image_id"]: int(row["fold"]) for row in folds}
    except ValueError as error:
        raise ValueError("folds.csv contains a non-integer fold.") from error
    if set(fold_by_id.values()) != set(range(5)):
        raise ValueError("Expected exactly the saved folds 0–4.")
    training = [row for row in rows if fold_by_id[row["image_id"]] != validation_fold]
    validation = [row for row in rows if fold_by_id[row["image_id"]] == validation_fold]
    return training, validation


def verify_saved_stratification(data_dir: Path, folds_csv: Path, seed: int = 42) -> dict[str, int]:
    """Check that folds.csv is the original shuffled, combination-stratified K-fold split."""
    rows = read_csv(data_dir / "train.csv", ("image_id", *LABEL_COLUMNS))
    folds = read_csv(folds_csv, ("image_id", "fold"))
    saved = {row["image_id"]: int(row["fold"]) for row in folds}
    if len(saved) != len(rows) or set(saved) != {row["image_id"] for row in rows}:
        raise ValueError("Cannot verify folds: IDs are missing or duplicated.")
    combinations = ["".join(row[label] for label in LABEL_COLUMNS) for row in rows]
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    # One rare combination has only four images, so sklearn emits an expected warning.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The least populated class", category=UserWarning)
        for fold, (_, validation_indices) in enumerate(splitter.split(rows, combinations)):
            if any(saved[rows[index]["image_id"]] != fold for index in validation_indices):
                raise ValueError("folds.csv does not match the saved stratified five-fold split.")
    return {combination: combinations.count(combination) for combination in sorted(set(combinations))}


class ImageDataset(Dataset):
    """Return an ID, normalized image, and optionally five targets."""

    def __init__(
        self,
        rows: Sequence[dict[str, str]],
        image_dir: Path,
        image_size: int,
        mean: Sequence[float],
        std: Sequence[float],
        horizontal_flip: bool = False,
    ) -> None:
        self.rows = list(rows)
        self.image_dir = image_dir
        self.image_size = image_size
        self.mean = tuple(mean)
        self.normalize = Normalize(mean, std)
        self.to_tensor = ToTensor()
        self.horizontal_flip = horizontal_flip
        if len(self.mean) != 3 or len(std) != 3:
            raise ValueError("RGB normalization needs three means and standard deviations.")
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Missing image directory: {image_dir}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        image_id = row["image_id"]
        path = self.image_dir / f"{image_id}.jpg"
        with Image.open(path) as source:
            image = ImageOps.contain(source.convert("RGB"), (self.image_size, self.image_size), Image.Resampling.BICUBIC)
        if self.horizontal_flip and torch.rand(()).item() < 0.5:
            image = ImageOps.mirror(image)
        fill = tuple(round(value * 255) for value in self.mean)
        canvas = Image.new("RGB", (self.image_size, self.image_size), fill)
        canvas.paste(image, ((self.image_size - image.width) // 2, (self.image_size - image.height) // 2))
        targets = torch.tensor([float(row[label]) for label in LABEL_COLUMNS], dtype=torch.float32) if "Cat" in row else torch.empty(0)
        return image_id, self.normalize(self.to_tensor(canvas)), targets


def score_probabilities(targets: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    """Macro F1 and AP, with the same five-label ordering everywhere."""
    if targets.shape != probabilities.shape or targets.ndim != 2 or targets.shape[1] != len(LABEL_COLUMNS):
        raise ValueError("Targets and probabilities must both have shape (images, 5).")
    if not 0 < threshold < 1 or not np.isfinite(probabilities).all():
        raise ValueError("Threshold must be in (0, 1) and probabilities must be finite.")
    if np.any((probabilities < 0) | (probabilities > 1)) or not np.isin(targets, [0, 1]).all():
        raise ValueError("Expected binary targets and probabilities in [0, 1].")
    per_f1 = f1_score(targets, probabilities >= threshold, average=None, zero_division=0)
    per_ap = []
    for index in range(len(LABEL_COLUMNS)):
        if len(np.unique(targets[:, index])) < 2:
            per_ap.append(math.nan)
        else:
            per_ap.append(float(average_precision_score(targets[:, index], probabilities[:, index])))
    return {
        "threshold": threshold,
        "macro_f1": float(np.mean(per_f1)),
        "per_label_f1": dict(zip(LABEL_COLUMNS, map(float, per_f1))),
        "macro_ap": float(np.nanmean(per_ap)) if np.isfinite(per_ap).any() else None,
        "per_label_ap": dict(zip(LABEL_COLUMNS, [value if math.isfinite(value) else None for value in per_ap])),
    }


def write_probabilities(path: Path, image_ids: Sequence[str], probabilities: np.ndarray, fold: int) -> None:
    """Save validation predictions with explicit ID and fold provenance."""
    if len(image_ids) != len(set(image_ids)) or probabilities.shape != (len(image_ids), len(LABEL_COLUMNS)):
        raise ValueError("Prediction IDs must be unique and aligned with a five-column matrix.")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Predictions must be finite probabilities in [0, 1].")
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("image_id", "fold", *LABEL_COLUMNS))
        writer.writerows((image_id, fold, *map(float, values)) for image_id, values in zip(image_ids, probabilities))


def write_run_record(path: Path, record: dict) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as file:
        json.dump(record, file, indent=2, allow_nan=False)
        file.write("\n")


def form_submission(
    sample_csv: Path, test_ids: Sequence[str], probabilities: np.ndarray, threshold: float
) -> list[dict[str, str | int]]:
    """Return checked binary rows in the competition's sample-submission order."""
    sample = read_csv(sample_csv, ("image_id", *LABEL_COLUMNS))
    sample_ids = [row["image_id"] for row in sample]
    if len(sample_ids) != len(set(sample_ids)) or len(test_ids) != len(set(test_ids)):
        raise ValueError("Test or sample-submission IDs are duplicated.")
    if set(sample_ids) != set(test_ids) or probabilities.shape != (len(test_ids), len(LABEL_COLUMNS)):
        raise ValueError("Test predictions and sample submission must have the same IDs and five labels.")
    if not 0 < threshold < 1 or not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Invalid threshold or test probabilities.")
    by_id = dict(zip(test_ids, probabilities))
    return [{"image_id": image_id, **{label: int(value >= threshold) for label, value in zip(LABEL_COLUMNS, by_id[image_id])}} for image_id in sample_ids]


def write_submission(path: Path, rows: Sequence[dict[str, str | int]]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("image_id", *LABEL_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
