"""Short ConvNeXt throughput probe; no model weights or results are saved."""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Callable

import torch
from PIL import Image, ImageOps
from torchvision.transforms import Normalize, ToTensor

LABEL_COLUMNS = ("Cat", "Dog", "Moris", "Motya", "Biatrix")


def load_split(data_dir: Path, folds_csv: Path, validation_fold: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Join labels to the stored fold assignment by image ID."""
    with (data_dir / "train.csv").open(newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["image_id", *LABEL_COLUMNS]:
            raise ValueError(f"Unexpected train.csv columns: {reader.fieldnames}")
        rows = list(reader)
    with folds_csv.open(newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["image_id", "fold"]:
            raise ValueError(f"Unexpected folds.csv columns: {reader.fieldnames}")
        fold_rows = list(reader)
    ids = [row["image_id"] for row in rows]
    fold_by_id = {row["image_id"]: int(row["fold"]) for row in fold_rows}
    if not (len(ids) == len(set(ids)) == len(fold_rows) == len(fold_by_id)):
        raise ValueError("Train or fold IDs are duplicated.")
    if set(ids) != set(fold_by_id) or set(fold_by_id.values()) != set(range(5)):
        raise ValueError("Train IDs and saved five-fold assignment do not match.")
    if validation_fold not in range(5):
        raise ValueError("validation_fold must be 0–4.")
    if any(row[label] not in {"0", "1"} for row in rows for label in LABEL_COLUMNS):
        raise ValueError("Labels must be binary and nonmissing.")
    training = [row for row in rows if fold_by_id[row["image_id"]] != validation_fold]
    validation = [row for row in rows if fold_by_id[row["image_id"]] == validation_fold]
    return training, validation


def prepare_image(path: Path, size: int, mean: tuple[float, ...], std: tuple[float, ...]) -> torch.Tensor:
    """Keep the whole image and pad with the checkpoint mean colour."""
    with Image.open(path) as source:
        image = ImageOps.contain(source.convert("RGB"), (size, size), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (size, size), tuple(round(value * 255) for value in mean))
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return Normalize(mean, std)(ToTensor()(canvas))


def load_sample(
    data_dir: Path, rows: list[dict[str, str]], size: int, mean: tuple[float, ...], std: tuple[float, ...]
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return one effective batch and mean image preparation time."""
    started = time.perf_counter()
    images = torch.stack([
        prepare_image(data_dir / "train" / f'{row["image_id"]}.jpg', size, mean, std)
        for row in rows
    ])
    seconds_per_image = (time.perf_counter() - started) / len(rows)
    targets = torch.tensor([[int(row[label]) for label in LABEL_COLUMNS] for row in rows], dtype=torch.float32)
    return images, targets, seconds_per_image


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def profile_updates(
    model: torch.nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    microbatch_size: int,
    warmup_updates: int = 1,
    timed_updates: int = 2,
    on_progress: Callable[[str, int, int, float], None] | None = None,
) -> dict[str, float]:
    """Time AdamW updates and validation; optionally report synchronized microbatches."""
    if len(images) % microbatch_size:
        raise ValueError("Effective batch must divide evenly into microbatches.")
    model = model.to(device)
    images, targets = images.to(device), targets.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    timings = []
    model.train()
    for update in range(warmup_updates + timed_updates):
        synchronize(device)
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for batch_index, first in enumerate(range(0, len(images), microbatch_size), start=1):
            last = first + microbatch_size
            logits = model(images[first:last])
            if logits.shape != targets[first:last].shape:
                raise ValueError("The model must return five logits per image.")
            loss = criterion(logits, targets[first:last]) * (len(logits) / len(images))
            loss.backward()
            if on_progress is not None:
                synchronize(device)
                on_progress("train", update + 1, batch_index, time.perf_counter() - started)
        optimizer.step()
        synchronize(device)
        if on_progress is not None:
            on_progress("optimizer", update + 1, 0, time.perf_counter() - started)
        if update >= warmup_updates:
            timings.append(time.perf_counter() - started)
    model.eval()
    synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, first in enumerate(range(0, len(images), microbatch_size), start=1):
            model(images[first:first + microbatch_size])
            if on_progress is not None:
                synchronize(device)
                on_progress("validation", 1, batch_index, time.perf_counter() - started)
    synchronize(device)
    return {
        "train_seconds_per_effective_batch": sum(timings) / len(timings),
        "validation_seconds_per_effective_batch": time.perf_counter() - started,
    }


def estimate_hours(
    n_training: int,
    n_validation: int,
    effective_batch_size: int,
    image_seconds: float,
    timings: dict[str, float],
    folds: int = 5,
    epochs_per_fold: int = 10,
) -> dict[str, float]:
    """Rough upper-scope estimate before logging, saving, and thermal overhead."""
    train_epoch = math.ceil(n_training / effective_batch_size) * timings["train_seconds_per_effective_batch"]
    validation_epoch = math.ceil(n_validation / effective_batch_size) * timings["validation_seconds_per_effective_batch"]
    image_epoch = (n_training + n_validation) * image_seconds
    return {
        "train_epoch_hours": train_epoch / 3600,
        "validation_epoch_hours": validation_epoch / 3600,
        "five_fold_hours_before_overheads": folds * epochs_per_fold * (train_epoch + validation_epoch + image_epoch) / 3600,
    }
