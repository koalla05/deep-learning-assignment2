"""Load completed runs and combine their probabilities without retraining."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from stage2.custom_convnext import CustomConvNeXt
from stage2.experiment import LABEL_COLUMNS
from stage2.run_records import file_sha256

FINETUNED_VARIANTS = {"standard", "residual_spatial_block"}


def architecture_name(model_name: str) -> str:
    """'hf-hub:timm/convnext_base.fb_in22k_ft_in1k_384@rev' -> 'convnext_base'."""
    return model_name.removeprefix("hf-hub:").split("/")[-1].split("@")[0].split(".")[0]


def load_finetuned_network(record: dict, checkpoint_path: Path) -> torch.nn.Module:
    """Rebuild a completed Exp1/Exp2 network after checking its checkpoint hash."""
    if record.get("status") != "completed" or record["model"]["variant"] not in FINETUNED_VARIANTS:
        raise ValueError("Expected a completed standard or custom ConvNeXt run.")
    if record["data"]["label_columns"] != list(LABEL_COLUMNS):
        raise ValueError("The recorded label order differs from the submission schema.")
    expected_hash = record["artifacts"]["best_checkpoint_sha256"]
    actual_hash = file_sha256(checkpoint_path)
    if actual_hash != expected_hash:
        raise ValueError(f"Checkpoint SHA-256 mismatch: {actual_hash}; expected {expected_hash}.")

    import timm

    # The checkpoint holds every weight, so build the bare architecture locally; an
    # "hf-hub:" model name would fetch its config from Hugging Face even without weights.
    backbone = timm.create_model(architecture_name(record["model"]["name"]), pretrained=False, num_classes=len(LABEL_COLUMNS))
    network = (
        CustomConvNeXt(backbone, reduction=record["model"]["block_reduction"])
        if record["model"]["variant"] == "residual_spatial_block" else backbone
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = {name.removeprefix("network."): value for name, value in checkpoint["state_dict"].items()}
    network.load_state_dict(state, strict=True)
    return network.eval()


def read_probabilities(path: Path, image_ids: Sequence[str]) -> np.ndarray:
    """Read a validation or test probability CSV and align it to the given IDs."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing probabilities: {path}")
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or reader.fieldnames[-len(LABEL_COLUMNS):] != list(LABEL_COLUMNS):
            raise ValueError(f"{path} must end with the columns {list(LABEL_COLUMNS)}.")
        by_id = {row["image_id"]: [float(row[label]) for label in LABEL_COLUMNS] for row in reader}
    if len(by_id) != len(image_ids) or set(by_id) != set(image_ids):
        raise ValueError(f"{path} does not cover exactly the expected image IDs.")
    probabilities = np.asarray([by_id[image_id] for image_id in image_ids])
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError(f"{path} contains values outside [0, 1].")
    return probabilities


def write_test_probabilities(path: Path, image_ids: Sequence[str], probabilities: np.ndarray) -> None:
    """Save test probabilities keyed by image ID; test images have no fold."""
    if len(image_ids) != len(set(image_ids)) or probabilities.shape != (len(image_ids), len(LABEL_COLUMNS)):
        raise ValueError("Prediction IDs must be unique and aligned with a five-column matrix.")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Predictions must be finite probabilities in [0, 1].")
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("image_id", *LABEL_COLUMNS))
        writer.writerows((image_id, *map(float, values)) for image_id, values in zip(image_ids, probabilities))


@torch.inference_mode()
def predict_flipped_probabilities(network: torch.nn.Module, batches, device: torch.device) -> tuple[list[str], np.ndarray]:
    """Probabilities for the left-right mirror of every image, in loader order.

    The loader's images are already letterboxed. Competition photos are 450x375, so
    they fill the 384-pixel width and only the top and bottom are padded; flipping the
    canvas is therefore the same as flipping the photo first, as the training flip did.
    """
    network.to(device).eval()
    image_ids: list[str] = []
    probabilities: list[np.ndarray] = []
    for ids, images, _ in batches:
        image_ids.extend(ids)
        probabilities.append(network(images.to(device).flip(-1)).sigmoid().cpu().numpy())
    if not probabilities:
        raise ValueError("Cannot predict an empty dataset.")
    return image_ids, np.concatenate(probabilities)


def mean_probabilities(members: Sequence[np.ndarray]) -> np.ndarray:
    """Equal-weight average; weights are not tuned on the small fold-0 sample."""
    if len(members) < 2 or any(member.shape != members[0].shape for member in members):
        raise ValueError("An ensemble needs at least two aligned probability matrices.")
    return np.mean(np.stack(members), axis=0)
