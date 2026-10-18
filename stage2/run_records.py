"""Provenance and failure records for notebook runs."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import traceback
from typing import Callable, Iterator

from stage2.experiment import write_run_record


def file_sha256(path: Path) -> str:
    """Hash large checkpoints without loading them into memory."""
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def validate_evaluation_record(
    record_path: Path, checkpoint_path: Path, expected_configuration: dict,
    experiment: str = "02_convnext_custom",
) -> dict:
    """Reject unverified checkpoints or configuration changes before evaluation."""
    record = json.loads(record_path.read_text())
    if experiment not in {"01_convnext_standard", "02_convnext_custom"}:
        raise ValueError("Unknown experiment for checkpoint evaluation.")
    if record.get("status") != "completed" or record.get("experiment") != experiment:
        raise ValueError("Evaluation requires a completed matching run record, not a smoke or timing run.")
    fields = {
        "model": ("name", "variant", "initial_head_checkpoint", "initial_head_sha256"),
        "data": ("image_size", "transform", "fold_file", "fold_sha256", "mean", "std", "label_columns"),
        "training": (
            "optimizer", "scheduler", "loss", "max_epochs", "patience", "physical_batch_size",
            "effective_batch_size", "precision", "stage_policy", "checkpoint_selection", "gradient_clip_norm",
        ),
        "metrics": ("threshold",),
    }
    if experiment == "02_convnext_custom":
        fields["model"] += ("block_reduction",)
    differences = [
        f"{section}.{key}"
        for section, keys in fields.items()
        for key in keys
        if key not in record.get(section, {})
        or record[section][key] != expected_configuration[section][key]
    ]
    differences.extend(key for key in ("fold", "seed") if record.get(key) != expected_configuration[key])
    if differences:
        raise ValueError(f"Saved settings differ or lack provenance: {differences}")
    checkpoint_hash = record.get("artifacts", {}).get("best_checkpoint_sha256")
    if not checkpoint_hash or file_sha256(checkpoint_path) != checkpoint_hash:
        raise ValueError("Checkpoint SHA256 does not match the original completed run record.")
    return record


def validate_matched_control(custom: dict, control: dict) -> list[str]:
    """Compare the two recorded recipes, not the current notebook settings."""
    if control.get("status") != "completed" or control.get("model", {}).get("variant") != "standard":
        raise ValueError("The comparison run must be a completed standard ConvNeXt run.")
    checks = {
        "model": custom["model"]["name"] == control["model"]["name"],
        "starting head": custom["model"]["initial_head_checkpoint"] == control["model"]["initial_head_checkpoint"],
        "fold and seed": (custom["fold"], custom["seed"]) == (control["fold"], control["seed"]),
        "preprocessing": all(
            custom["data"][key] == control["data"][key]
            for key in ("image_size", "mean", "std", "transform", "fold_file")
        ),
        "threshold": custom["metrics"]["threshold"] == control["metrics"]["threshold"],
        "optimizer": custom["training"]["optimizer"].replace("backbone_and_block_lr=", "backbone_lr=")
                     == control["training"]["optimizer"],
        "full fine-tuning": custom["training"]["stage_policy"] == "full backbone, residual block and classifier trainable"
                           and control["training"]["stage_policy"] == "full backbone and classifier trainable",
    }
    for key in (
        "effective_batch_size", "precision", "max_epochs", "patience", "loss",
        "gradient_clip_norm", "scheduler", "checkpoint_selection",
    ):
        checks[key] = key in custom["training"] and key in control["training"] and custom["training"][key] == control["training"][key]
    # Older controls predate content hashes. Do not silently certify their provenance.
    for section, key in (("model", "initial_head_sha256"), ("data", "fold_sha256"), ("data", "label_columns")):
        checks[key] = key in custom[section] and key in control[section] and custom[section][key] == control[section][key]
    differences = [name for name, matched in checks.items() if not matched]
    if differences:
        raise ValueError(f"Control settings differ or lack provenance: {differences}")
    return list(checks)


@contextmanager
def record_failures(
    directory: Path | None, configuration: dict, phase: str,
    backup_failure: Callable[[Path], None] | None = None,
) -> Iterator[None]:
    """Persist safe failure metadata and always propagate the original exception."""
    try:
        yield
    except BaseException as error:
        if directory is not None:
            frames = traceback.extract_tb(error.__traceback__)
            failure = {
                **configuration, "status": "failed", "phase": phase,
                # Exception text and source lines can contain credentials or image data.
                "error": {"type": type(error).__name__, "frames": [
                    {"file": Path(frame.filename).name, "line": frame.lineno, "function": frame.name}
                    for frame in frames
                ]},
            }
            try:
                path = directory / "failure.json"
                write_run_record(path, failure)
                if backup_failure is not None:
                    try:
                        backup_failure(path)
                    except Exception as backup_error:
                        error.add_note(f"Failure record could not be backed up: {type(backup_error).__name__}")
            except Exception as recording_error:
                error.add_note(f"Failure record could not be written: {type(recording_error).__name__}")
        raise
