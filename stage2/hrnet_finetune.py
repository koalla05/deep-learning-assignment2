"""CLI for joint HRNet-W32 fine-tuning and selected-checkpoint prediction export."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import lightning as L
import numpy as np
import timm
import torch
from torch.utils.data import DataLoader

from stage2.batchnorm import FrozenBatchNormStatistics, freeze_batchnorm_statistics
from stage2.experiment import (
    ImageDataset, LABEL_COLUMNS, preflight_images, read_csv, score_probabilities,
    split_rows, verify_saved_stratification,
)
from stage2.finetune import DriveRunBackup, FineTuneClassifier, copy_verified
from stage2.hrnet_head import load_wandb_key, model_state_sha256, repository_root, sha256


@dataclass(frozen=True)
class Config:
    model_name: str = "hrnet_w32.ms_in1k"
    image_size: int = 384
    fold: int = 0
    seed: int = 42
    threshold: float = 0.25
    batch_size: int = 2
    effective_batch_size: int = 16
    max_epochs: int = 20
    patience: int = 4
    backbone_lr: float = 3e-5
    head_lr: float = 3e-4
    weight_decay: float = 1e-4
    precision: str = "16-mixed"
    num_workers: int = 2
    budget_hours: float = 5.0
    export_reserve_minutes: float = 45.0
    run_id: str = "hrnet_w32_finetune_fold0_seed42"


def read_head_identity(root: Path, head_dir: Path, config: Config) -> dict:
    """Reject a head trained on different labels, folds, or image processing."""
    meta_path = head_dir / "head_best.json"
    backbone_path = head_dir / "backbone.json"
    head_path = head_dir / "head_best.pt"
    if not all(path.is_file() for path in (meta_path, backbone_path, head_path)):
        raise FileNotFoundError(f"Head transfer needs head_best.json, backbone.json, and head_best.pt in {head_dir}")
    meta = json.loads(meta_path.read_text())
    backbone = json.loads(backbone_path.read_text())
    expected = {
        "model_name": config.model_name, "image_size": config.image_size,
        "fold": config.fold, "labels": list(LABEL_COLUMNS),
        "train_csv_sha256": sha256(root / "data" / "train.csv"),
        "fold_csv_sha256": sha256(root / "folds.csv"),
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"Local head {key} differs: expected {value!r}, found {meta.get(key)!r}")
    if meta.get("checkpoint_sha256") != sha256(head_path):
        raise ValueError("Transferred head_best.pt failed its SHA-256 check")
    if backbone.get("model_name") != config.model_name or not backbone.get("pretrained_state_sha256"):
        raise ValueError("Missing or mismatched pretrained HRNet fingerprint")
    if (meta.get("pretrained_config", {}).get("mean") != backbone.get("pretrained_config", {}).get("mean")
            or meta.get("pretrained_config", {}).get("std") != backbone.get("pretrained_config", {}).get("std")):
        raise ValueError("Head and backbone normalization disagree")
    return {"head": meta, "backbone": backbone}


def preflight_inputs(root: Path, head_dir: Path, config: Config) -> tuple[list[dict], list[dict], dict]:
    verify_saved_stratification(root / "data", root / "folds.csv", config.seed)
    training, validation = split_rows(root / "data", root / "folds.csv", config.fold)
    preflight_images(training + validation, root / "data" / "train")
    identity = read_head_identity(root, head_dir, config)
    test_rows = read_csv(root / "data" / "test.csv", ("image_id",))
    sample = read_csv(root / "data" / "sample_submission.csv", ("image_id", *LABEL_COLUMNS))
    if len(test_rows) != len(sample) or {row["image_id"] for row in test_rows} != {row["image_id"] for row in sample}:
        raise ValueError("Test and sample-submission IDs differ")
    preflight_images(test_rows, root / "data" / "test")
    print(f"Inputs verified: {len(training)} train, {len(validation)} validation, {len(test_rows)} test; "
          f"head epoch {identity['head']['epoch']}; labels {list(LABEL_COLUMNS)}", flush=True)
    return training, validation, identity


def smoke_test(root: Path, head_dir: Path, config: Config) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Smoke test requires a CUDA GPU")
    training, validation, identity = preflight_inputs(root, head_dir, config)
    network = load_initialized_network(config, head_dir, identity).cuda().train()
    freeze_batchnorm_statistics(network)
    mean = tuple(identity["head"]["normalization"]["mean"])
    std = tuple(identity["head"]["normalization"]["std"])
    batches = prediction_rows(training[:config.batch_size], root / "data" / "train", config, mean, std)
    _, images, targets = next(iter(batches))
    images, targets = images.cuda(), targets.cuda()
    optimizer = torch.optim.AdamW(network.parameters(), lr=config.backbone_lr)
    use_fp16 = config.precision == "16-mixed"
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.float16 if use_fp16 else torch.bfloat16,
                        enabled=config.precision in {"16-mixed", "bf16-mixed"}):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(network(images), targets)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize()
    if not torch.isfinite(loss):
        raise ValueError("Smoke update produced nonfinite loss")
    print(f"Smoke update: batch {config.batch_size}, precision {config.precision}, "
          f"loss {float(loss):.5f}, {time.perf_counter() - start:.1f}s, "
          f"peak allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)


def load_initialized_network(config: Config, head_dir: Path, identity: dict) -> torch.nn.Module:
    network = timm.create_model(config.model_name, pretrained=True, num_classes=0)
    actual = model_state_sha256(network)
    expected = identity["backbone"]["pretrained_state_sha256"]
    if actual != expected:
        raise ValueError(f"Colab pretrained weights differ from local head extraction: {actual} != {expected}")
    network.reset_classifier(len(LABEL_COLUMNS))
    state = torch.load(head_dir / "head_best.pt", map_location="cpu", weights_only=True)
    network.get_classifier().load_state_dict(state, strict=True)
    return network


def prediction_rows(rows: list[dict], directory: Path, config: Config, mean: tuple, std: tuple) -> DataLoader:
    dataset = ImageDataset(rows, directory, config.image_size, mean, std)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=False,
                      num_workers=config.num_workers, pin_memory=True)


class SelectedCheckpoint(L.Callback):
    """Select by macro F1, then validation BCE; save full resume state each epoch."""

    def __init__(self, run_dir: Path, backup: DriveRunBackup | None, config: Config, started: float) -> None:
        super().__init__()
        self.run_dir = run_dir
        self.backup = backup
        self.patience = config.patience
        self.budget_seconds = config.budget_hours * 3600
        self.reserve_seconds = config.export_reserve_minutes * 60
        self.best_f1 = -1.0
        self.best_loss = math.inf
        self.best_epoch = 0
        self.best_path = ""
        self.stale_epochs = 0
        self.history: list[dict] = []
        self.started = started

    @property
    def state_key(self) -> str:
        return "hrnet_selected_checkpoint"

    def state_dict(self) -> dict:
        return {"best_f1": self.best_f1, "best_loss": self.best_loss,
                "best_epoch": self.best_epoch, "best_path": self.best_path,
                "stale_epochs": self.stale_epochs, "history": self.history}

    def load_state_dict(self, state_dict: dict) -> None:
        for key, value in state_dict.items():
            setattr(self, key, value)

    def on_validation_end(self, trainer: L.Trainer, module: L.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        epoch = trainer.current_epoch + 1
        metrics = trainer.callback_metrics
        f1, loss = float(metrics["val_macro_f1"]), float(metrics["val_loss"])
        if not math.isfinite(f1) or not math.isfinite(loss):
            raise ValueError("Nonfinite validation metric")
        improved = f1 > self.best_f1 or (f1 == self.best_f1 and loss < self.best_loss)
        if improved:
            self.best_f1, self.best_loss, self.best_epoch = f1, loss, epoch
            self.stale_epochs = 0
            path = self.run_dir / "checkpoints" / f"best_epoch_{epoch:02d}.ckpt"
            if path.exists():
                raise FileExistsError(path)
            self.best_path = str(path)
            trainer.save_checkpoint(path, weights_only=False)
        else:
            self.stale_epochs += 1
        record = {"epoch": epoch, "train_loss": float(metrics["train_loss_current"]),
                  "val_loss": loss, "macro_f1": f1, "macro_ap": float(metrics["val_macro_ap"]),
                  **{f"f1_{label}": float(metrics[f"val_f1_{label.lower()}"]) for label in LABEL_COLUMNS},
                  **{f"ap_{label}": float(metrics[f"val_ap_{label.lower()}"]) for label in LABEL_COLUMNS},
                  "elapsed_minutes": (time.monotonic() - self.started) / 60}
        self.history.append(record)
        history_path = self.run_dir / "history.csv"
        with history_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=record.keys())
            writer.writeheader()
            writer.writerows(self.history)
        resume_path = self.run_dir / "checkpoints" / f"resume_epoch_{epoch:02d}.ckpt"
        if resume_path.exists():
            raise FileExistsError(resume_path)
        if self.backup:
            self.backup.save_epoch(trainer, Path(self.best_path), epoch)
        else:
            trainer.save_checkpoint(resume_path, weights_only=False)
        print(f"epoch {epoch}: macro F1 {f1:.5f}; per-label F1 "
              f"{ {label: round(record[f'f1_{label}'], 4) for label in LABEL_COLUMNS} }; "
              f"macro AP {record['macro_ap']:.5f}; train/val BCE "
              f"{record['train_loss']:.5f}/{loss:.5f}; best epoch {self.best_epoch}", flush=True)
        elapsed = time.monotonic() - self.started
        average_epoch = elapsed / epoch
        if elapsed + average_epoch >= self.budget_seconds - self.reserve_seconds:
            print("Stopping at the time budget boundary to preserve prediction export time.", flush=True)
            trainer.should_stop = True
        if self.stale_epochs >= self.patience:
            trainer.should_stop = True


class HRNetClassifier(FineTuneClassifier):
    """Add per-label AP to the shared training metrics."""

    def __init__(self, network: torch.nn.Module, backbone_lr: float, head_lr: float,
                 weight_decay: float, threshold: float) -> None:
        super().__init__(network, backbone_lr, head_lr, weight_decay, threshold)
        self.training_losses: list[tuple[float, int]] = []

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        self.training_losses.clear()

    def training_step(self, batch: tuple, batch_index: int) -> torch.Tensor:
        loss = super().training_step(batch, batch_index)
        self.training_losses.append((float(loss.detach()), len(batch[1])))
        return loss

    def on_validation_epoch_end(self) -> None:
        targets = np.concatenate(self.validation_targets)
        probabilities = np.concatenate(self.validation_probabilities)
        metrics = score_probabilities(targets, probabilities, self.threshold)
        if self.training_losses:
            weighted = sum(loss * count for loss, count in self.training_losses)
            self.log("train_loss_current", weighted / sum(count for _, count in self.training_losses))
        self.log("val_macro_f1", metrics["macro_f1"], prog_bar=True)
        self.log("val_macro_ap", metrics["macro_ap"])
        for label in LABEL_COLUMNS:
            self.log(f"val_f1_{label.lower()}", metrics["per_label_f1"][label])
            self.log(f"val_ap_{label.lower()}", metrics["per_label_ap"][label])
        self.validation_targets.clear()
        self.validation_probabilities.clear()


@torch.inference_mode()
def predict_logits(module: HRNetClassifier, loader: DataLoader, device: torch.device) -> tuple[list[str], np.ndarray]:
    module.to(device).eval()
    image_ids, parts = [], []
    for ids, images, _ in loader:
        image_ids.extend(ids)
        parts.append(module(images.to(device)).float().cpu().numpy())
    if not parts:
        raise ValueError("Prediction split is empty")
    logits = np.concatenate(parts)
    if not np.isfinite(logits).all():
        raise ValueError("Nonfinite logits")
    return image_ids, logits


def write_matrix(path: Path, image_ids: list[str], values: np.ndarray, fold: int | None = None) -> None:
    if path.exists() or len(image_ids) != len(set(image_ids)) or values.shape != (len(image_ids), len(LABEL_COLUMNS)):
        raise ValueError(f"Existing path or invalid prediction alignment: {path}")
    with path.open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("image_id", *(("fold",) if fold is not None else ()), *LABEL_COLUMNS))
        for image_id, row in zip(image_ids, values):
            writer.writerow((image_id, *((fold,) if fold is not None else ()), *map(float, row)))


def verify_prediction_pair(logits_path: Path, probabilities_path: Path, expected_ids: list[str]) -> None:
    """Read both CSVs back and check IDs, column order, and sigmoid agreement."""
    matrices = []
    for path in (logits_path, probabilities_path):
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames not in (["image_id", *LABEL_COLUMNS], ["image_id", "fold", *LABEL_COLUMNS]):
                raise ValueError(f"Unexpected prediction schema: {path}")
            rows = list(reader)
        if [row["image_id"] for row in rows] != expected_ids:
            raise ValueError(f"Prediction ID order differs: {path}")
        values = np.asarray([[float(row[label]) for label in LABEL_COLUMNS] for row in rows])
        if values.shape != (len(expected_ids), len(LABEL_COLUMNS)) or not np.isfinite(values).all():
            raise ValueError(f"Invalid saved values: {path}")
        matrices.append(values)
    logits, probabilities = matrices
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Saved probabilities are outside [0, 1]")
    sigmoid = torch.from_numpy(logits).sigmoid().numpy()
    if not np.allclose(sigmoid, probabilities, atol=1e-6, rtol=0):
        raise ValueError("Saved logits and probabilities disagree")


def export_predictions(
    root: Path, run_dir: Path, config: Config, identity: dict,
    checkpoint: Path, validation: list[dict], backup: DriveRunBackup | None,
) -> dict:
    device = torch.device("cuda")
    model = timm.create_model(config.model_name, pretrained=False, num_classes=len(LABEL_COLUMNS))
    module = HRNetClassifier.load_from_checkpoint(
        checkpoint, network=model, backbone_lr=config.backbone_lr, head_lr=config.head_lr,
        weight_decay=config.weight_decay, threshold=config.threshold, map_location=device,
    )
    mean = tuple(identity["head"]["normalization"]["mean"])
    std = tuple(identity["head"]["normalization"]["std"])
    test_rows = read_csv(root / "data" / "test.csv", ("image_id",))
    sample = read_csv(root / "data" / "sample_submission.csv", ("image_id", *LABEL_COLUMNS))
    expected_test = [row["image_id"] for row in sample]
    if len(set(expected_test)) != len(expected_test) or set(expected_test) != {row["image_id"] for row in test_rows}:
        raise ValueError("Test IDs do not match sample submission")
    preflight_images(test_rows, root / "data" / "test")
    results: dict = {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint), "files": {}}
    for partition, rows, directory, fold in (
        ("validation", validation, root / "data" / "train", config.fold),
        ("test", test_rows, root / "data" / "test", None),
    ):
        ids, logits = predict_logits(module, prediction_rows(rows, directory, config, mean, std), device)
        if ids != [row["image_id"] for row in rows]:
            raise ValueError(f"{partition} prediction IDs lost loader alignment")
        probabilities = torch.from_numpy(logits).sigmoid().numpy()
        if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError("Invalid sigmoid probabilities")
        logits_path = run_dir / f"{partition}_logits.csv"
        probabilities_path = run_dir / f"{partition}_probabilities.csv"
        write_matrix(logits_path, ids, logits, fold)
        write_matrix(probabilities_path, ids, probabilities, fold)
        verify_prediction_pair(logits_path, probabilities_path, ids)
        results["files"][partition] = {
            "logits": {"path": logits_path.name, "sha256": sha256(logits_path)},
            "probabilities": {"path": probabilities_path.name, "sha256": sha256(probabilities_path)},
            "rows": len(ids),
        }
        if partition == "validation":
            targets = np.asarray([[int(row[label]) for label in LABEL_COLUMNS] for row in rows])
            target_path = run_dir / "validation_targets.csv"
            write_matrix(target_path, ids, targets, fold)
            results["metrics"] = score_probabilities(targets, probabilities, config.threshold)
            results["files"][partition]["targets"] = {"path": target_path.name, "sha256": sha256(target_path)}
    if backup:
        backup.save_results([run_dir / item["path"] for section in results["files"].values()
                             for item in section.values() if isinstance(item, dict) and "path" in item])
    return results


def run(root: Path, head_dir: Path, drive_dir: Path | None, config: Config, resume: Path | None) -> Path:
    started = time.monotonic()
    if not torch.cuda.is_available():
        raise RuntimeError("Joint fine-tuning requires a Colab CUDA GPU")
    if config.effective_batch_size % config.batch_size or config.max_epochs < 1 or config.patience < 1:
        raise ValueError("Invalid batch size, epoch cap, or patience")
    random.seed(config.seed)
    np.random.seed(config.seed)
    L.seed_everything(config.seed, workers=True)
    training, validation, identity = preflight_inputs(root, head_dir, config)
    run_dir = root / "artifacts" / "hrnet_finetune" / config.run_id
    if resume is None:
        run_dir.mkdir(parents=True, exist_ok=False)
    elif not run_dir.is_dir() or not resume.is_file():
        raise FileNotFoundError("Resume needs an existing run directory and complete checkpoint")
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    if drive_dir is not None and not drive_dir.is_dir():
        raise FileNotFoundError(f"Mount Drive before training: {drive_dir}")
    backup = DriveRunBackup(run_dir, drive_dir / config.run_id) if drive_dir else None
    if resume is None:
        network = load_initialized_network(config, head_dir, identity)
    else:
        network = timm.create_model(config.model_name, pretrained=False, num_classes=len(LABEL_COLUMNS))
    freeze_batchnorm_statistics(network)
    mean = tuple(identity["head"]["normalization"]["mean"])
    std = tuple(identity["head"]["normalization"]["std"])
    train_set = ImageDataset(training, root / "data" / "train", config.image_size, mean, std, horizontal_flip=True)
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers, pin_memory=True)
    val_loader = prediction_rows(validation, root / "data" / "train", config, mean, std)
    module = HRNetClassifier(network, config.backbone_lr, config.head_lr, config.weight_decay, config.threshold)
    selected = SelectedCheckpoint(run_dir, backup, config, started)
    record_path = run_dir / "run.json"
    if resume is None:
        staged_manifest = root / "source_manifest.json"
        staged = json.loads(staged_manifest.read_text()) if staged_manifest.is_file() else None
        record = {"config": asdict(config), "head_checkpoint_sha256": sha256(head_dir / "head_best.pt"),
                  "pretrained_state_sha256": identity["backbone"]["pretrained_state_sha256"],
                  "train_csv_sha256": sha256(root / "data" / "train.csv"),
                  "fold_csv_sha256": sha256(root / "folds.csv"),
                  "train_rows": len(training), "validation_rows": len(validation),
                  "torch": torch.__version__, "timm": timm.__version__,
                  "lightning": L.__version__, "python": platform.python_version(),
                  "gpu": torch.cuda.get_device_name(), "git_revision": (
                      staged["git_revision"] if staged else subprocess.run(
                          ["git", "rev-parse", "HEAD"], cwd=root,
                          capture_output=True, text=True).stdout.strip()),
                  "source_manifest_sha256": sha256(staged_manifest) if staged else None}
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        if backup:
            copy_verified(record_path, backup.drive_dir / "results" / record_path.name)
    else:
        record = json.loads(record_path.read_text())
        if record["config"] != asdict(config):
            raise ValueError("Resume configuration differs from original run")
    logger = False
    if load_wandb_key(root):
        from lightning.pytorch.loggers import WandbLogger
        logger = WandbLogger(project="ai301-stage2", name=config.run_id,
                             id=config.run_id, save_dir=str(run_dir), log_model=False)
        logger.log_hyperparams(asdict(config))
    trainer = L.Trainer(
        accelerator="gpu", devices=1, precision=config.precision, max_epochs=config.max_epochs,
        accumulate_grad_batches=config.effective_batch_size // config.batch_size,
        callbacks=[FrozenBatchNormStatistics(), selected], logger=logger,
        enable_checkpointing=False, enable_progress_bar=False, deterministic="warn",
    )
    try:
        trainer.fit(module, train_loader, val_loader, ckpt_path=str(resume) if resume else None)
    finally:
        if logger:
            logger.experiment.finish()
    if not selected.best_path:
        raise RuntimeError("Training ended without a selected checkpoint")
    results = export_predictions(root, run_dir, config, identity, Path(selected.best_path), validation, backup)
    results.update({"selected_epoch": selected.best_epoch, "label_order": list(LABEL_COLUMNS),
                    "threshold": config.threshold, "head_checkpoint_sha256": record["head_checkpoint_sha256"]})
    results_path = run_dir / "results.json"
    if results_path.exists():
        raise FileExistsError(results_path)
    results_path.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    if backup:
        backup.save_results([run_dir / "history.csv", results_path])
    print(f"Selected epoch {selected.best_epoch}; macro F1 {results['metrics']['macro_f1']:.5f}; "
          f"per-label F1 {results['metrics']['per_label_f1']}; macro AP {results['metrics']['macro_ap']:.5f}; "
          f"per-label AP {results['metrics']['per_label_ap']}; artifacts {run_dir}", flush=True)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--head-dir", type=Path, required=True)
    parser.add_argument("--drive-dir", type=Path)
    parser.add_argument("--run-id", default=Config.run_id)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--max-epochs", type=int, default=Config.max_epochs)
    parser.add_argument("--precision", default=Config.precision)
    parser.add_argument("--resume", type=Path)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--preflight-only", action="store_true")
    action.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    config = Config(run_id=args.run_id, batch_size=args.batch_size,
                    max_epochs=args.max_epochs, precision=args.precision)
    root = repository_root(args.root)
    if args.preflight_only:
        preflight_inputs(root, args.head_dir.resolve(), config)
        return
    if args.smoke_only:
        smoke_test(root, args.head_dir.resolve(), config)
        return
    run(root, args.head_dir.resolve(),
        args.drive_dir.resolve() if args.drive_dir else None, config,
        args.resume.resolve() if args.resume else None)


if __name__ == "__main__":
    main()
