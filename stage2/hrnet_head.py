"""Resumable frozen-HRNet feature extraction and local linear-head training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import timm
import torch
from torch.utils.data import DataLoader

from stage2.experiment import ImageDataset, LABEL_COLUMNS, score_probabilities, split_rows, verify_saved_stratification


@dataclass(frozen=True)
class HeadConfig:
    model_name: str = "hrnet_w32.ms_in1k"
    image_size: int = 384
    validation_fold: int = 0
    seed: int = 42
    threshold: float = 0.25
    batch_size: int = 2
    chunk_images: int = 64
    head_epochs: int = 8
    head_lr: float = 1e-3
    weight_decay: float = 1e-4
    run_id: str = "hrnet_w32_head_fold0_seed42"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_state_sha256(network: torch.nn.Module) -> str:
    """Fingerprint the actual pretrained tensors, independent of cache location."""
    digest = hashlib.sha256()
    for name, tensor in sorted(network.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def repository_root(start: Path) -> Path:
    for candidate in (start.resolve(), *start.resolve().parents):
        if (candidate / "folds.csv").is_file() and (candidate / "stage2" / "experiment.py").is_file():
            return candidate
    raise FileNotFoundError("Cannot find repository root containing folds.csv and stage2/experiment.py")


def load_wandb_key(root: Path) -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return True
    if os.environ.get("WB_API_KEY"):
        os.environ["WANDB_API_KEY"] = os.environ["WB_API_KEY"]
        return True
    env_path = root / ".env"
    if env_path.is_file():
        for line in env_path.read_text().splitlines():
            if line.startswith(("WANDB_API_KEY=", "WB_API_KEY=")):
                key = line.partition("=")[2].strip().strip('"\'')
                if key:
                    os.environ["WANDB_API_KEY"] = key
                    return True
    return False


def prepare_run(root: Path, config: HeadConfig) -> tuple[Path, list[dict], list[dict]]:
    if not config.run_id.replace("_", "").replace("-", "").isalnum():
        raise ValueError("run_id must contain only letters, numbers, underscores, or hyphens")
    seed_everything(config.seed)
    data_dir = root / "data"
    verify_saved_stratification(data_dir, root / "folds.csv", config.seed)
    training, validation = split_rows(data_dir, root / "folds.csv", config.validation_fold)
    if not training or not validation or {r["image_id"] for r in training} & {r["image_id"] for r in validation}:
        raise ValueError("Empty or overlapping train/validation partitions")
    for row in training + validation:
        image_path = data_dir / "train" / f"{row['image_id']}.jpg"
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image: {image_path}")
    run_dir = root / "artifacts" / "hrnet_head" / config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    identity = {
        "config": asdict(config), "labels": LABEL_COLUMNS,
        "train_csv_sha256": sha256(data_dir / "train.csv"),
        "fold_csv_sha256": sha256(root / "folds.csv"),
        "training_count": len(training), "validation_count": len(validation),
    }
    identity_path = run_dir / "identity.json"
    if identity_path.exists():
        previous = json.loads(identity_path.read_text())
        current = json.loads(json.dumps(identity))
        initial_epoch_cap = previous.pop("initial_head_epochs", None)
        previous_epoch_cap = previous["config"].pop("head_epochs")
        current["config"].pop("head_epochs")
        if previous != current or config.head_epochs < previous_epoch_cap:
            raise ValueError(f"Run identity changed. Choose a new run_id: {run_dir}")
        if config.head_epochs > previous_epoch_cap:
            identity["initial_head_epochs"] = initial_epoch_cap or previous_epoch_cap
            atomic_json(identity_path, identity)
    else:
        atomic_json(identity_path, identity)
    return run_dir, training, validation


def load_backbone(config: HeadConfig, device: torch.device) -> tuple[torch.nn.Module, tuple[float, ...], tuple[float, ...]]:
    network = timm.create_model(config.model_name, pretrained=True, num_classes=0)
    pretrained = network.pretrained_cfg
    mean, std = tuple(pretrained["mean"]), tuple(pretrained["std"])
    if len(mean) != 3 or len(std) != 3:
        raise ValueError("Expected RGB normalization in pretrained model configuration")
    network.to(device).eval()
    network.requires_grad_(False)
    return network, mean, std


def extract_partition(
    network: torch.nn.Module, rows: list[dict], root: Path, run_dir: Path,
    config: HeadConfig, mean: tuple[float, ...], std: tuple[float, ...],
    device: torch.device, partition: str, flip: bool,
) -> None:
    """Write atomic feature shards; a restart verifies IDs and skips completed shards."""
    output_dir = run_dir / "features" / partition
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    for start_index in range(0, len(rows), config.chunk_images):
        part = rows[start_index:start_index + config.chunk_images]
        target = output_dir / f"{start_index:06d}.npz"
        expected_ids = [row["image_id"] for row in part]
        if target.exists():
            with np.load(target, allow_pickle=False) as saved:
                if saved["image_ids"].tolist() != expected_ids or saved["features"].shape[0] != len(part):
                    raise ValueError(f"Feature shard does not match the current split: {target}")
            continue
        dataset = ImageDataset(part, root / "data" / "train", config.image_size, mean, std)
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
        feature_parts: list[np.ndarray] = []
        ids: list[str] = []
        with torch.inference_mode():
            for batch_ids, images, _ in loader:
                if flip:
                    images = images.flip(-1)
                features = network(images.to(device)).float().cpu().numpy()
                if features.ndim != 2 or not np.isfinite(features).all():
                    raise ValueError("Backbone produced invalid pooled features")
                ids.extend(batch_ids)
                feature_parts.append(features)
        if ids != expected_ids:
            raise ValueError("Feature and label image IDs lost alignment")
        targets = np.asarray([[int(row[label]) for label in LABEL_COLUMNS] for row in part], dtype=np.float32)
        temporary = target.with_suffix(".partial")
        with temporary.open("wb") as stream:
            np.savez(stream, image_ids=np.asarray(ids), features=np.concatenate(feature_parts), targets=targets)
        temporary.replace(target)
        done = min(start_index + len(part), len(rows))
        if done % 1024 < config.chunk_images or done == len(rows):
            print(f"{partition}: {done}/{len(rows)} images, {time.monotonic() - start:.0f}s this session", flush=True)


def read_partition(run_dir: Path, partition: str, rows: list[dict], chunk_images: int) -> tuple[np.ndarray, np.ndarray]:
    features, targets, ids = [], [], []
    for start in range(0, len(rows), chunk_images):
        path = run_dir / "features" / partition / f"{start:06d}.npz"
        with np.load(path, allow_pickle=False) as shard:
            ids.extend(shard["image_ids"].tolist())
            features.append(shard["features"])
            targets.append(shard["targets"])
    if ids != [row["image_id"] for row in rows]:
        raise ValueError(f"{partition} feature IDs do not match the fold manifest")
    return np.concatenate(features), np.concatenate(targets)


def start_wandb(root: Path, run_dir: Path, config: HeadConfig):
    import wandb

    has_key = load_wandb_key(root)
    mode = os.environ.get("WANDB_MODE", "online" if has_key else "offline")
    try:
        return wandb.init(project="ai301-stage2", name=config.run_id, id=config.run_id,
                          resume="allow", dir=str(run_dir), config=asdict(config), mode=mode)
    except Exception as error:
        if mode != "online":
            raise
        print(f"W&B online connection unavailable ({type(error).__name__}); keeping local metrics and offline W&B logs.", flush=True)
        return wandb.init(project="ai301-stage2", name=config.run_id, id=config.run_id,
                          resume="allow", dir=str(run_dir), config=asdict(config), mode="offline")


def train_head(
    root: Path, run_dir: Path, config: HeadConfig, training: list[dict], validation: list[dict],
    mean: tuple[float, ...], std: tuple[float, ...], model_config: dict,
) -> None:
    train_x, train_y = read_partition(run_dir, "train", training, config.chunk_images)
    flip_x, flip_y = read_partition(run_dir, "train_flip", training, config.chunk_images)
    val_x, val_y = read_partition(run_dir, "validation", validation, config.chunk_images)
    if not np.array_equal(train_y, flip_y) or train_x.shape[1] != val_x.shape[1]:
        raise ValueError("Feature targets or feature dimensions differ")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    classifier = torch.nn.Linear(train_x.shape[1], len(LABEL_COLUMNS)).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=config.head_lr, weight_decay=config.weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    x = torch.from_numpy(np.concatenate((train_x, flip_x))).float()
    y = torch.from_numpy(np.concatenate((train_y, flip_y))).float()
    validation_x = torch.from_numpy(val_x).float().to(device)
    validation_y = torch.from_numpy(val_y).float().to(device)
    latest = run_dir / "head_latest.pt"
    best = run_dir / "head_best.pt"
    history_path = run_dir / "head_history.csv"
    start_epoch, best_score, best_loss = 0, -1.0, float("inf")
    if latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        classifier.load_state_dict(state["classifier"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch, best_score, best_loss = state["epoch"], state["best_macro_f1"], state["best_val_loss"]
        torch.set_rng_state(state["torch_rng"].cpu())
        print(f"Resuming head at epoch {start_epoch + 1}", flush=True)
    run = start_wandb(root, run_dir, config)
    try:
        for epoch in range(start_epoch, config.head_epochs):
            classifier.train()
            losses = []
            order = torch.randperm(len(x))
            for indices in order.split(256):
                optimizer.zero_grad(set_to_none=True)
                logits = classifier(x[indices].to(device))
                loss = loss_fn(logits, y[indices].to(device))
                loss.backward()
                optimizer.step()
                losses.append((float(loss.detach()), len(indices)))
            classifier.eval()
            with torch.inference_mode():
                val_logits = classifier(validation_x)
                val_loss = float(loss_fn(val_logits, validation_y))
                probabilities = val_logits.sigmoid().cpu().numpy()
            metrics = score_probabilities(val_y, probabilities, config.threshold)
            train_loss = sum(value * count for value, count in losses) / len(x)
            improved = metrics["macro_f1"] > best_score or (metrics["macro_f1"] == best_score and val_loss < best_loss)
            if improved:
                best_score, best_loss = metrics["macro_f1"], val_loss
                temporary = best.with_suffix(".partial")
                torch.save({name: tensor.detach().cpu() for name, tensor in classifier.state_dict().items()}, temporary)
                temporary.replace(best)
                atomic_json(run_dir / "head_best.json", {
                    "epoch": epoch + 1, "macro_f1": best_score, "val_loss": best_loss,
                    "metrics": metrics, "checkpoint_sha256": sha256(best),
                    "model_name": config.model_name, "pretrained_config": model_config,
                    "normalization": {"mean": mean, "std": std}, "image_size": config.image_size,
                    "fold": config.validation_fold, "labels": LABEL_COLUMNS,
                    "train_csv_sha256": sha256(root / "data" / "train.csv"),
                    "fold_csv_sha256": sha256(root / "folds.csv"),
                })
            state = {
                "epoch": epoch + 1, "classifier": classifier.state_dict(),
                "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
                "best_macro_f1": best_score, "best_val_loss": best_loss,
            }
            temporary = latest.with_suffix(".partial")
            torch.save(state, temporary)
            temporary.replace(latest)
            record = {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss,
                      "macro_f1": metrics["macro_f1"], "macro_ap": metrics["macro_ap"],
                      **{f"f1_{label}": value for label, value in metrics["per_label_f1"].items()},
                      **{f"ap_{label}": value for label, value in metrics["per_label_ap"].items()}}
            with history_path.open("a", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=record.keys())
                if stream.tell() == 0:
                    writer.writeheader()
                writer.writerow(record)
            run.log(record, step=epoch + 1)
            print(f"epoch {epoch + 1}: macro F1 {metrics['macro_f1']:.5f}; per-label F1 {metrics['per_label_f1']}; macro AP {metrics['macro_ap']:.5f}; per-label AP {metrics['per_label_ap']}; train/val BCE {train_loss:.5f}/{val_loss:.5f}", flush=True)
    finally:
        run.finish()


def run_head(root: Path, config: HeadConfig) -> Path:
    run_dir, training, validation = prepare_run(root, config)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"HRNet frozen feature extraction on {device}; {len(training)} train, {len(validation)} validation", flush=True)
    network, mean, std = load_backbone(config, device)
    model_config = dict(network.pretrained_cfg)
    backbone_sha256 = model_state_sha256(network)
    provenance = {"model_name": config.model_name, "pretrained_config": model_config,
                  "pretrained_state_sha256": backbone_sha256,
                  "timm": timm.__version__, "torch": torch.__version__, "numpy": np.__version__,
                  "python": platform.python_version(), "device": str(device),
                  "git_revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()}
    provenance_path = run_dir / "backbone.json"
    if provenance_path.exists():
        previous = json.loads(provenance_path.read_text())
        if previous.get("pretrained_state_sha256") not in (None, backbone_sha256):
            raise ValueError("Pretrained backbone weights changed since feature extraction began")
    atomic_json(provenance_path, provenance)
    extract_partition(network, training, root, run_dir, config, mean, std, device, "train", False)
    extract_partition(network, training, root, run_dir, config, mean, std, device, "train_flip", True)
    extract_partition(network, validation, root, run_dir, config, mean, std, device, "validation", False)
    del network
    train_head(root, run_dir, config, training, validation, mean, std, model_config)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", default=HeadConfig.run_id)
    parser.add_argument("--batch-size", type=int, default=HeadConfig.batch_size)
    parser.add_argument("--head-epochs", type=int, default=HeadConfig.head_epochs)
    args = parser.parse_args()
    if args.batch_size < 1 or args.head_epochs < 1:
        parser.error("batch size and head epochs must be positive")
    config = HeadConfig(run_id=args.run_id, batch_size=args.batch_size, head_epochs=args.head_epochs)
    run_head(repository_root(args.root), config)


if __name__ == "__main__":
    main()
