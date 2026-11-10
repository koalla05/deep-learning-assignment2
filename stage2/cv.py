"""Five-fold cross-validation of one CNN recipe: one fold at a time, resumable, results written once.

Layout of one model (``experiments/cv/<model>/``, or a Drive folder on Colab)::

    fold<k>/head/head.pt, head.json         phase 1: linear head on frozen features
    fold<k>/checkpoints/last.ckpt           full training state after every epoch (resume)
    fold<k>/checkpoints/best.ckpt           selected weights (best held-out macro F1)
    fold<k>/training_done.json              phase 2 finished; the selected checkpoint is final
    fold<k>/oof.csv, oof_logits.csv         held-out fold probabilities and logits
    fold<k>/test.csv, test_logits.csv       test probabilities and logits of this fold's model
    fold<k>/fold.json                       completion record with SHA-256 of every file above

A fold counts as finished only when ``fold.json`` exists and every hash in it matches,
so an interruption at any point loses at most the current epoch. Rerunning the
notebook skips finished folds, resumes a fold from ``last.ckpt``, and never retrains
a fold whose training already ended. Checkpoints are Git-ignored; the CSV and JSON
files are small and are committed.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import lightning as L
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from stage2.custom_convnext import CustomConvNeXt
from stage2.ensemble import read_probabilities
from stage2.experiment import LABEL_COLUMNS, ImageDataset, score_probabilities

TRANSFORM = "mean-colour letterbox, bicubic, checkpoint normalization; train horizontal flip p=0.5"


@dataclass(frozen=True)
class Recipe:
    """Everything that defines a model's training; identical for all five folds."""

    key: str  # folder name under experiments/cv/
    title: str
    model_name: str
    backbone_lr: float
    head_lr: float
    weight_decay: float
    max_epochs: int
    patience: int
    freeze_batchnorm: bool = False
    variant: str = "standard"  # or "residual_spatial_block" (Exp2)
    block_reduction: int = 4
    image_size: int = 384
    head_only_lr: float = 1e-3
    head_epochs: int = 8
    head_patience: int = 3
    effective_batch_size: int = 16
    gradient_clip_norm: float = 1.0
    threshold: float = 0.25
    seed: int = 42
    shared_head_from: str | None = None  # reuse another model's per-fold heads (Exp2 reuses Exp1's)


@dataclass(frozen=True)
class Runtime:
    """Hardware-dependent settings; they may differ between folds without changing the recipe."""

    batch_size: int
    precision: str
    num_workers: int
    device: str
    smoke_test: bool = False
    use_wandb: bool = False
    wandb_project: str = "ai301-stage2"


# ---------------------------------------------------------------------------
# Small file helpers: every write goes to a temporary file first, then is renamed.


def sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def sha256_matches(path: Path, expected: str) -> bool:
    """True if the file has the recorded hash, also when git changed its line endings (CRLF <-> LF)."""
    if sha256(path) == expected:
        return True
    if path.suffix not in (".csv", ".json"):
        return False
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return expected in (hashlib.sha256(data).hexdigest(), hashlib.sha256(data.replace(b"\n", b"\r\n")).hexdigest())


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(text)
    os.replace(temporary, path)


def write_json_once(path: Path, record: dict) -> None:
    """Write a record that must never change once it exists."""
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    _atomic_text(path, json.dumps(record, indent=2, allow_nan=False) + "\n")


def write_matrix(path: Path, image_ids: Sequence[str], values: np.ndarray, fold: int | Sequence[int] | None = None) -> None:
    """Probabilities or logits keyed by image ID, in the label order of the competition.

    `fold` is one fold for all rows, one fold per row, or None (test images have no fold).
    """
    if len(image_ids) != len(set(image_ids)) or values.shape != (len(image_ids), len(LABEL_COLUMNS)):
        raise ValueError("Prediction IDs must be unique and aligned with a five-column matrix.")
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite values for {path.name}")
    folds = None if fold is None else [int(fold)] * len(image_ids) if np.ndim(fold) == 0 else [int(value) for value in fold]
    if folds is not None and len(folds) != len(image_ids):
        raise ValueError("Expected one fold per image.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("image_id", *(("fold",) if folds is not None else ()), *LABEL_COLUMNS))
        for index, (image_id, row) in enumerate(zip(image_ids, values)):
            writer.writerow((image_id, *((folds[index],) if folds is not None else ()), *map(float, row)))
    os.replace(temporary, path)


def git_commit(root: Path) -> str | None:
    if not (root / ".git").exists():
        return None
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    return result.stdout.strip() or None


def versions() -> dict:
    import lightning
    import timm

    return {"python": platform.python_version(), "torch": torch.__version__, "timm": timm.__version__, "lightning": lightning.__version__}


@contextlib.contextmanager
def record_failure(directory: Path, phase: str, context: dict):
    """Keep a short failure note (type and stack frames, no message text) and re-raise.

    Every attempt gets its own file, so a later failure never hides an earlier one.
    """
    try:
        yield
    except BaseException as error:
        frames = [{"file": Path(frame.filename).name, "line": frame.lineno, "function": frame.name}
                  for frame in traceback.extract_tb(error.__traceback__)]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        try:
            write_json_once(directory / f"failure_{stamp}_{phase}.json", {
                **context, "status": "failed", "phase": phase, "utc": stamp,
                "error": {"type": type(error).__name__, "frames": frames},
            })
        except Exception as note_error:  # The original error matters more than its note.
            error.add_note(f"Failure note could not be written: {type(note_error).__name__}")
        raise


# ---------------------------------------------------------------------------
# Status of a fold.


def fold_dir(model_dir: Path, fold: int) -> Path:
    return model_dir / f"fold{fold}"


def load_fold_record(model_dir: Path, fold: int) -> dict:
    """The completion record of a fold, after checking that no saved file has changed."""
    directory = fold_dir(model_dir, fold)
    record = json.loads((directory / "fold.json").read_text())
    if record.get("status") != "completed" or record.get("fold") != fold:
        raise ValueError(f"{directory / 'fold.json'} is not a completed record of fold {fold}.")
    for name, entry in record["files"].items():
        path = directory / entry["path"]
        if not path.is_file() or sha256(path) != entry["sha256"]:
            raise ValueError(f"{path} is missing or differs from its recorded SHA-256 ({name}).")
    return record


def fold_status(model_dir: Path, fold: int) -> str:
    """'completed', 'trained' (predictions missing), 'in progress', 'not started', or 'broken'."""
    directory = fold_dir(model_dir, fold)
    if (directory / "fold.json").exists():
        try:
            load_fold_record(model_dir, fold)
            return "completed"
        except (ValueError, KeyError, json.JSONDecodeError):
            return "broken"
    if (directory / "training_done.json").exists():
        return "trained"
    if (directory / "checkpoints" / "last.ckpt").exists() or (directory / "head" / "head.json").exists():
        return "in progress"
    return "not started"


# ---------------------------------------------------------------------------
# Model construction and the frozen-feature head warm-up (phase 1).


def build_network(recipe: Recipe, pretrained: bool = True) -> nn.Module:
    import timm

    backbone = timm.create_model(recipe.model_name, pretrained=pretrained, num_classes=len(LABEL_COLUMNS))
    if recipe.variant == "residual_spatial_block":
        return CustomConvNeXt(backbone, reduction=recipe.block_reduction)
    if recipe.variant != "standard":
        raise ValueError(f"Unknown variant {recipe.variant!r}")
    return backbone


def preprocessing(network: nn.Module) -> dict:
    import timm

    base = network.backbone if isinstance(network, CustomConvNeXt) else network
    config = timm.data.resolve_model_data_config(base)
    return {"mean": tuple(config["mean"]), "std": tuple(config["std"])}


def apply_classifier(classifier: nn.Module, features: torch.Tensor) -> torch.Tensor:
    """BiT's classifier is a 1x1 convolution on the pooled map; the others are linear layers."""
    if isinstance(classifier, nn.Conv2d):
        return classifier(features[:, :, None, None]).flatten(1)
    return classifier(features)


def _autocast(runtime: Runtime):
    if runtime.device.startswith("cuda") and runtime.precision.startswith(("16", "bf16")):
        dtype = torch.bfloat16 if runtime.precision.startswith("bf16") else torch.float16
        return torch.autocast("cuda", dtype=dtype)
    return contextlib.nullcontext()


@torch.no_grad()  # Not inference_mode: these features are later used as inputs of a trained layer.
def pooled_features(network: nn.Module, loader: DataLoader, runtime: Runtime, flip: bool = False) -> torch.Tensor:
    """Pooled pre-classifier features of the frozen pretrained backbone (eval mode).

    For the custom ConvNeXt the residual block starts as an identity, so its features
    are those of the plain backbone.
    """
    base = network.backbone if isinstance(network, CustomConvNeXt) else network
    base.to(runtime.device).eval()
    parts = []
    for _, images, _ in loader:
        images = images.to(runtime.device)
        with _autocast(runtime):
            features = base.forward_head(base.forward_features(images.flip(-1) if flip else images), pre_logits=True)
        parts.append(features.float().cpu())
    return torch.cat(parts)


def train_head(
    classifier: nn.Module, train_views: tuple[torch.Tensor, torch.Tensor], train_targets: torch.Tensor,
    validation_features: torch.Tensor, validation_targets: np.ndarray, recipe: Recipe, runtime: Runtime, fold: int,
    log: Callable[[str], None] = print,
) -> tuple[dict, list[dict]]:
    """Train only the classifier on cached features; return its best state and the history.

    Each epoch shows every training image once, in its original or mirrored view with
    probability 1/2. That is the horizontal-flip augmentation of a frozen-backbone run,
    with the same number of updates, but the backbone runs only twice instead of once
    per epoch. Selection: best held-out macro F1 at the fixed threshold.
    """
    device = runtime.device
    classifier.to(device).train()
    original, mirrored = (view.to(device) for view in train_views)
    targets = train_targets.to(device)
    validation = validation_features.to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=recipe.head_only_lr, weight_decay=recipe.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(recipe.seed + fold)
    epochs = 1 if runtime.smoke_test else recipe.head_epochs
    best_state, best_score, wait, history = None, -math.inf, 0, []
    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(targets), generator=generator).to(device)
        use_mirror = (torch.rand(len(targets), generator=generator) < 0.5).to(device)
        total = 0.0
        for start in range(0, len(order), recipe.effective_batch_size):
            rows = order[start:start + recipe.effective_batch_size]
            features = torch.where(use_mirror[rows, None], mirrored[rows], original[rows])
            loss = loss_fn(apply_classifier(classifier, features), targets[rows])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(classifier.parameters(), recipe.gradient_clip_norm)
            optimizer.step()
            total += float(loss.detach()) * len(rows)
        classifier.eval()
        with torch.inference_mode():
            probabilities = apply_classifier(classifier, validation).sigmoid().cpu().numpy()
        classifier.train()
        metrics = score_probabilities(validation_targets, probabilities, recipe.threshold)
        history.append({"epoch": epoch, "train_loss": total / len(order), "val_macro_f1": metrics["macro_f1"],
                        "val_macro_ap": metrics["macro_ap"], **{f"val_f1_{k.lower()}": v for k, v in metrics["per_label_f1"].items()}})
        log(f"  head epoch {epoch}: train loss {total / len(order):.5f}, held-out macro F1 {metrics['macro_f1']:.5f}")
        if metrics["macro_f1"] > best_score:
            best_score, wait = metrics["macro_f1"], 0
            best_state = {name: value.detach().cpu().clone() for name, value in classifier.state_dict().items()}
        else:
            wait += 1
            if wait >= recipe.head_patience:
                break
    return best_state, history


# ---------------------------------------------------------------------------
# One fold, end to end.


class EpochReport(L.Callback):
    """A one-line progress message after every validation epoch (also visible in Colab logs)."""

    def __init__(self, fold: int, log: Callable[[str], None]) -> None:
        super().__init__()
        self.fold, self.report, self.started = fold, log, None

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self.started = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        # Lightning validates before this hook, so the epoch's train and validation metrics are both ready.
        if self.started is None:
            return
        metrics = trainer.callback_metrics
        value = lambda name: f"{float(metrics[name]):.5f}" if name in metrics else "n/a"
        self.report(f"  fold {self.fold} epoch {trainer.current_epoch + 1}/{trainer.max_epochs}: "
                 f"{(time.perf_counter() - self.started) / 60:.1f} min, train loss {value('train_loss')}, "
                 f"val loss {value('val_loss')}, val macro F1 {value('val_macro_f1')}")


def _loader(rows, image_dir: Path, recipe: Recipe, runtime: Runtime, norm: dict, train: bool, seed: int | None = None) -> DataLoader:
    dataset = ImageDataset(rows, image_dir, recipe.image_size, norm["mean"], norm["std"], horizontal_flip=train)
    options = dict(batch_size=runtime.batch_size, num_workers=runtime.num_workers, pin_memory=runtime.device.startswith("cuda"))
    if train:
        return DataLoader(dataset, shuffle=True, generator=torch.Generator().manual_seed(seed), **options)
    return DataLoader(dataset, shuffle=False, **options)


@torch.inference_mode()
def predict_logits(network: nn.Module, loader: DataLoader, runtime: Runtime) -> tuple[list[str], np.ndarray]:
    network.to(runtime.device).eval()
    image_ids, parts = [], []
    for ids, images, _ in loader:
        with _autocast(runtime):
            logits = network(images.to(runtime.device))
        image_ids.extend(ids)
        parts.append(logits.float().cpu().numpy())
    logits = np.concatenate(parts)
    if not np.isfinite(logits).all():
        raise ValueError("The selected checkpoint produced nonfinite logits.")
    return image_ids, logits


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-logits))


def run_fold(
    recipe: Recipe, runtime: Runtime, root: Path, model_dir: Path, fold: int,
    train_rows: list[dict], validation_rows: list[dict], test_rows: list[dict],
    log: Callable[[str], None] = print,
) -> dict:
    """Train (or resume) one fold and save its held-out and test predictions.

    Steps are skipped when their output already exists, so the same call continues
    an interrupted fold: head warm-up -> fine-tuning (resumed from last.ckpt) ->
    predictions -> completion record.
    """
    from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger

    from stage2.batchnorm import FrozenBatchNormStatistics
    from stage2.finetune import FineTuneClassifier

    directory = fold_dir(model_dir, fold)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "fold.json").exists():
        return load_fold_record(model_dir, fold)
    if runtime.smoke_test:
        train_rows, validation_rows, test_rows = train_rows[:64], validation_rows[:32], test_rows[:32]
    context = {"model": recipe.key, "fold": fold, "recipe": asdict(recipe), "runtime": asdict(runtime)}
    L.seed_everything(recipe.seed + fold, workers=True)
    validation_targets = np.asarray([[int(row[label]) for label in LABEL_COLUMNS] for row in validation_rows])
    network = build_network(recipe)
    norm = preprocessing(network)
    train_loader = _loader(train_rows, root / "data" / "train", recipe, runtime, norm, train=True, seed=recipe.seed + fold)
    validation_loader = _loader(validation_rows, root / "data" / "train", recipe, runtime, norm, train=False)

    # Phase 1: a new linear head for this fold, trained on frozen features of the training folds.
    # head.json is written last, so it marks a finished warm-up; a lone head.pt is redone.
    head_dir = directory / "head"
    head_path, head_record_path = head_dir / "head.pt", head_dir / "head.json"
    if recipe.shared_head_from and not head_record_path.exists():
        shared = model_dir.parent / recipe.shared_head_from / f"fold{fold}" / "head"
        if (shared / "head.json").is_file():
            shared_record = json.loads((shared / "head.json").read_text())
            if sha256(shared / "head.pt") != shared_record["head_sha256"]:
                raise ValueError(f"{shared / 'head.pt'} differs from its record.")
            head_dir.mkdir(parents=True, exist_ok=True)
            temporary = head_dir / "head.pt.partial"
            temporary.write_bytes((shared / "head.pt").read_bytes())
            os.replace(temporary, head_path)
            write_json_once(head_record_path, {**shared_record, "copied_from": str(shared.relative_to(model_dir.parent))})
            log(f"Fold {fold}: reusing the {recipe.shared_head_from} head of this fold (matched starting point).")
        else:
            log(f"Fold {fold}: no {recipe.shared_head_from} head for this fold yet; training an own head the same way.")
    if not head_record_path.exists():
        with record_failure(directory, "head", context):
            started = time.perf_counter()
            log(f"Fold {fold}: extracting frozen features for the head warm-up")
            plain_train = _loader(train_rows, root / "data" / "train", recipe, runtime, norm, train=False)
            original = pooled_features(network, plain_train, runtime)
            mirrored = pooled_features(network, plain_train, runtime, flip=True)
            validation_features = pooled_features(network, validation_loader, runtime)
            targets = torch.tensor([[float(row[label]) for label in LABEL_COLUMNS] for row in train_rows])
            extraction_seconds = time.perf_counter() - started
            state, history = train_head(network.get_classifier(), (original, mirrored), targets,
                                        validation_features, validation_targets, recipe, runtime, fold, log)
            head_dir.mkdir(parents=True, exist_ok=True)
            best = max(history, key=lambda row: row["val_macro_f1"])
            with (head_dir / "head_history.csv").open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(history[0]))
                writer.writeheader()
                writer.writerows(history)
            temporary = head_dir / "head.pt.partial"
            torch.save(state, temporary)
            os.replace(temporary, head_path)
            write_json_once(head_record_path, {
                "fold": fold, "method": "linear head on cached frozen features (original + mirrored view)",
                "optimizer": f"AdamW(lr={recipe.head_only_lr}, weight_decay={recipe.weight_decay})",
                "epochs_completed": len(history), "best_epoch": best["epoch"], "best_heldout_macro_f1": best["val_macro_f1"],
                "feature_extraction_seconds": extraction_seconds, "seconds": time.perf_counter() - started,
                "head_sha256": sha256(head_path),
            })
            log(f"Fold {fold}: head ready, best held-out macro F1 {best['val_macro_f1']:.5f} at epoch {best['epoch']} "
                f"({(time.perf_counter() - started) / 60:.1f} min; a warm-up score, not a result)")
    if sha256(head_path) != json.loads(head_record_path.read_text())["head_sha256"]:
        raise ValueError(f"{head_path} differs from the head recorded in head.json.")
    network.get_classifier().load_state_dict(torch.load(head_path, map_location="cpu", weights_only=True), strict=True)

    # Phase 2: fine-tune everything, saving the full state after every epoch.
    checkpoint_dir = directory / "checkpoints"
    done_path = directory / "training_done.json"
    if not done_path.exists():
        with record_failure(directory, "fine_tuning", context):
            network.requires_grad_(True)
            network.train()
            module = FineTuneClassifier(network, recipe.backbone_lr, recipe.head_lr, recipe.weight_decay, recipe.threshold)
            best_checkpoint = ModelCheckpoint(dirpath=checkpoint_dir, filename="best", monitor="val_macro_f1", mode="max",
                                              save_top_k=1, save_weights_only=True, enable_version_counter=False)
            resume_checkpoint = ModelCheckpoint(dirpath=checkpoint_dir, save_top_k=0, save_last=True, enable_version_counter=False)
            callbacks = [best_checkpoint, resume_checkpoint, LearningRateMonitor(logging_interval="epoch"), EpochReport(fold, log)]
            if not runtime.smoke_test:
                callbacks.append(EarlyStopping(monitor="val_macro_f1", mode="max", patience=recipe.patience))
            if recipe.freeze_batchnorm:
                callbacks.append(FrozenBatchNormStatistics())
            loggers = [CSVLogger(save_dir=directory, name="curves")]
            if runtime.use_wandb:
                from lightning.pytorch.loggers import WandbLogger

                id_path = directory / "wandb_id.txt"
                if not id_path.exists():
                    _atomic_text(id_path, f"{recipe.key}-fold{fold}-{int(time.time())}")
                loggers.append(WandbLogger(project=runtime.wandb_project, name=f"cv_{recipe.key}_fold{fold}", group=f"cv_{recipe.key}",
                                           id=id_path.read_text().strip(), resume="allow", save_dir=str(directory), log_model=False,
                                           config={**asdict(recipe), **asdict(runtime), "fold": fold}))
            trainer = L.Trainer(
                accelerator=runtime.device.split(":")[0], devices=1, precision=runtime.precision,
                max_epochs=1 if runtime.smoke_test else recipe.max_epochs,
                accumulate_grad_batches=recipe.effective_batch_size // runtime.batch_size,
                callbacks=callbacks, logger=loggers, gradient_clip_val=recipe.gradient_clip_norm,
                limit_train_batches=8 if runtime.smoke_test else 1.0, num_sanity_val_steps=0,
                log_every_n_steps=10, enable_progress_bar=False,
            )
            last = checkpoint_dir / "last.ckpt"
            resumed_from_epoch = None
            if last.exists():  # mmap reads only the loop counters, not the whole optimizer state.
                state = torch.load(last, map_location="cpu", weights_only=False, mmap=True)
                resumed_from_epoch = int(state["loops"]["fit_loop"]["epoch_progress"]["current"]["completed"])
                del state
            if resumed_from_epoch:
                log(f"Fold {fold}: resuming fine-tuning after epoch {resumed_from_epoch}")
            started = time.perf_counter()
            try:
                trainer.fit(module, train_loader, validation_loader, ckpt_path=last if last.exists() else None)
            finally:
                if runtime.use_wandb:
                    import wandb

                    wandb.finish()
            best_path = Path(best_checkpoint.best_model_path)
            if not best_path.is_file():
                raise RuntimeError("Fine-tuning ended without a selected checkpoint.")
            epochs_completed = trainer.fit_loop.epoch_progress.current.completed
            stopped_early = epochs_completed < trainer.max_epochs
            write_json_once(done_path, {
                "fold": fold, "epochs_completed": epochs_completed, "stopped_early": stopped_early,
                "best_macro_f1": float(best_checkpoint.best_model_score), "best_checkpoint": str(best_path.relative_to(directory)),
                "best_checkpoint_sha256": sha256(best_path), "resumed_from_epoch": resumed_from_epoch,
                "last_session_seconds": time.perf_counter() - started, "device": runtime.device,
            })
            log(f"Fold {fold}: fine-tuning finished after {epochs_completed} epochs; "
                f"best held-out macro F1 {float(best_checkpoint.best_model_score):.5f}")

    # Predictions of the selected checkpoint. Rewritten if a session died before fold.json.
    done = json.loads(done_path.read_text())
    best_path = directory / done["best_checkpoint"]
    if sha256(best_path) != done["best_checkpoint_sha256"]:
        raise ValueError(f"{best_path} differs from the checkpoint recorded when training ended.")
    with record_failure(directory, "prediction", context):
        state = torch.load(best_path, map_location="cpu", weights_only=True)["state_dict"]
        network.load_state_dict({name.removeprefix("network."): value for name, value in state.items()}, strict=True)
        started = time.perf_counter()
        validation_ids, validation_logits = predict_logits(network, validation_loader, runtime)
        if validation_ids != [row["image_id"] for row in validation_rows]:
            raise ValueError("The validation loader changed the image order.")
        test_loader = _loader(test_rows, root / "data" / "test", recipe, runtime, norm, train=False)
        test_ids, test_logits = predict_logits(network, test_loader, runtime)
        if test_ids != [row["image_id"] for row in test_rows]:
            raise ValueError("The test loader changed the image order.")
        validation_probabilities, test_probabilities = sigmoid(validation_logits), sigmoid(test_logits)
        files = {"oof": "oof.csv", "oof_logits": "oof_logits.csv", "test": "test.csv", "test_logits": "test_logits.csv"}
        write_matrix(directory / files["oof"], validation_ids, validation_probabilities, fold)
        write_matrix(directory / files["oof_logits"], validation_ids, validation_logits, fold)
        write_matrix(directory / files["test"], test_ids, test_probabilities)
        write_matrix(directory / files["test_logits"], test_ids, test_logits)
        prediction_seconds = time.perf_counter() - started

    metrics = score_probabilities(validation_targets, validation_probabilities, recipe.threshold)
    record = {
        "status": "completed", "model": recipe.key, "title": recipe.title, "fold": fold, "seed": recipe.seed,
        "source": "trained", "smoke_test": runtime.smoke_test,
        "recipe": asdict(recipe), "runtime": asdict(runtime),
        "data": {"fold_file": "folds.csv", "fold_sha256": sha256(root / "folds.csv"), "label_columns": list(LABEL_COLUMNS),
                 "transform": TRANSFORM, "mean": list(norm["mean"]), "std": list(norm["std"]),
                 "train_images": len(train_rows), "validation_images": len(validation_rows), "test_images": len(test_rows)},
        "head": json.loads(head_record_path.read_text()),
        "training": done,
        "metrics": {**metrics, "macro_f1_at_0_5": score_probabilities(validation_targets, validation_probabilities, 0.5)["macro_f1"]},
        "files": {name: {"path": path, "sha256": sha256(directory / path)} for name, path in files.items()},
        "best_checkpoint_sha256": done["best_checkpoint_sha256"], "prediction_seconds": prediction_seconds,
        "git_commit": git_commit(root), "versions": versions(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_json_once(directory / "fold.json", record)
    log(f"Fold {fold}: saved. Held-out macro F1 at {recipe.threshold}: {metrics['macro_f1']:.5f} "
        f"(" + ", ".join(f"{k} {v:.3f}" for k, v in metrics["per_label_f1"].items()) + ")")
    return load_fold_record(model_dir, fold)


def import_fold(
    recipe: Recipe, root: Path, model_dir: Path, fold: int, experiment: str, run_id: str,
    validation_rows: list[dict], test_ids: list[str], log: Callable[[str], None] = print,
) -> dict:
    """Use an earlier single-fold run as this model's fold, after checking its provenance.

    The fold-0 runs of Exp1-Exp7 were trained on folds 1-4 and selected on fold 0,
    exactly what fold 0 of the cross-validation needs, so they are copied instead of
    retrained. Their head warm-up differed a little from the per-fold warm-up above
    (see each notebook); the record says where every file came from.
    """
    directory = fold_dir(model_dir, fold)
    if (directory / "fold.json").exists():
        return load_fold_record(model_dir, fold)
    results = root / "experiments" / experiment / "results"
    source = json.loads((results / run_id / f"{run_id}.json").read_text())
    if source.get("status") != "completed" or source.get("fold") != fold:
        raise ValueError(f"{run_id} is not a completed fold-{fold} run.")
    if source["data"]["fold_sha256"] != sha256(root / "folds.csv") or source["data"]["label_columns"] != list(LABEL_COLUMNS):
        raise ValueError(f"{run_id} used a different fold file or label order.")
    validation_ids = [row["image_id"] for row in validation_rows]
    validation_targets = np.asarray([[int(row[label]) for label in LABEL_COLUMNS] for row in validation_rows])
    validation_probabilities = read_probabilities(results / run_id / f"{run_id}_val.csv", validation_ids)
    rescored = score_probabilities(validation_targets, validation_probabilities, recipe.threshold)
    if not np.isclose(rescored["macro_f1"], source["metrics"]["macro_f1"], rtol=0, atol=1e-9):
        raise ValueError(f"{run_id}: re-scored macro F1 differs from its record.")
    test_record = json.loads((results / f"{run_id}_test" / f"{run_id}_test.json").read_text())
    if test_record["source_checkpoint_sha256"] != source["artifacts"]["best_checkpoint_sha256"]:
        raise ValueError(f"{run_id}: the saved test probabilities come from another checkpoint.")
    test_probabilities = read_probabilities(results / f"{run_id}_test" / f"{run_id}_test.csv", test_ids)
    files = {"oof": "oof.csv", "test": "test.csv"}
    write_matrix(directory / "oof.csv", validation_ids, validation_probabilities, fold)
    write_matrix(directory / "test.csv", test_ids, test_probabilities)
    write_json_once(directory / "fold.json", {
        "status": "completed", "model": recipe.key, "title": recipe.title, "fold": fold, "seed": recipe.seed,
        "source": "imported", "smoke_test": False,
        "imported_from": {"experiment": source["experiment"], "run_id": run_id,
                          "record": str((results / run_id / f"{run_id}.json").relative_to(root)),
                          "model": source["model"], "training": source["training"]},
        "recipe": asdict(recipe),
        "data": {"fold_file": "folds.csv", "fold_sha256": source["data"]["fold_sha256"], "label_columns": list(LABEL_COLUMNS),
                 "validation_images": len(validation_ids), "test_images": len(test_ids)},
        "metrics": {**rescored, "macro_f1_at_0_5": score_probabilities(validation_targets, validation_probabilities, 0.5)["macro_f1"]},
        "files": {name: {"path": path, "sha256": sha256(directory / path)} for name, path in files.items()},
        "best_checkpoint_sha256": source["artifacts"]["best_checkpoint_sha256"],
        "git_commit": git_commit(root), "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    log(f"Fold {fold}: imported {experiment}/{run_id} (held-out macro F1 {rescored['macro_f1']:.5f}).")
    return load_fold_record(model_dir, fold)


# ---------------------------------------------------------------------------
# Collecting the five folds.


@dataclass
class CVPredictions:
    image_ids: list[str]           # training images in train.csv order
    folds: np.ndarray              # held-out fold of every training image
    targets: np.ndarray            # (images, 5)
    oof: np.ndarray                # out-of-fold probabilities (images, 5)
    test_ids: list[str]
    test_by_fold: dict[int, np.ndarray]  # each fold model's test probabilities
    records: dict[int, dict]

    @property
    def test_mean(self) -> np.ndarray:
        """Equal-weight mean of the fold models: the fold ensemble."""
        return np.mean(np.stack([self.test_by_fold[fold] for fold in sorted(self.test_by_fold)]), axis=0)


def collect_predictions(model_dir: Path, train_rows: list[dict], fold_by_id: dict[str, int], test_ids: list[str]) -> CVPredictions:
    """Out-of-fold matrix over all training images and the five test matrices, all verified."""
    image_ids = [row["image_id"] for row in train_rows]
    folds = np.asarray([fold_by_id[image_id] for image_id in image_ids])
    oof = np.full((len(image_ids), len(LABEL_COLUMNS)), np.nan)
    position = {image_id: index for index, image_id in enumerate(image_ids)}
    records, test_by_fold = {}, {}
    for fold in sorted(set(folds.tolist())):
        record = load_fold_record(model_dir, fold)
        if record.get("smoke_test"):
            raise ValueError(f"Fold {fold} is a smoke run, not a result.")
        held_out = [image_id for image_id in image_ids if fold_by_id[image_id] == fold]
        oof[[position[image_id] for image_id in held_out]] = read_probabilities(fold_dir(model_dir, fold) / record["files"]["oof"]["path"], held_out)
        test_by_fold[fold] = read_probabilities(fold_dir(model_dir, fold) / record["files"]["test"]["path"], test_ids)
        records[fold] = record
    if np.isnan(oof).any():
        raise ValueError("Some training images have no out-of-fold prediction.")
    # Folds may come from different machines; they must still be one recipe on one split.
    recipes = {json.dumps(record.get("recipe"), sort_keys=True) for record in records.values()}
    if len(recipes) > 1:
        raise ValueError(f"The folds of {model_dir.name} were trained with different recipes; compare the 'recipe' in their fold.json.")
    if len({record["data"]["fold_sha256"] for record in records.values() if "data" in record}) > 1:
        raise ValueError(f"The folds of {model_dir.name} used different fold files.")
    targets = np.asarray([[int(row[label]) for label in LABEL_COLUMNS] for row in train_rows])
    return CVPredictions(image_ids, folds, targets, oof, list(test_ids), test_by_fold, records)


def read_cv_summary(model_dir: Path, train_ids: list[str], test_ids: list[str]) -> dict:
    """Load what a model notebook saved: OOF and fold-mean test probabilities plus the summary record."""
    record = json.loads((model_dir / "cv.json").read_text())
    for entry in record["files"].values():
        if not sha256_matches(model_dir / entry["path"], entry["sha256"]):
            raise ValueError(f"{model_dir / entry['path']} differs from its recorded SHA-256.")
    with (model_dir / record["files"]["oof"]["path"]).open(newline="") as file:
        folds = {row["image_id"]: int(row["fold"]) for row in csv.DictReader(file)}
    return {
        "record": record,
        "oof": read_probabilities(model_dir / record["files"]["oof"]["path"], train_ids),
        "folds": np.asarray([folds[image_id] for image_id in train_ids]),
        "test": read_probabilities(model_dir / record["files"]["test_mean"]["path"], test_ids),
    }


def save_cv_summary(model_dir: Path, root: Path, predictions: CVPredictions, recipe: Recipe, summary: dict) -> dict:
    """Write the model-level result once: OOF probabilities, fold-mean test probabilities, cv.json.

    The record lists the SHA-256 of every fold record it was built from. A rerun with the
    same folds returns the saved record; different folds need a new folder, so an earlier
    summary is never replaced.
    """
    fold_hashes = {str(fold): sha256(fold_dir(model_dir, fold) / "fold.json") for fold in sorted(predictions.records)}
    record_path = model_dir / "cv.json"
    if record_path.exists():
        saved = json.loads(record_path.read_text())
        if saved["fold_records_sha256"] != fold_hashes:
            raise FileExistsError(f"{record_path} was built from other fold results; move it away to rebuild.")
        return saved
    files = {"oof": "oof.csv", "test_mean": "test_mean.csv"}
    write_matrix(model_dir / files["oof"], predictions.image_ids, predictions.oof, predictions.folds)
    write_matrix(model_dir / files["test_mean"], predictions.test_ids, predictions.test_mean)
    record = {
        "status": "completed", "model": recipe.key, "title": recipe.title, "recipe": asdict(recipe),
        "folds": sorted(predictions.records), "fold_sources": {str(k): v["source"] for k, v in predictions.records.items()},
        "fold_records_sha256": fold_hashes,
        "data": {"fold_file": "folds.csv", "fold_sha256": sha256(root / "folds.csv"), "label_columns": list(LABEL_COLUMNS)},
        **summary,
        "files": {name: {"path": path, "sha256": sha256(model_dir / path)} for name, path in files.items()},
        "git_commit": git_commit(root), "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "public_leaderboard_score": {},
    }
    write_json_once(record_path, record)
    return record


def epoch_history(directory: Path) -> list[dict]:
    """One row per fine-tuning epoch, joined over every session (a resumed run adds a version)."""
    rows: dict[int, dict] = {}
    for metrics in sorted((directory / "curves").glob("version_*/metrics.csv"), key=lambda path: int(path.parent.name.split("_")[1])):
        with metrics.open(newline="") as file:
            for row in csv.DictReader(file):
                if row.get("epoch") in (None, ""):
                    continue
                epoch = int(row["epoch"]) + 1
                for name in ("train_loss", "val_loss", "val_macro_f1", "val_macro_ap"):
                    if row.get(name) not in (None, ""):
                        rows.setdefault(epoch, {"epoch": epoch})[name] = float(row[name])
    return [rows[epoch] for epoch in sorted(rows)]


def ensure_wandb_key(root: Path, remove_runtime_env: bool = False) -> None:
    """Find WANDB_API_KEY in the environment, the ignored repo .env, or Colab Secrets; never print it."""
    key = os.environ.get("WANDB_API_KEY") or os.environ.get("WB_API_KEY")
    env_file = root / ".env"
    if not key and env_file.is_file():
        for line in env_file.read_text().splitlines():
            name, separator, value = line.partition("=")
            if separator and name.strip() in {"WANDB_API_KEY", "WB_API_KEY"} and value.strip().strip("\"'"):
                key = value.strip().strip("\"'")
                break
    if key:
        os.environ["WANDB_API_KEY"] = key
        if remove_runtime_env and env_file.is_file() and root.is_relative_to("/content"):
            env_file.unlink()
        return
    try:
        from google.colab import userdata
    except ImportError:
        raise RuntimeError("Set WANDB_API_KEY in the repo .env file or environment, or set STAGE2_USE_WANDB=0.") from None
    secret = userdata.get("WANDB_API_KEY")
    if not secret:
        raise RuntimeError("Add WANDB_API_KEY to Colab Secrets and grant this notebook access.")
    os.environ["WANDB_API_KEY"] = secret
