"""Optimization and timing policy for a full ConvNeXt fine-tuning run."""

from __future__ import annotations

import math
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Callable

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from stage2.train import MultilabelClassifier


def load_trained_head(network: torch.nn.Module, checkpoint_path: Path) -> None:
    """Load only the fold-specific classifier parameters, never optimizer state."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing trained head: {checkpoint_path}")
    classifier = network.get_classifier()
    if not isinstance(classifier, torch.nn.Module):
        raise TypeError("The network must expose a module classifier.")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    classifier.load_state_dict(state, strict=True)


def fine_tuning_groups(
    network: torch.nn.Module, backbone_lr: float, head_lr: float, weight_decay: float
) -> list[dict]:
    """Assign each trainable parameter one LR and the appropriate decay policy."""
    if backbone_lr <= 0 or head_lr <= 0 or weight_decay < 0:
        raise ValueError("Learning rates must be positive and weight decay nonnegative.")
    classifier = network.get_classifier()
    if not isinstance(classifier, torch.nn.Module):
        raise TypeError("The network must expose a module classifier.")
    head_ids = {id(parameter) for parameter in classifier.parameters()}
    norm_ids = {
        id(parameter)
        for module in network.modules()
        if "norm" in type(module).__name__.lower()
        for parameter in module.parameters(recurse=False)
    }
    groups: dict[tuple[float, float], list[torch.nn.Parameter]] = {}
    for name, parameter in network.named_parameters():
        if not parameter.requires_grad:
            continue
        learning_rate = head_lr if id(parameter) in head_ids else backbone_lr
        decay = 0.0 if name.endswith("bias") or id(parameter) in norm_ids else weight_decay
        groups.setdefault((learning_rate, decay), []).append(parameter)
    if not any(groups.values()) or not any(id(parameter) in head_ids for group in groups.values() for parameter in group):
        raise ValueError("Fine-tuning requires a trainable classifier and network.")
    return [
        {"params": parameters, "lr": learning_rate, "weight_decay": decay}
        for (learning_rate, decay), parameters in groups.items()
        if parameters
    ]


def warmup_cosine_multiplier(step: int, warmup_steps: int, total_steps: int, minimum: float = 0.1) -> float:
    """Linear warm-up for one epoch, then cosine decay to a fraction of peak LR."""
    if not 0 < warmup_steps <= total_steps or not 0 <= minimum <= 1:
        raise ValueError("Expected 0 < warmup_steps <= total_steps and 0 <= minimum <= 1.")
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    if warmup_steps == total_steps:
        return 1.0
    progress = min(1.0, (step - warmup_steps + 1) / (total_steps - warmup_steps))
    return minimum + (1 - minimum) * (1 + math.cos(math.pi * progress)) / 2


class FineTuneClassifier(MultilabelClassifier):
    """Use shared loss and metrics with differential learning rates."""

    def __init__(
        self, network: torch.nn.Module, backbone_lr: float, head_lr: float,
        weight_decay: float, threshold: float,
    ) -> None:
        super().__init__(network, head_lr, weight_decay, threshold)
        self.backbone_lr = backbone_lr

    def configure_optimizers(self) -> dict:
        groups = fine_tuning_groups(self.network, self.backbone_lr, self.learning_rate, self.weight_decay)
        optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999))
        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = math.ceil(total_steps / self.trainer.max_epochs)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: warmup_cosine_multiplier(step, warmup_steps, total_steps),
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        super().on_before_optimizer_step(optimizer)
        head_ids = {id(parameter) for parameter in self.network.get_classifier().parameters()}
        for name, belongs_to_head in (("head", True), ("backbone", False)):
            gradients = [
                parameter.grad.detach().float().norm(2)
                for parameter in self.network.parameters()
                if (id(parameter) in head_ids) == belongs_to_head and parameter.grad is not None
            ]
            if gradients:
                self.log(f"gradient_norm_{name}", torch.linalg.vector_norm(torch.stack(gradients)), on_step=True, on_epoch=True)


class SmokeRuntimeProfile(L.Callback):
    """Measure a bounded smoke loop, including loading and optimizer updates."""

    def __init__(self) -> None:
        super().__init__()
        self.train_seconds: list[float] = []
        self.train_images = 0
        self.validation_seconds: list[float] = []
        self.validation_images = 0
        self.validation_tail_seconds: float | None = None
        self._previous_end: float | None = None
        self._validation_end: float | None = None

    @staticmethod
    def _timestamp(device: torch.device) -> float:
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter()

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._previous_end = self._timestamp(pl_module.device)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        ended = self._timestamp(pl_module.device)
        if self._previous_end is None:
            raise RuntimeError("Training profiling started without an epoch boundary.")
        if not torch.isfinite(outputs["loss"]).all():
            raise ValueError("Smoke training produced a nonfinite loss.")
        # Exclude the entire first accumulated update, including AdamW setup.
        if batch_idx >= trainer.accumulate_grad_batches:
            self.train_seconds.append(ended - self._previous_end)
            self.train_images += len(batch[1])
        self._previous_end = ended

    def on_validation_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._previous_end = self._timestamp(pl_module.device)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        ended = self._timestamp(pl_module.device)
        if self._previous_end is None:
            raise RuntimeError("Validation profiling started without an epoch boundary.")
        if batch_idx > 0:
            self.validation_seconds.append(ended - self._previous_end)
            self.validation_images += len(batch[1])
        self._previous_end = ended
        self._validation_end = ended

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        ended = self._timestamp(pl_module.device)
        if self._validation_end is None:
            raise RuntimeError("Smoke profiling requires validation and checkpoint saving.")
        self.validation_tail_seconds = ended - self._validation_end

    def estimate(
        self, training_images: int, validation_images: int, effective_batch_size: int,
        planned_epochs: int, margin: float = 1.1,
    ) -> dict:
        """Extrapolate one fold; this is not the full-epoch runtime gate."""
        if min(training_images, validation_images, effective_batch_size, planned_epochs) < 1 or margin < 1:
            raise ValueError("Expected positive counts and a projection margin of at least one.")
        if not self.train_images or not self.validation_images or self.validation_tail_seconds is None:
            raise RuntimeError("Run a complete smoke profile with batches after warm-up first.")
        if self.train_images < 2 * effective_batch_size or self.train_images % effective_batch_size:
            raise RuntimeError("Profile at least two complete effective batches after warm-up.")
        train_per_image = sum(self.train_seconds) / self.train_images
        validation_per_image = sum(self.validation_seconds) / self.validation_images
        train_epoch = training_images * train_per_image
        validation_epoch = validation_images * validation_per_image
        epoch = train_epoch + validation_epoch + self.validation_tail_seconds
        final_evaluation = validation_epoch
        return {
            "status": "smoke_runtime_estimate",
            "measured_training_images": self.train_images,
            "measured_validation_images": self.validation_images,
            "warmup_optimizer_updates_excluded": 1,
            "warmup_validation_batches_excluded": 1,
            "train_seconds_per_effective_batch": train_per_image * effective_batch_size,
            "validation_seconds_per_image": validation_per_image,
            "validation_metrics_and_checkpoint_seconds": self.validation_tail_seconds,
            "estimated_train_epoch_seconds": train_epoch,
            "estimated_validation_epoch_seconds": validation_epoch,
            "estimated_epoch_seconds": epoch,
            "estimated_final_validation_seconds": final_evaluation,
            "planned_epochs": planned_epochs,
            "projection_margin": margin,
            "estimated_run_hours_with_margin": (planned_epochs * epoch + final_evaluation) * margin / 3600,
            "limitations": (
                "Short-sample extrapolation, not a full-epoch measurement or permission to train. "
                "Includes image loading and observed local checkpoint cost; excludes model setup, "
                "Drive backup, full-fold metric scaling, test inference and thermal drift. "
                "Smoke uses a shortened learning-rate schedule and external logging may differ."
            ),
        }


class TimedBudgetCheckpoint(ModelCheckpoint):
    """Time complete epochs and optionally enforce a runtime budget."""

    def __init__(
        self, *args, maximum_hours: float, projection_margin: float = 1.1,
        profile_only: bool = False, enforce_budget: bool = True,
        epoch_backup: Callable[[L.Trainer, Path, int], None] | None = None,
        **kwargs,
    ) -> None:
        # The backup and timing hooks run at validation end, so save the best
        # checkpoint there rather than letting Lightning defer it to train end.
        kwargs.setdefault("save_on_train_epoch_end", False)
        super().__init__(*args, **kwargs)
        self.maximum_hours = maximum_hours
        self.projection_margin = projection_margin
        self.profile_only = profile_only
        self.enforce_budget = enforce_budget
        self.epoch_backup = epoch_backup
        self.epoch_seconds: list[float] = []
        self.projected_seconds: float | None = None
        self.exceeded_budget = False
        self._epoch_started: float | None = None

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._epoch_started = time.perf_counter()

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        super().on_validation_end(trainer, pl_module)
        if trainer.sanity_checking or self._epoch_started is None:
            return
        if self.epoch_backup is not None:
            if not self.best_model_path:
                raise RuntimeError("Cannot back up an epoch without a best-model checkpoint.")
            self.epoch_backup(trainer, Path(self.best_model_path), trainer.current_epoch + 1)
        if pl_module.device.type == "cuda":
            torch.cuda.synchronize(pl_module.device)
        elapsed = time.perf_counter() - self._epoch_started
        self.epoch_seconds.append(elapsed)
        self._epoch_started = None
        if len(self.epoch_seconds) == 1:
            self.projected_seconds = elapsed * trainer.max_epochs * self.projection_margin
            self.exceeded_budget = self.projected_seconds > self.maximum_hours * 3600
            print(
                f"First complete epoch: {elapsed / 60:.1f} min; "
                f"projected {trainer.max_epochs}-epoch run with margin: "
                f"{self.projected_seconds / 3600:.2f} h."
            )
            if self.profile_only or (self.exceeded_budget and self.enforce_budget):
                if self.exceeded_budget and self.enforce_budget:
                    print("Over the runtime gate. Stop after this epoch and discuss a smaller run or another device.")
                else:
                    print("Profile-only run: stop after the first complete epoch.")
                trainer.should_stop = True
            elif self.exceeded_budget:
                print("Over the runtime estimate; budget enforcement is disabled. Continuing training.")


def copy_verified(source: Path, destination: Path) -> dict[str, str | int]:
    """Copy a finished artifact to Drive, verifying bytes before publishing it."""
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite Drive artifact: {destination}")
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists():
        raise FileExistsError(f"An incomplete Drive copy needs inspection: {temporary}")
    digest = hashlib.sha256()
    with source.open("rb") as original, temporary.open("xb") as copy:
        while chunk := original.read(8 * 1024 * 1024):
            digest.update(chunk)
            copy.write(chunk)
        copy.flush()
        os.fsync(copy.fileno())
    copied_digest = hashlib.sha256()
    with temporary.open("rb") as copy:
        while chunk := copy.read(8 * 1024 * 1024):
            copied_digest.update(chunk)
    if copied_digest.digest() != digest.digest():
        raise IOError(f"Drive copy failed checksum verification: {temporary}")
    os.replace(temporary, destination)
    return {"file": destination.name, "bytes": source.stat().st_size, "sha256": digest.hexdigest()}


class DriveRunBackup:
    """Keep epoch-level recovery files outside Colab's temporary VM."""

    def __init__(self, local_dir: Path, drive_dir: Path) -> None:
        self.local_dir = local_dir
        self.drive_dir = drive_dir
        self.last_best_stamp: tuple[int, int] | None = None

    def save_epoch(self, trainer: L.Trainer, best_model_path: Path, epoch: int) -> None:
        if epoch < 1:
            raise ValueError("Epoch numbers start at one.")
        resume_path = self.local_dir / "checkpoints" / f"resume_epoch_{epoch:02d}.ckpt"
        if resume_path.exists():
            raise FileExistsError(f"Refusing to overwrite resume checkpoint: {resume_path}")
        trainer.save_checkpoint(resume_path, weights_only=False)
        backed_up = [copy_verified(resume_path, self.drive_dir / "checkpoints" / resume_path.name)]
        best_stat = best_model_path.stat()
        best_stamp = (best_stat.st_mtime_ns, best_stat.st_size)
        if self.last_best_stamp != best_stamp:
            best_name = f"best_epoch_{epoch:02d}.ckpt"
            backed_up.append(copy_verified(best_model_path, self.drive_dir / "checkpoints" / best_name))
            self.last_best_stamp = best_stamp
        manifest = self.local_dir / f"backup_epoch_{epoch:02d}.json"
        manifest.write_text(json.dumps({"epoch": epoch, "files": backed_up}, indent=2) + "\n")
        copy_verified(manifest, self.drive_dir / manifest.name)
        print(f"Verified Drive backup for epoch {epoch}: {self.drive_dir}")

    def save_results(self, paths: list[Path]) -> None:
        for path in paths:
            copy_verified(path, self.drive_dir / "results" / path.name)
