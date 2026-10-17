"""Local, bounded training diagnostics independent of model architecture."""

from __future__ import annotations

import csv
import os
from pathlib import Path
import platform
import resource

import lightning as L
import torch

from stage2.experiment import write_run_record


class TrainingDiagnostics(L.Callback):
    """Save per-layer gradient summaries and explicitly scoped memory measurements."""

    def __init__(self, directory: Path, gradient_groups: dict[str, torch.nn.Module] | None = None) -> None:
        super().__init__()
        self.directory = directory
        self.gradient_groups = gradient_groups or {}
        self.layer_summaries: dict[str, tuple[torch.Tensor, torch.Tensor, int]] = {}
        self.memory: dict = {}

    def on_train_start(self, trainer, pl_module) -> None:
        if pl_module.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(pl_module.device)
        self.memory = {
            "device": str(pl_module.device),
            "system_memory_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
            "mps_sampled_allocated_high_water_bytes": None,
            "mps_sampled_driver_high_water_bytes": None,
            "cuda_peak_allocated_bytes": None,
            "process_lifetime_peak_rss_bytes": None,
            "scope": "Training and validation; excludes model setup and final inference. "
                     "MPS is sampled at batch/epoch boundaries, not a true allocator peak. "
                     "RSS is the process-lifetime high-water mark, not isolated to this run.",
        }
        self._sample_memory(pl_module.device)

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self.layer_summaries.clear()

    def on_before_optimizer_step(self, trainer, pl_module, optimizer) -> None:
        for name, layer in pl_module.network.named_modules():
            gradients = [
                parameter.grad.detach().float().norm()
                for parameter in layer.parameters(recurse=False) if parameter.grad is not None
            ]
            if not gradients:
                continue
            norm = torch.linalg.vector_norm(torch.stack(gradients))
            if name in self.layer_summaries:
                total, maximum, count = self.layer_summaries[name]
                self.layer_summaries[name] = (total + norm, torch.maximum(maximum, norm), count + 1)
            else:
                self.layer_summaries[name] = (norm, norm, 1)
        for name, group in self.gradient_groups.items():
            gradients = [p.grad.detach().float().norm() for p in group.parameters() if p.grad is not None]
            if gradients:
                pl_module.log(
                    f"gradient_norm_{name}", torch.linalg.vector_norm(torch.stack(gradients)),
                    on_step=True, on_epoch=True,
                )

    def _sample_memory(self, device: torch.device) -> None:
        if device.type == "mps":
            for key, value in (
                ("mps_sampled_allocated_high_water_bytes", torch.mps.current_allocated_memory()),
                ("mps_sampled_driver_high_water_bytes", torch.mps.driver_allocated_memory()),
            ):
                self.memory[key] = max(self.memory[key] or 0, value)
        elif device.type == "cuda":
            self.memory["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self.memory["process_lifetime_peak_rss_bytes"] = rss if platform.system() == "Darwin" else rss * 1024

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        self._sample_memory(pl_module.device)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        if not trainer.sanity_checking:
            self._sample_memory(pl_module.device)

    def on_validation_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / "layer_gradients.csv"
        new_file = not path.exists()
        with path.open("a", newline="") as file:
            writer = csv.writer(file)
            if new_file:
                writer.writerow(("epoch", "layer", "updates", "mean_l2_norm", "max_l2_norm"))
            for name, (total, maximum, count) in self.layer_summaries.items():
                values = torch.stack((total / count, maximum)).cpu()
                if not torch.isfinite(values).all():
                    raise ValueError(f"Nonfinite gradient statistics in layer {name}.")
                writer.writerow((trainer.current_epoch, name, count, *values.tolist()))
        self._sample_memory(pl_module.device)

    def on_train_end(self, trainer, pl_module) -> None:
        self._sample_memory(pl_module.device)
        write_run_record(self.directory / "memory.json", self.memory)
