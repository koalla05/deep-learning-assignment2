"""Local, bounded training diagnostics independent of model architecture."""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path
import platform
import resource

import lightning as L
import torch

from stage2.experiment import write_run_record


def plot_gradient_flow(csv_path: Path, image_path: Path) -> Path:
    """Plot layer-wise gradient norms across epochs from a training diagnostics CSV."""
    import matplotlib.pyplot as plt
    import numpy as np

    with csv_path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("No layer-gradient measurements to plot.")
    epochs = sorted({int(row["epoch"]) for row in rows})
    layers = list(dict.fromkeys(row["layer"] for row in rows))
    values = np.full((len(epochs), len(layers)), np.nan)
    epoch_index = {epoch: index for index, epoch in enumerate(epochs)}
    layer_index = {layer: index for index, layer in enumerate(layers)}
    for row in rows:
        norm = float(row["mean_l2_norm"])
        if not math.isfinite(norm) or norm < 0:
            raise ValueError("Layer-gradient norms must be finite and nonnegative.")
        values[epoch_index[int(row["epoch"])], layer_index[row["layer"]]] = math.log10(max(norm, 1e-12))
    figure, axis = plt.subplots(figsize=(max(10, min(20, len(layers) * 0.12)), 3 + len(epochs) * 0.35))
    heatmap = axis.imshow(values, aspect="auto", cmap="viridis", interpolation="nearest")
    axis.set(xlabel="Layers in model order", ylabel="Epoch", title="Mean pre-clipping gradient norm (log10 L2)")
    axis.set_yticks(range(len(epochs)), [str(epoch + 1) for epoch in epochs])
    ticks = sorted(set(np.linspace(0, len(layers) - 1, min(12, len(layers)), dtype=int)))
    axis.set_xticks(ticks, [layers[index] for index in ticks], rotation=70, ha="right", fontsize=7)
    figure.colorbar(heatmap, ax=axis, label="log10 gradient L2 norm")
    figure.tight_layout()
    if image_path.exists():
        plt.close(figure)
        raise FileExistsError(f"Refusing to overwrite {image_path}")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(image_path, dpi=140)
    plt.close(figure)
    return image_path


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
