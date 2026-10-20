"""Keep pretrained BatchNorm running statistics fixed while the rest of a CNN trains."""

from __future__ import annotations

import lightning as L
import torch
from torch.nn.modules.batchnorm import _BatchNorm


def freeze_batchnorm_statistics(network: torch.nn.Module) -> int:
    """Put every BatchNorm layer in eval mode; scale and shift stay trainable.

    Small physical batches give noisy batch statistics, so the pretrained running
    mean and variance are used in training as well as in validation.
    """
    layers = [module for module in network.modules() if isinstance(module, _BatchNorm)]
    for layer in layers:
        layer.eval()
    return len(layers)


class FrozenBatchNormStatistics(L.Callback):
    """Lightning switches the whole module to train mode each epoch; undo that for BatchNorm."""

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if freeze_batchnorm_statistics(pl_module.network) == 0:
            raise ValueError("This network has no BatchNorm layers to freeze.")
