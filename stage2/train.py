"""Small Lightning adapter for five-logit image classifiers."""

from __future__ import annotations

from collections.abc import Iterable

import lightning as L
import numpy as np
import torch

from stage2.experiment import LABEL_COLUMNS, score_probabilities


class MultilabelClassifier(L.LightningModule):
    """Own optimization and logging; the supplied network owns the architecture."""

    def __init__(
        self,
        network: torch.nn.Module,
        learning_rate: float,
        weight_decay: float,
        threshold: float,
        frozen_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.network = network
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.threshold = threshold
        self.frozen_backbone = frozen_backbone
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.validation_targets: list[np.ndarray] = []
        self.validation_probabilities: list[np.ndarray] = []

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.network(images)
        if logits.ndim != 2 or logits.shape[1] != len(LABEL_COLUMNS):
            raise ValueError("The network must return five logits per image.")
        return logits

    def on_train_epoch_start(self) -> None:
        if self.frozen_backbone:
            # Lightning switches the whole module to train mode each epoch.
            self.network.eval()
            self.network.get_classifier().train()

    def training_step(self, batch: tuple, batch_index: int) -> torch.Tensor:
        _, images, targets = batch
        loss = self.loss_fn(self(images), targets)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(images))
        return loss

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        gradients = [parameter.grad.detach().float().norm(2) for parameter in self.parameters() if parameter.grad is not None]
        if gradients:
            norm = torch.linalg.vector_norm(torch.stack(gradients)).item()
            self.log("gradient_norm", norm, on_step=True, on_epoch=True)

    def on_validation_epoch_start(self) -> None:
        self.validation_targets.clear()
        self.validation_probabilities.clear()

    def validation_step(self, batch: tuple, batch_index: int) -> None:
        _, images, targets = batch
        logits = self(images)
        loss = self.loss_fn(logits, targets)
        self.log("val_loss", loss, on_step=False, on_epoch=True, batch_size=len(images))
        self.validation_targets.append(targets.detach().cpu().numpy())
        self.validation_probabilities.append(logits.sigmoid().detach().cpu().numpy())

    def on_validation_epoch_end(self) -> None:
        targets = np.concatenate(self.validation_targets)
        probabilities = np.concatenate(self.validation_probabilities)
        metrics = score_probabilities(targets, probabilities, self.threshold)
        self.log("val_macro_f1", metrics["macro_f1"], prog_bar=True)
        if metrics["macro_ap"] is not None:
            self.log("val_macro_ap", metrics["macro_ap"])
        for label, value in metrics["per_label_f1"].items():
            self.log(f"val_f1_{label.lower()}", value)
        self.validation_targets.clear()
        self.validation_probabilities.clear()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        trainable = [parameter for parameter in self.network.parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError("The model has no trainable parameters.")
        return torch.optim.AdamW(trainable, lr=self.learning_rate, weight_decay=self.weight_decay)


def freeze_except_classifier(network: torch.nn.Module) -> None:
    """The task-specific classifier is the only trainable part of a head run."""
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    classifier = network.get_classifier()
    if not isinstance(classifier, torch.nn.Module):
        raise TypeError("This model does not expose a module classifier.")
    for parameter in classifier.parameters():
        parameter.requires_grad_(True)


@torch.inference_mode()
def predict_probabilities(module: MultilabelClassifier, batches: Iterable[tuple], device: torch.device) -> tuple[list[str], np.ndarray]:
    """Keep IDs in loader order; return probabilities without model-specific code."""
    module.to(device).eval()
    image_ids: list[str] = []
    probabilities: list[np.ndarray] = []
    for ids, images, _ in batches:
        logits = module(images.to(device))
        image_ids.extend(ids)
        probabilities.append(logits.sigmoid().cpu().numpy())
    if not probabilities:
        raise ValueError("Cannot predict an empty dataset.")
    return image_ids, np.concatenate(probabilities)
