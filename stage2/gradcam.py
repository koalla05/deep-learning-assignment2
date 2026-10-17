"""Class-specific GradCAM for a five-logit image model."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def gradcam_map(model: nn.Module, image: torch.Tensor, target_layer: nn.Module, label_index: int) -> torch.Tensor:
    """Return a [0, 1] heatmap for one image and one target logit."""
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("GradCAM expects a single NCHW image batch.")
    if not 0 <= label_index < 5:
        raise ValueError("label_index must be one of the five task labels.")
    activations: list[torch.Tensor] = []

    def capture(_module: nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
        if not isinstance(output, torch.Tensor) or output.ndim != 4:
            raise ValueError("GradCAM target must produce an NCHW feature map.")
        activations.append(output)

    hook = target_layer.register_forward_hook(capture)
    try:
        with torch.enable_grad():
            logits = model(image)
            if logits.shape != (1, 5) or len(activations) != 1:
                raise ValueError("Expected one five-logit prediction and one target activation.")
            gradients = torch.autograd.grad(logits[0, label_index], activations[0])[0]
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            heatmap = (weights * activations[0]).sum(dim=1, keepdim=True).relu()
            heatmap = F.interpolate(heatmap, size=image.shape[-2:], mode="bilinear", align_corners=False)
            heatmap = heatmap[0, 0].detach()
            peak = heatmap.max()
            return heatmap / peak if peak > 0 else torch.zeros_like(heatmap)
    finally:
        hook.remove()
