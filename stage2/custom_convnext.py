"""The single architectural change used in the matched ConvNeXt experiment."""

from __future__ import annotations

import torch
from torch import nn


class ResidualSpatialBlock(nn.Module):
    """A narrow spatial update, initialized to leave its input unchanged."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        if channels < 1 or reduction < 1:
            raise ValueError("channels and reduction must be positive.")
        hidden = max(1, channels // reduction)
        self.reduce = nn.Conv2d(channels, hidden, kernel_size=1)
        self.depthwise = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.restore = nn.Conv2d(hidden, channels, kernel_size=1)
        self.activation = nn.GELU()
        nn.init.zeros_(self.restore.weight)
        nn.init.zeros_(self.restore.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        update = self.activation(self.reduce(features))
        update = self.activation(self.depthwise(update))
        return features + self.restore(update)


class CustomConvNeXt(nn.Module):
    """Insert one residual block between ConvNeXt's spatial map and head."""

    def __init__(self, backbone: nn.Module, reduction: int = 4) -> None:
        super().__init__()
        if not all(callable(getattr(backbone, name, None)) for name in ("forward_features", "forward_head", "get_classifier")):
            raise TypeError("Expected a ConvNeXt with forward_features, forward_head and get_classifier.")
        channels = getattr(backbone, "num_features", None)
        if not isinstance(channels, int) or channels < 1:
            raise ValueError("The backbone must expose a positive num_features.")
        self.backbone = backbone
        self.block = ResidualSpatialBlock(channels, reduction)

    def get_classifier(self) -> nn.Module:
        return self.backbone.get_classifier()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(images)
        if features.ndim != 4 or features.shape[1] != self.backbone.num_features:
            raise ValueError("Expected a channels-first ConvNeXt spatial feature map.")
        return self.backbone.forward_head(self.block(features))

    @property
    def gradcam_target_layer(self) -> nn.Module:
        """Use the last pretrained spatial stage for comparable GradCAMs."""
        return self.backbone.stages[-1]
