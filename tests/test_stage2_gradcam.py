"""Check class-specific GradCAM without pretrained downloads."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from stage2.gradcam import gradcam_map


class TinyImageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Conv2d(3, 4, 3, padding=1)
        self.classifier = nn.Linear(4, 5)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images).relu().mean(dim=(-2, -1)))


class GradCAMTests(unittest.TestCase):
    def test_returns_finite_image_sized_heatmap_without_parameter_updates(self) -> None:
        model = TinyImageModel().eval()
        original = model.features.weight.detach().clone()
        heatmap = gradcam_map(model, torch.randn(1, 3, 12, 10), model.features, 2)
        self.assertEqual(tuple(heatmap.shape), (12, 10))
        self.assertTrue(torch.isfinite(heatmap).all())
        self.assertGreaterEqual(heatmap.min().item(), 0)
        self.assertLessEqual(heatmap.max().item(), 1)
        torch.testing.assert_close(model.features.weight, original)

    def test_rejects_invalid_batch_or_label(self) -> None:
        model = TinyImageModel()
        with self.assertRaises(ValueError):
            gradcam_map(model, torch.randn(2, 3, 8, 8), model.features, 0)
        with self.assertRaises(ValueError):
            gradcam_map(model, torch.randn(1, 3, 8, 8), model.features, 5)


if __name__ == "__main__":
    unittest.main()
