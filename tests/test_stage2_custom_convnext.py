"""Architectural contracts for the custom ConvNeXt experiment."""

from __future__ import annotations

import unittest

import torch
from torch import nn
import timm

from stage2.custom_convnext import CustomConvNeXt, ResidualSpatialBlock
from stage2.finetune import fine_tuning_groups


class TinySpatialBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_features = 8
        self.stages = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1))
        self.head = nn.Linear(8, 5)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.stages(images)

    def forward_head(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features.mean(dim=(-2, -1)))

    def get_classifier(self) -> nn.Module:
        return self.head

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward_head(self.forward_features(images))


class CustomConvNeXtTests(unittest.TestCase):
    def test_zero_initialized_block_preserves_pretrained_logits(self) -> None:
        backbone = TinySpatialBackbone()
        images = torch.randn(2, 3, 8, 8)
        expected = backbone(images)
        custom = CustomConvNeXt(backbone, reduction=4)
        torch.testing.assert_close(custom(images), expected, rtol=0, atol=0)
        self.assertIs(custom.get_classifier(), backbone.head)
        self.assertIs(custom.gradcam_target_layer, backbone.stages[-1])

    def test_block_receives_gradients_after_first_update(self) -> None:
        custom = CustomConvNeXt(TinySpatialBackbone())
        images = torch.randn(2, 3, 8, 8)
        custom(images).sum().backward()
        self.assertIsNotNone(custom.block.restore.weight.grad)
        self.assertGreater(custom.block.restore.weight.grad.abs().sum().item(), 0)
        self.assertTrue(torch.equal(custom.block.reduce.weight.grad, torch.zeros_like(custom.block.reduce.weight.grad)))
        optimizer = torch.optim.AdamW(custom.parameters(), lr=1e-3)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        custom(images).sum().backward()
        for parameter in (custom.block.reduce.weight, custom.block.depthwise.weight):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_optimizer_groups_cover_new_block_and_classifier(self) -> None:
        custom = CustomConvNeXt(TinySpatialBackbone())
        groups = fine_tuning_groups(custom, 1e-5, 1e-4, 1e-2)
        assignment = {id(parameter): (group["lr"], group["weight_decay"]) for group in groups for parameter in group["params"]}
        self.assertEqual(len(assignment), len(list(custom.parameters())))
        self.assertEqual(assignment[id(custom.block.restore.weight)], (1e-5, 1e-2))
        self.assertEqual(assignment[id(custom.get_classifier().weight)], (1e-4, 1e-2))

    def test_invalid_backbone_or_block_dimensions_fail_early(self) -> None:
        with self.assertRaises(TypeError):
            CustomConvNeXt(nn.Linear(2, 2))
        with self.assertRaises(ValueError):
            ResidualSpatialBlock(8, reduction=0)

    def test_timm_convnext_feature_and_head_seam(self) -> None:
        backbone = timm.create_model("convnext_tiny", pretrained=False, num_classes=5).eval()
        custom = CustomConvNeXt(backbone).eval()
        image = torch.randn(1, 3, 64, 64)
        with torch.inference_mode():
            torch.testing.assert_close(custom(image), backbone(image), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
