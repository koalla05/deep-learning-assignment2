"""BatchNorm statistics stay fixed during training; affine parameters still learn."""

from __future__ import annotations

import unittest

import torch

from stage2.batchnorm import FrozenBatchNormStatistics, freeze_batchnorm_statistics


def small_network() -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.BatchNorm2d(4), torch.nn.ReLU())


class BatchNormFreezeTests(unittest.TestCase):
    def test_running_statistics_do_not_move_but_scale_still_gets_gradients(self) -> None:
        network = small_network().train()
        batch_norm = network[1]
        before = batch_norm.running_mean.clone(), batch_norm.running_var.clone()
        self.assertEqual(freeze_batchnorm_statistics(network), 1)
        self.assertFalse(batch_norm.training)
        network(torch.randn(8, 3, 6, 6)).sum().backward()
        self.assertTrue(torch.equal(before[0], batch_norm.running_mean))
        self.assertTrue(torch.equal(before[1], batch_norm.running_var))
        self.assertIsNotNone(batch_norm.weight.grad)

    def test_callback_rejects_a_network_without_batchnorm(self) -> None:
        class Module:
            network = torch.nn.Linear(2, 2)

        with self.assertRaisesRegex(ValueError, "no BatchNorm"):
            FrozenBatchNormStatistics().on_train_epoch_start(None, Module())


if __name__ == "__main__":
    unittest.main()
