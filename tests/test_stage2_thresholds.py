"""Threshold choice only where held-out data support it, and fixed-length training."""

from __future__ import annotations

import contextlib
import io
import unittest

import numpy as np
import torch

from stage2.experiment import LABEL_COLUMNS
from stage2.thresholds import best_plateau, choose_thresholds


class ThresholdChoiceTests(unittest.TestCase):
    def test_well_supported_label_is_tuned_and_rare_label_keeps_default(self) -> None:
        rng = np.random.default_rng(1)
        n = 600
        targets = np.zeros((n, 5), dtype=int)
        targets[:300, 0] = 1  # Plenty of positives: tunable.
        targets[:3, 2] = 1  # Three positives: not tunable.
        probabilities = np.clip(targets * 0.6 + rng.uniform(0, 0.4, (n, 5)), 0, 1)
        choices = {choice.label: choice for choice in choose_thresholds(targets, probabilities, min_positives=100, bootstrap=20)}
        self.assertTrue(choices["Cat"].tuned)
        self.assertGreater(choices["Cat"].threshold, 0.25)
        self.assertGreaterEqual(choices["Cat"].f1_chosen, choices["Cat"].f1_default)
        self.assertIsNotNone(choices["Cat"].bootstrap_interval)
        self.assertFalse(choices[LABEL_COLUMNS[2]].tuned)
        self.assertEqual(choices[LABEL_COLUMNS[2]].threshold, 0.25)
        self.assertIsNone(choices[LABEL_COLUMNS[2]].bootstrap_interval)

    def test_plateau_is_every_threshold_with_the_best_f1(self) -> None:
        targets = np.array([0, 0, 1, 1])
        probabilities = np.array([0.1, 0.2, 0.8, 0.9])
        plateau = best_plateau(targets, probabilities)
        self.assertAlmostEqual(float(plateau.min()), 0.25)
        self.assertAlmostEqual(float(plateau.max()), 0.8)

    def test_rejects_misaligned_inputs(self) -> None:
        with self.assertRaises(ValueError):
            choose_thresholds(np.zeros((4, 5)), np.zeros((4, 4)))


class StopAfterEpochsTests(unittest.TestCase):
    def test_stops_after_the_fixed_epochs_but_keeps_the_longer_schedule(self) -> None:
        import lightning as L

        from stage2.finetune import StopAfterEpochs
        from stage2.train import MultilabelClassifier

        torch.manual_seed(0)
        network = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 5))
        from torch.utils.data import DataLoader

        rows = [(f"{i}", torch.randn(3, 2, 2), torch.zeros(5)) for i in range(4)]
        batches = DataLoader(rows, batch_size=1)
        module = MultilabelClassifier(network, 1e-3, 0.0, 0.25)
        stopper = StopAfterEpochs(2)
        trainer = L.Trainer(accelerator="cpu", max_epochs=8, callbacks=[stopper], logger=False,
                            enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            trainer.fit(module, batches)
        self.assertEqual(trainer.current_epoch, 2)
        self.assertEqual(trainer.max_epochs, 8)
        self.assertEqual(len(stopper.epoch_seconds), 2)
        with self.assertRaises(ValueError):
            StopAfterEpochs(0)


if __name__ == "__main__":
    unittest.main()
