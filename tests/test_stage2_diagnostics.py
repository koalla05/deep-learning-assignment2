"""Real Lightning gradient summaries and correctly scoped memory measurements."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import lightning as L
import torch

from stage2.diagnostics import TrainingDiagnostics
from stage2.finetune import FineTuneClassifier
from test_stage2_finetune import TinyNetwork


class DiagnosticsTests(unittest.TestCase):
    def test_real_fit_saves_layer_norms_block_curve_and_memory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            network = TinyNetwork()
            diagnostics = TrainingDiagnostics(Path(directory), {"block": network.backbone})
            module = FineTuneClassifier(network, 1e-3, 1e-2, 1e-2, 0.25)
            examples = [
                (str(i), torch.ones(4) * (i + 1), torch.tensor([i % 2] * 5, dtype=torch.float32))
                for i in range(8)
            ]
            loader = torch.utils.data.DataLoader(examples, batch_size=2)
            trainer = L.Trainer(
                accelerator="cpu", devices=1, max_epochs=2, accumulate_grad_batches=2,
                callbacks=[diagnostics], logger=False, enable_checkpointing=False,
                enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0,
            )
            trainer.fit(module, loader, loader)
            with (Path(directory) / "layer_gradients.csv").open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual({int(row["epoch"]) for row in rows}, {0, 1})
            self.assertEqual({row["layer"] for row in rows}, {"", "backbone.0", "backbone.1", "head"})
            self.assertTrue(all(int(row["updates"]) == 2 for row in rows))
            self.assertTrue(all(0 <= float(row["mean_l2_norm"]) <= float(row["max_l2_norm"]) + 1e-6 for row in rows))
            self.assertIn("gradient_norm_block_epoch", trainer.callback_metrics)
            memory = json.loads((Path(directory) / "memory.json").read_text())
            self.assertGreater(memory["process_lifetime_peak_rss_bytes"], 0)
            self.assertGreater(memory["system_memory_bytes"], 0)
            self.assertIsNone(memory["cuda_peak_allocated_bytes"])
            self.assertIsNone(memory["mps_sampled_allocated_high_water_bytes"])

    def test_mps_high_water_is_sampled_and_never_mislabeled_as_peak(self) -> None:
        diagnostics = TrainingDiagnostics(Path("unused"))
        module = SimpleNamespace(device=torch.device("mps"))
        with patch("torch.mps.current_allocated_memory", side_effect=[100, 80, 150]), patch(
            "torch.mps.driver_allocated_memory", side_effect=[200, 240, 220]
        ):
            diagnostics.on_train_start(None, module)
            diagnostics.on_train_batch_end(None, module, None, None, 0)
            diagnostics.on_validation_batch_end(SimpleNamespace(sanity_checking=False), module, None, None, 0)
        self.assertEqual(diagnostics.memory["mps_sampled_allocated_high_water_bytes"], 150)
        self.assertEqual(diagnostics.memory["mps_sampled_driver_high_water_bytes"], 240)
        self.assertIn("not a true allocator peak", diagnostics.memory["scope"])


if __name__ == "__main__":
    unittest.main()
