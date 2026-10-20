"""Recovery contracts for HRNet prediction artifacts."""

from __future__ import annotations

import tempfile
import unittest
import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch
import time

import lightning as L
import numpy as np
import torch

from stage2.hrnet_finetune import (
    Config, EpochBackup, HRNetClassifier, SelectedCheckpoint, StopAfterSelection,
    export_saved_run, write_matrix,
)


class TinyNetwork(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Linear(4, 4)
        self.head = torch.nn.Linear(4, 5)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(images))

    def get_classifier(self) -> torch.nn.Module:
        return self.head


class HRNetRecoveryTests(unittest.TestCase):
    def test_epoch_state_contains_selection_and_per_label_bce(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "checkpoints").mkdir()
            examples = [
                (str(index), torch.tensor([float(index % 3)] * 4),
                 torch.tensor([float((index + label) % 2) for label in range(5)]))
                for index in range(20)
            ]
            loader = torch.utils.data.DataLoader(examples, batch_size=4)
            module = HRNetClassifier(TinyNetwork(), 1e-3, 1e-2, 1e-4, 0.25)
            started = time.monotonic()
            selected = SelectedCheckpoint(run_dir, started)
            trainer = L.Trainer(
                accelerator="cpu", devices=1, max_epochs=2, logger=False,
                callbacks=[selected, EpochBackup(run_dir, selected, None),
                           StopAfterSelection(selected, Config(), started)],
                enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
            )
            trainer.fit(module, loader, loader)
            state = torch.load(run_dir / "checkpoints" / "resume_epoch_02.ckpt", weights_only=False)
            selection = state["callbacks"]["hrnet_selected_checkpoint"]
            self.assertEqual(len(selection["history"]), 2)
            self.assertIn("bce_Moris", selection["history"][-1])
            self.assertTrue((run_dir / "checkpoints" / f"best_epoch_{selection['best_epoch']:02d}.ckpt").is_file())

    def test_prediction_file_can_be_reused_only_with_matching_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "validation_probabilities.csv"
            values = np.full((2, 5), 0.25)
            write_matrix(path, ["first", "second"], values, fold=0)
            original = path.read_bytes()
            write_matrix(path, ["first", "second"], values, fold=0)
            self.assertEqual(path.read_bytes(), original)
            with self.assertRaises(ValueError):
                write_matrix(path, ["first", "second"], values + 0.01, fold=0)
            with self.assertRaises(ValueError):
                write_matrix(path, ["second", "first"], values, fold=0)

    def test_export_only_uses_best_epoch_in_resume_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = Config()
            run_dir = root / "artifacts" / "hrnet_finetune" / config.run_id
            checkpoints = run_dir / "checkpoints"
            checkpoints.mkdir(parents=True)
            (run_dir / "run.json").write_text(json.dumps({
                "config": asdict(config), "head_checkpoint_sha256": "head",
            }))
            selected = checkpoints / "best_epoch_01.ckpt"
            selected.write_bytes(b"selected checkpoint")
            resume = checkpoints / "resume_epoch_02.ckpt"
            torch.save({"epoch": 1, "callbacks": {"hrnet_selected_checkpoint": {
                "best_epoch": 1, "history": [{"epoch": 1, "macro_f1": 0.8},
                                              {"epoch": 2, "macro_f1": 0.7}],
            }}}, resume)
            with patch("stage2.hrnet_finetune.torch.cuda.is_available", return_value=True), \
                 patch("stage2.hrnet_finetune.preflight_inputs", return_value=([], [], {})), \
                 patch("stage2.hrnet_finetune.finish_exports", return_value=run_dir) as finish:
                self.assertEqual(export_saved_run(root, root / "head", None, config, resume), run_dir)
            self.assertEqual(finish.call_args.args[5], selected)
            self.assertEqual(finish.call_args.args[6], 1)
            self.assertTrue((run_dir / "history.csv").is_file())

    def test_full_resume_restores_selection_before_next_backup(self) -> None:
        class StopAfterFirst(L.Callback):
            def on_validation_end(self, trainer, module) -> None:
                if not trainer.sanity_checking:
                    trainer.should_stop = True

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "checkpoints").mkdir()
            examples = [
                (str(index), torch.ones(4) * (index % 3),
                 torch.tensor([float((index + label) % 2) for label in range(5)]))
                for index in range(20)
            ]
            loader = torch.utils.data.DataLoader(examples, batch_size=4)
            for stopping in (True, False):
                started = time.monotonic()
                selected = SelectedCheckpoint(run_dir, started)
                callbacks = [selected, EpochBackup(run_dir, selected, None),
                             StopAfterSelection(selected, Config(), started)]
                if stopping:
                    callbacks.append(StopAfterFirst())
                trainer = L.Trainer(
                    accelerator="cpu", devices=1, max_epochs=2, logger=False,
                    callbacks=callbacks, enable_checkpointing=False,
                    enable_progress_bar=False, enable_model_summary=False,
                )
                checkpoint = run_dir / "checkpoints" / "resume_epoch_01.ckpt" if not stopping else None
                trainer.fit(HRNetClassifier(TinyNetwork(), 1e-3, 1e-2, 1e-4, 0.25),
                            loader, loader, ckpt_path=str(checkpoint) if checkpoint else None)
            self.assertEqual([row["epoch"] for row in selected.history], [1, 2])
            self.assertTrue((run_dir / "checkpoints" / "resume_epoch_02.ckpt").is_file())


if __name__ == "__main__":
    unittest.main()
