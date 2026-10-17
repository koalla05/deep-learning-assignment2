"""Optimizer and checkpoint contracts for the standard ConvNeXt run."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from stage2.finetune import DriveRunBackup, FineTuneClassifier, SmokeRuntimeProfile, TimedBudgetCheckpoint, copy_verified, fine_tuning_groups, load_trained_head, warmup_cosine_multiplier


class TinyNetwork(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))
        self.layer_scale = torch.nn.Parameter(torch.ones(4))
        self.head = torch.nn.Linear(4, 5)

    def get_classifier(self) -> torch.nn.Module:
        return self.head

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(images) * self.layer_scale)


class FineTuneTests(unittest.TestCase):
    def test_all_parameters_have_one_lr_and_correct_decay(self) -> None:
        network = TinyNetwork()
        groups = fine_tuning_groups(network, backbone_lr=1e-5, head_lr=1e-4, weight_decay=1e-2)
        by_parameter = {id(parameter): (group["lr"], group["weight_decay"]) for group in groups for parameter in group["params"]}
        self.assertEqual(len(by_parameter), len(list(network.parameters())))
        for name, parameter in network.named_parameters():
            expected_lr = 1e-4 if name.startswith("head.") else 1e-5
            expected_decay = 0.0 if name.endswith("bias") or name.startswith("backbone.1.") else 1e-2
            self.assertEqual(by_parameter[id(parameter)], (expected_lr, expected_decay))

    def test_head_checkpoint_is_loaded_strictly(self) -> None:
        network = TinyNetwork()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "head.pt"
            expected = {name: torch.ones_like(value) for name, value in network.head.state_dict().items()}
            torch.save(expected, path)
            load_trained_head(network, path)
            self.assertTrue(all(torch.equal(value, expected[name]) for name, value in network.head.state_dict().items()))
            torch.save({"wrong": torch.ones(1)}, path)
            with self.assertRaises(RuntimeError):
                load_trained_head(network, path)

    def test_warmup_and_decay_reach_expected_values(self) -> None:
        self.assertAlmostEqual(warmup_cosine_multiplier(0, 10, 100), 0.1)
        self.assertAlmostEqual(warmup_cosine_multiplier(9, 10, 100), 1.0)
        self.assertAlmostEqual(warmup_cosine_multiplier(99, 10, 100), 0.1)
        self.assertAlmostEqual(warmup_cosine_multiplier(20, 20, 20), 1.0)

    def test_profile_stops_after_first_full_epoch_without_changing_projection(self) -> None:
        checkpoint = TimedBudgetCheckpoint(maximum_hours=5, profile_only=True)
        checkpoint._epoch_started = 100.0
        trainer = SimpleNamespace(sanity_checking=False, max_epochs=10, should_stop=False)
        module = SimpleNamespace(device=torch.device("cpu"))
        with patch.object(ModelCheckpoint, "on_validation_end"), patch(
            "stage2.finetune.time.perf_counter", return_value=220.0
        ):
            checkpoint.on_validation_end(trainer, module)
        self.assertEqual(checkpoint.epoch_seconds, [120.0])
        self.assertAlmostEqual(checkpoint.projected_seconds, 1320.0)
        self.assertFalse(checkpoint.exceeded_budget)
        self.assertTrue(trainer.should_stop)

    def test_verified_copy_does_not_replace_existing_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.txt"
            destination = root / "drive" / "result.txt"
            source.write_text("complete result")
            record = copy_verified(source, destination)
            self.assertEqual(destination.read_text(), source.read_text())
            self.assertEqual(record["bytes"], len(source.read_bytes()))
            self.assertFalse(destination.with_name("result.txt.partial").exists())
            with self.assertRaises(FileExistsError):
                copy_verified(source, destination)

    def test_epoch_backup_includes_resume_state_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local = root / "local"
            drive = root / "drive"
            best = local / "checkpoints" / "best.ckpt"
            best.parent.mkdir(parents=True)
            best.write_bytes(b"best model")
            backup = DriveRunBackup(local, drive)

            def save_checkpoint(path: Path, weights_only: bool) -> None:
                self.assertFalse(weights_only)
                path.write_bytes(b"model and optimizer")

            trainer = SimpleNamespace(save_checkpoint=save_checkpoint)
            backup.save_epoch(trainer, best, 1)
            self.assertEqual((drive / "checkpoints" / "resume_epoch_01.ckpt").read_bytes(), b"model and optimizer")
            self.assertEqual((drive / "checkpoints" / "best_epoch_01.ckpt").read_bytes(), b"best model")
            self.assertTrue((drive / "backup_epoch_01.json").is_file())
            backup.save_epoch(trainer, best, 2)
            self.assertTrue((drive / "checkpoints" / "resume_epoch_02.ckpt").is_file())
            self.assertFalse((drive / "checkpoints" / "best_epoch_02.ckpt").exists())

    def test_lightning_writes_full_epoch_checkpoint_inside_callback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            network = TinyNetwork()
            module = FineTuneClassifier(network, 1e-3, 1e-2, 1e-2, 0.25)
            backup = DriveRunBackup(root / "local", root / "drive")
            checkpoint = TimedBudgetCheckpoint(
                dirpath=root / "local" / "checkpoints", filename="best",
                monitor="val_macro_f1", mode="max", save_top_k=1,
                save_weights_only=True, maximum_hours=5, profile_only=True,
                epoch_backup=backup.save_epoch,
            )
            examples = [
                (str(index), torch.ones(4) * (index + 1), torch.tensor([index % 2] * 5, dtype=torch.float32))
                for index in range(4)
            ]
            loader = torch.utils.data.DataLoader(examples, batch_size=2)
            trainer = L.Trainer(
                accelerator="cpu", devices=1, max_epochs=2, callbacks=[checkpoint],
                logger=False, enable_checkpointing=True, enable_progress_bar=False,
                enable_model_summary=False, num_sanity_val_steps=0,
            )
            trainer.fit(module, loader, loader)
            resume = root / "drive" / "checkpoints" / "resume_epoch_01.ckpt"
            self.assertTrue(resume.is_file())
            state = torch.load(resume, map_location="cpu", weights_only=False)
            self.assertIn("optimizer_states", state)
            self.assertIn("lr_schedulers", state)
            self.assertTrue(state["optimizer_states"])

    def test_smoke_profile_excludes_warmup_and_projects_final_validation(self) -> None:
        profile = SmokeRuntimeProfile()
        trainer = SimpleNamespace(accumulate_grad_batches=2)
        module = SimpleNamespace(device=torch.device("cpu"))
        batch = (["a", "b"], torch.ones(2, 4), torch.ones(2, 5))
        with patch("stage2.finetune.time.perf_counter", side_effect=[0, 20, 30, 31, 33, 34, 36, 40, 50, 51, 52, 55]):
            profile.on_train_epoch_start(trainer, module)
            for index in range(6):
                profile.on_train_batch_end(trainer, module, {"loss": torch.tensor(1.0)}, batch, index)
            profile.on_validation_epoch_start(trainer, module)
            for index in range(3):
                profile.on_validation_batch_end(trainer, module, None, batch, index)
            profile.on_train_end(trainer, module)
        estimate = profile.estimate(80, 20, effective_batch_size=4, planned_epochs=8)
        self.assertEqual(estimate["measured_training_images"], 8)
        self.assertEqual(estimate["measured_validation_images"], 4)
        self.assertEqual(estimate["train_seconds_per_effective_batch"], 3)
        self.assertEqual(estimate["estimated_train_epoch_seconds"], 60)
        self.assertEqual(estimate["estimated_validation_epoch_seconds"], 10)
        self.assertEqual(estimate["validation_metrics_and_checkpoint_seconds"], 3)
        self.assertAlmostEqual(estimate["estimated_run_hours_with_margin"], (8 * 73 + 10) * 1.1 / 3600)

    def test_smoke_profile_rejects_missing_or_incomplete_measurements(self) -> None:
        profile = SmokeRuntimeProfile()
        with self.assertRaises(RuntimeError):
            profile.estimate(80, 20, 4, 8)
        profile.train_images = 6
        profile.validation_images = 4
        profile.validation_tail_seconds = 1
        with self.assertRaises(RuntimeError):
            profile.estimate(80, 20, 4, 8)
        with self.assertRaises(ValueError):
            profile.estimate(0, 20, 4, 8)

    def test_smoke_profile_observes_real_lightning_checkpoint_tail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profile = SmokeRuntimeProfile()
            checkpoint = ModelCheckpoint(
                dirpath=directory, monitor="val_macro_f1", mode="max",
                save_weights_only=True, save_on_train_epoch_end=False,
            )
            examples = [
                (str(index), torch.ones(4), torch.tensor([index % 2] * 5, dtype=torch.float32))
                for index in range(12)
            ]
            loader = torch.utils.data.DataLoader(examples, batch_size=2)
            trainer = L.Trainer(
                accelerator="cpu", devices=1, max_epochs=1,
                accumulate_grad_batches=2, callbacks=[checkpoint, profile],
                logger=False, enable_progress_bar=False, enable_model_summary=False,
                num_sanity_val_steps=0,
            )
            trainer.fit(FineTuneClassifier(TinyNetwork(), 1e-3, 1e-2, 1e-2, 0.25), loader, loader)
            estimate = profile.estimate(80, 20, 4, 8)
            self.assertTrue(Path(checkpoint.best_model_path).is_file())
            self.assertEqual(estimate["measured_training_images"], 8)
            self.assertEqual(estimate["measured_validation_images"], 10)
            self.assertGreater(estimate["validation_metrics_and_checkpoint_seconds"], 0)
            self.assertGreater(estimate["estimated_run_hours_with_margin"], 0)


if __name__ == "__main__":
    unittest.main()
