"""Checkpoint identity, recipe provenance and safe failure persistence."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from stage2.run_records import file_sha256, record_failures, validate_evaluation_record


def custom_record() -> dict:
    return {
        "run_id": "original", "experiment": "02_convnext_custom", "status": "completed",
        "fold": 0, "seed": 42,
        "model": {
            "name": "pinned-model", "variant": "residual_spatial_block", "block_reduction": 4,
            "initial_head_checkpoint": "head.pt", "initial_head_sha256": "head-hash",
        },
        "data": {
            "image_size": 384, "transform": "letterbox", "fold_file": "folds.csv",
            "fold_sha256": "fold-hash", "mean": [0.5] * 3, "std": [0.2] * 3,
            "label_columns": ["Cat", "Dog", "Moris", "Motya", "Biatrix"],
        },
        "training": {
            "optimizer": "AdamW", "scheduler": "warmup cosine", "loss": "BCEWithLogitsLoss",
            "max_epochs": 8, "patience": 3, "physical_batch_size": 2, "effective_batch_size": 16,
            "precision": "32-true", "stage_policy": "full", "checkpoint_selection": "F1",
            "gradient_clip_norm": 1.0,
        },
        "metrics": {"threshold": 0.25},
    }


class RunRecordTests(unittest.TestCase):
    def test_standard_evaluation_requires_completed_matching_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "best.ckpt"
            checkpoint.write_bytes(b"standard weights")
            expected = custom_record()
            expected["experiment"] = "01_convnext_standard"
            expected["model"] = {key: value for key, value in expected["model"].items() if key != "block_reduction"}
            expected["model"]["variant"] = "standard"
            record = {**expected, "artifacts": {"best_checkpoint_sha256": file_sha256(checkpoint)}}
            path = root / "standard.json"
            path.write_text(json.dumps(record))
            self.assertEqual(validate_evaluation_record(path, checkpoint, expected, experiment="01_convnext_standard")["run_id"], "original")
            with self.assertRaisesRegex(ValueError, "matching"):
                validate_evaluation_record(path, checkpoint, expected)
            checkpoint.write_bytes(b"changed weights")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                validate_evaluation_record(path, checkpoint, expected, experiment="01_convnext_standard")

    def test_failure_backup_is_best_effort_and_never_masks_original(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backup = root / "drive" / "failure.json"
            error = RuntimeError("secret=do-not-persist")

            def save_failure(path: Path) -> None:
                backup.parent.mkdir()
                backup.write_bytes(path.read_bytes())

            with self.assertRaises(RuntimeError) as raised:
                with record_failures(root / "run", {"run_id": "test"}, "training", save_failure):
                    raise error
            self.assertIs(raised.exception, error)
            self.assertEqual(backup.read_bytes(), (root / "run" / "failure.json").read_bytes())
            self.assertNotIn("do-not-persist", backup.read_text())

            other = RuntimeError("original")
            with self.assertRaises(RuntimeError) as raised:
                with record_failures(root / "another", {}, "training", lambda path: (_ for _ in ()).throw(OSError("drive failed"))):
                    raise other
            self.assertIs(raised.exception, other)
            self.assertTrue((root / "another" / "failure.json").is_file())
            self.assertIn("OSError", other.__notes__[0])

    def test_evaluation_accepts_only_original_completed_checkpoint_and_recipe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "renamed.ckpt"
            checkpoint.write_bytes(b"original weights")
            expected = custom_record()
            record = {**expected, "artifacts": {"best_checkpoint_sha256": file_sha256(checkpoint)}}
            path = root / "original.json"
            path.write_text(json.dumps(record))
            self.assertEqual(validate_evaluation_record(path, checkpoint, expected)["run_id"], "original")
            for section, key, value in (
                ("training", "max_epochs", 10), ("training", "optimizer", "SGD"),
                ("model", "initial_head_sha256", "other-head"),
                ("data", "fold_sha256", "other-folds"),
                ("data", "label_columns", list(reversed(expected["data"]["label_columns"]))),
                ("metrics", "threshold", 0.5),
            ):
                changed = deepcopy(expected)
                changed[section][key] = value
                with self.subTest(section=section, key=key), self.assertRaisesRegex(ValueError, "settings differ"):
                    validate_evaluation_record(path, checkpoint, changed)
            checkpoint.write_bytes(b"other weights, same filename")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                validate_evaluation_record(path, checkpoint, expected)
            for status in ("failed", "profile_only", "smoke_runtime_estimate"):
                path.write_text(json.dumps({**record, "status": status}))
                with self.subTest(status=status), self.assertRaisesRegex(ValueError, "completed"):
                    validate_evaluation_record(path, checkpoint, expected)
            legacy = deepcopy(record)
            del legacy["data"]["fold_sha256"]
            path.write_text(json.dumps(legacy))
            with self.assertRaisesRegex(ValueError, "provenance"):
                validate_evaluation_record(path, checkpoint, expected)
            legacy = deepcopy(record)
            del legacy["artifacts"]["best_checkpoint_sha256"]
            path.write_text(json.dumps(legacy))
            with self.assertRaisesRegex(ValueError, "SHA256"):
                validate_evaluation_record(path, checkpoint, expected)

    def test_failure_propagates_and_does_not_persist_secret_messages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for phase in ("model", "training", "evaluation", "results"):
                target = path / phase
                error = RuntimeError("secret=do-not-persist")
                with self.subTest(phase=phase), self.assertRaises(RuntimeError) as raised:
                    with record_failures(target, {"run_id": "test", "seed": 42}, phase):
                        raise error
                self.assertIs(raised.exception, error)
                raw = (target / "failure.json").read_text()
                record = json.loads(raw)
                self.assertNotIn("do-not-persist", raw)
                self.assertEqual(record["phase"], phase)
                self.assertEqual(record["error"]["type"], "RuntimeError")
                self.assertTrue(record["error"]["frames"])
            with record_failures(path / "success", {}, "training"):
                pass
            self.assertFalse((path / "success").exists())
            with self.assertRaises(KeyboardInterrupt):
                with record_failures(path / "interrupted", {}, "training"):
                    raise KeyboardInterrupt()
            self.assertEqual(json.loads((path / "interrupted/failure.json").read_text())["status"], "failed")

    def test_recording_failure_never_masks_original_error_or_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "failure.json").write_text("existing")
            error = ValueError("original")
            with self.assertRaises(ValueError) as raised:
                with record_failures(root, {}, "results"):
                    raise error
            self.assertIs(raised.exception, error)
            self.assertIn("FileExistsError", error.__notes__[0])
            self.assertEqual((root / "failure.json").read_text(), "existing")


if __name__ == "__main__":
    unittest.main()
