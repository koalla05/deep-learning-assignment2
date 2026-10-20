"""Result and safety contracts for the unchanged ConvNeXt notebook."""

from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from stage2.experiment import LABEL_COLUMNS, write_probabilities, write_run_record
from stage2.run_records import file_sha256, record_failures
from test_stage2_run_records import custom_record


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "06_convnext_standard.ipynb"


def cell_source(cell_id: str) -> str:
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    return "".join(next(cell for cell in cells if cell["id"] == cell_id)["source"])


class StandardNotebookTests(unittest.TestCase):
    def test_safe_defaults_and_evaluation_record_pair(self) -> None:
        namespace = {}
        with patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(io.StringIO()):
            exec(compile(cell_source("1c62c645"), str(NOTEBOOK), "exec"), namespace)
        for flag in ("RUN_TRAINING", "ALLOW_WEIGHT_DOWNLOAD", "BACKUP_TO_DRIVE", "BUILD_SUBMISSION", "SUBMIT_TO_KAGGLE"):
            self.assertFalse(namespace[flag])
        with patch.dict(os.environ, {"STAGE2_EVAL_CHECKPOINT": "saved.ckpt"}, clear=True):
            with self.assertRaisesRegex(ValueError, "STAGE2_EVAL_RUN_RECORD"):
                exec(compile(cell_source("1c62c645"), str(NOTEBOOK), "exec"), {})

    def test_evaluation_only_writes_new_result_without_changing_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                "notebooks/06_convnext_standard.ipynb", "stage2/experiment.py", "stage2/train.py",
                "stage2/finetune.py", "stage2/run_records.py", "stage2/diagnostics.py", "stage2/gradcam.py",
            ):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("source")
            source = custom_record()
            source["experiment"] = "01_convnext_standard"
            source["model"]["variant"] = "standard"
            source["artifacts"] = {"best_checkpoint_sha256": "checkpoint-hash"}
            original = root / "original.json"
            original.write_text(json.dumps(source))
            before = original.read_bytes()
            run_dir = root / "experiments" / "exp1" / "results" / "reevaluation_evaluation"
            run_dir.mkdir(parents=True)
            metrics = {"macro_f1": 1.0, "per_label_f1": dict.fromkeys(LABEL_COLUMNS, 1.0)}
            namespace = {
                "RUN_TRAINING": False, "EVAL_CHECKPOINT": "best.ckpt", "EVAL_RUN_RECORD": str(original),
                "RUN_ID": "reevaluation", "ROOT": root, "run_dir": run_dir, "source_run_record": source,
                "record_failures": record_failures, "failure_configuration": {}, "backup_failure": None,
                "Path": Path, "write_probabilities": write_probabilities, "write_run_record": write_run_record,
                "file_sha256": file_sha256, "subprocess": __import__("subprocess"),
                "predicted_ids": ["a", "b"], "validation_probabilities": np.asarray([[0.8] * 5, [0.2] * 5]),
                "VALIDATION_FOLD": 0, "checkpoint_path": root / "best.ckpt",
                "metrics": metrics, "diagnostic_metrics": metrics, "prediction_seconds": 0.1,
                "DEVICE": "cpu", "gradcam_paths": [], "platform": __import__("platform"),
                "torch": type("Version", (), {"__version__": "test"})(),
                "timm": type("Version", (), {"__version__": "test"})(),
                "L": type("Version", (), {"__version__": "test"})(),
            }
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(cell_source("6f941b60"), str(NOTEBOOK), "exec"), namespace)
            saved = json.loads((run_dir / "reevaluation.json").read_text())
            self.assertEqual(saved["status"], "evaluated")
            self.assertEqual(saved["source_run_id"], "original")
            self.assertEqual(saved["training"], source["training"])
            self.assertEqual(saved["metrics"]["per_label_f1_at_0_5"], metrics["per_label_f1"])
            self.assertEqual(original.read_bytes(), before)
            self.assertTrue((run_dir / "reevaluation_val.csv").is_file())
            with self.assertRaises(FileExistsError):
                exec(compile(cell_source("6f941b60"), str(NOTEBOOK), "exec"), namespace)


if __name__ == "__main__":
    unittest.main()
