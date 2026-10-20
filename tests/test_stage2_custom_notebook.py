"""Safety switches and matched-control checks in the custom notebook."""

from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from stage2.experiment import LABEL_COLUMNS
from stage2.run_records import record_failures, validate_matched_control


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "07_convnext_custom.ipynb"


def cell_source(cell_id: str) -> str:
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    return "".join(next(cell for cell in cells if cell["id"] == cell_id)["source"])


class CustomNotebookTests(unittest.TestCase):
    def configure(self, environment: dict[str, str]) -> dict:
        namespace = {}
        with patch.dict(os.environ, environment, clear=True), contextlib.redirect_stdout(io.StringIO()):
            exec(compile(cell_source("1c62c645"), str(NOTEBOOK), "exec"), namespace)
        return namespace

    def test_defaults_do_not_train_download_or_build_artifacts(self) -> None:
        config = self.configure({})
        for flag in ("LOCAL_SMOKE_TEST", "RUN_TRAINING", "PROFILE_RUNTIME", "ALLOW_WEIGHT_DOWNLOAD",
                     "BACKUP_TO_DRIVE", "BUILD_SUBMISSION", "BUILD_GRADCAM", "SUBMIT_TO_KAGGLE"):
            with self.subTest(flag=flag):
                self.assertFalse(config[flag])

    def test_local_switch_enables_bounded_profile_without_external_actions(self) -> None:
        config = self.configure({
            "STAGE2_LOCAL_SMOKE_TEST": "1", "STAGE2_USE_WANDB": "1",
            "STAGE2_BACKUP_TO_DRIVE": "1", "STAGE2_ALLOW_WEIGHT_DOWNLOAD": "1",
        })
        for flag in ("RUN_TRAINING", "SMOKE_TEST", "PROFILE_RUNTIME"):
            self.assertTrue(config[flag])
        for flag in ("USE_WANDB", "BACKUP_TO_DRIVE", "ALLOW_WEIGHT_DOWNLOAD"):
            self.assertFalse(config[flag])
        self.assertEqual(config["SMOKE_TRAIN_BATCHES"], 32)
        self.assertEqual(config["SMOKE_VALIDATION_BATCHES"], 8)

    def test_incompatible_modes_fail_before_model_loading(self) -> None:
        for environment in (
            {"STAGE2_PROFILE_RUNTIME": "1"},
            {"STAGE2_LOCAL_SMOKE_TEST": "1", "STAGE2_PROFILE_ONLY": "1"},
            {"STAGE2_LOCAL_SMOKE_TEST": "1", "STAGE2_EVAL_CHECKPOINT": "saved.ckpt"},
        ):
            with self.subTest(environment=environment), self.assertRaises(ValueError):
                self.configure(environment)

    def test_saved_evaluation_requires_original_run_record(self) -> None:
        with self.assertRaisesRegex(ValueError, "STAGE2_EVAL_RUN_RECORD"):
            self.configure({"STAGE2_EVAL_CHECKPOINT": "saved.ckpt"})

    def test_all_run_modes_use_unique_experiment_result_directories(self) -> None:
        import ast

        block = next(
            node for node in ast.parse(cell_source("31a52846")).body
            if isinstance(node, ast.If) and ast.unparse(node.test) == "RUN_TRAINING or EVAL_CHECKPOINT"
        )
        for training, smoke, checkpoint, suffix in (
            (True, False, None, ""), (True, True, None, "_smoke"),
            (False, True, "saved.ckpt", "_evaluation"), (False, True, None, None),
        ):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                namespace = {
                    "ROOT": root, "RUN_ID": "new-run", "RUN_TRAINING": training,
                    "SMOKE_TEST": smoke, "EVAL_CHECKPOINT": checkpoint, "run_dir": None,
                }
                source = compile(ast.Module(body=[block], type_ignores=[]), str(NOTEBOOK), "exec")
                exec(source, namespace)
                if suffix is None:
                    self.assertIsNone(namespace["run_dir"])
                    self.assertEqual(list(root.iterdir()), [])
                else:
                    expected = root / "experiments/exp2/results" / f"new-run{suffix}"
                    self.assertEqual(namespace["run_dir"], expected)
                    self.assertTrue(expected.is_dir())
                    with self.assertRaises(FileExistsError):
                        exec(source, namespace)
                self.assertFalse((root / "artifacts").exists())

    def test_approved_budget_override_does_not_block_full_evaluation(self) -> None:
        import ast
        from types import SimpleNamespace

        config = self.configure({})
        self.assertFalse(config["ENFORCE_RUNTIME_BUDGET"])
        config.update(SMOKE_TEST=False, checkpoint=SimpleNamespace(exceeded_budget=True))
        assignment = next(
            node for node in ast.walk(ast.parse(cell_source("c33f51c5")))
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "budget_exceeded" for target in node.targets)
        )
        exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(NOTEBOOK), "exec"), config)
        self.assertFalse(config["budget_exceeded"])

    def test_half_threshold_reports_each_label(self) -> None:
        import numpy as np
        from types import SimpleNamespace
        from stage2.experiment import score_probabilities

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best.ckpt"
            checkpoint.touch()
            probabilities = np.asarray([[0.8] * 5, [0.2] * 5])
            config = {
                "record_failures": record_failures, "run_dir": None, "failure_configuration": {},
                "EVAL_CHECKPOINT": str(checkpoint), "Path": Path, "RUN_TRAINING": False,
                "FineTuneClassifier": SimpleNamespace(load_from_checkpoint=lambda *a, **k: object()),
                "network": object(), "BACKBONE_LR": 1e-5, "HEAD_LR": 1e-4, "WEIGHT_DECAY": 0.01,
                "THRESHOLD": 0.25, "validation_loader": object(), "DEVICE": "cpu",
                "time": SimpleNamespace(perf_counter=lambda: 0),
                "predict_probabilities": lambda *args: (["a", "b"], probabilities),
                "validation_rows": [
                    {"image_id": "a", **dict.fromkeys(LABEL_COLUMNS, "1")},
                    {"image_id": "b", **dict.fromkeys(LABEL_COLUMNS, "0")},
                ],
                "np": np, "LABEL_COLUMNS": LABEL_COLUMNS, "score_probabilities": score_probabilities,
            }
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(cell_source("fcefe6a1"), str(NOTEBOOK), "exec"), config)
            self.assertEqual(config["diagnostic_metrics"]["per_label_f1"], dict.fromkeys(LABEL_COLUMNS, 1.0))
            config["run_dir"] = Path(directory) / "failed-evaluation"

            def failed_prediction(*args):
                raise RuntimeError("prediction failed")

            config["predict_probabilities"] = failed_prediction
            with self.assertRaisesRegex(RuntimeError, "prediction failed"):
                exec(compile(cell_source("fcefe6a1"), str(NOTEBOOK), "exec"), config)
            failure = json.loads((config["run_dir"] / "failure.json").read_text())
            self.assertEqual(failure["phase"], "evaluation")
            self.assertEqual(failure["error"]["type"], "RuntimeError")

    def test_evaluation_only_persists_predictions_without_rewriting_training_history(self) -> None:
        import csv
        import hashlib
        import numpy as np
        from types import SimpleNamespace
        from stage2.experiment import write_probabilities, write_run_record
        from stage2.run_records import file_sha256
        from test_stage2_run_records import custom_record

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                "notebooks/07_convnext_custom.ipynb", "stage2/experiment.py", "stage2/train.py",
                "stage2/finetune.py", "stage2/custom_convnext.py", "stage2/gradcam.py",
                "stage2/run_records.py", "stage2/diagnostics.py",
            ):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("source")
            source = custom_record()
            source["artifacts"] = {"best_checkpoint_sha256": "checkpoint-hash"}
            original = root / "original.json"
            original.write_text(json.dumps(source))
            before = original.read_bytes()
            run_dir = root / "experiments/exp2/results/reevaluation_evaluation"
            run_dir.mkdir(parents=True)
            metrics = {"macro_f1": 1.0, "per_label_f1": dict.fromkeys(LABEL_COLUMNS, 1.0)}
            namespace = {
                "RUN_TRAINING": False, "EVAL_CHECKPOINT": "best.ckpt", "EVAL_RUN_RECORD": str(original),
                "RUN_ID": "reevaluation", "ROOT": root, "run_dir": run_dir, "source_run_record": source,
                "record_failures": record_failures, "failure_configuration": {},
                "Path": Path, "write_probabilities": write_probabilities, "write_run_record": write_run_record,
                "file_sha256": file_sha256, "hashlib": hashlib,
                "predicted_ids": ["a", "b"], "validation_probabilities": np.asarray([[0.8] * 5, [0.2] * 5]),
                "VALIDATION_FOLD": 0, "checkpoint_path": root / "best.ckpt",
                "metrics": metrics, "diagnostic_metrics": metrics, "prediction_seconds": 0.1,
                "DEVICE": "cpu", "gradcam_paths": [], "platform": SimpleNamespace(python_version=lambda: "3.12"),
                "torch": SimpleNamespace(__version__="test"), "timm": SimpleNamespace(__version__="test"),
                "L": SimpleNamespace(__version__="test"),
            }
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(cell_source("6f941b60"), str(NOTEBOOK), "exec"), namespace)
            saved = json.loads((run_dir / "reevaluation.json").read_text())
            self.assertEqual(saved["status"], "evaluated")
            self.assertEqual(saved["source_run_id"], "original")
            self.assertEqual(saved["training"], source["training"])
            self.assertEqual(saved["evaluation"]["device"], "cpu")
            self.assertEqual(saved["metrics"]["per_label_f1_at_0_5"], metrics["per_label_f1"])
            self.assertEqual(original.read_bytes(), before)
            with (run_dir / "reevaluation_val.csv").open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([row["image_id"] for row in rows], ["a", "b"])
            self.assertEqual([row["fold"] for row in rows], ["0", "0"])
            with self.assertRaises(FileExistsError):
                exec(compile(cell_source("6f941b60"), str(NOTEBOOK), "exec"), namespace)

    def test_control_requires_matching_optimizer_epoch_cap_and_unfreezing(self) -> None:
        config = self.configure({})
        config.update(
            CONTROL_RUN_ID="control", EVAL_CHECKPOINT="saved.ckpt",
            LABEL_COLUMNS=LABEL_COLUMNS,
            record_failures=record_failures, run_dir=None, failure_configuration={},
            validate_matched_control=validate_matched_control,
            preprocessing={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            metrics={"macro_f1": 0.9, "per_label_f1": dict.fromkeys(LABEL_COLUMNS, 0.9)},
        )
        record = {
            "status": "completed", "fold": 0, "seed": 42,
            "model": {"variant": "standard", "name": config["MODEL_NAME"], "initial_head_checkpoint": "head.pt", "initial_head_sha256": "head-hash"},
            "data": {
                "image_size": 384, **config["preprocessing"],
                "fold_file": "folds.csv", "fold_sha256": "fold-hash", "label_columns": list(LABEL_COLUMNS),
                "transform": "mean-colour letterbox, bicubic, checkpoint normalization; train horizontal flip p=0.5",
            },
            "metrics": {"threshold": 0.25, **config["metrics"]},
            "training": {
                "optimizer": "AdamW(backbone_lr=1e-05, head_lr=0.0001, weight_decay=0.01 except bias/norm, betas=(0.9, 0.999))",
                "effective_batch_size": 16, "precision": "32-true", "max_epochs": 8, "patience": 3,
                "loss": "BCEWithLogitsLoss", "gradient_clip_norm": 1.0,
                "stage_policy": "full backbone and classifier trainable",
                "scheduler": "one epoch step warmup then cosine to 0.1 peak",
                "checkpoint_selection": "best validation macro F1 at 0.25",
            },
        }
        config["source_run_record"] = {
            **record,
            "model": {**record["model"], "variant": "residual_spatial_block",
                      "initial_head_checkpoint": "experiments/exp0/checkpoints/head.pt"},
            "training": {
                **record["training"],
                "optimizer": record["training"]["optimizer"].replace("backbone_lr=", "backbone_and_block_lr="),
                "stage_policy": "full backbone, residual block and classifier trainable",
            },
        }
        source = compile(cell_source("custom-control-code"), str(NOTEBOOK), "exec")
        with tempfile.TemporaryDirectory() as directory:
            config["ROOT"] = Path(directory)
            config["head_path"] = Path(directory) / "head.pt"
            path = Path(directory) / "experiments/exp1/results/control/control.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(record))
            with contextlib.redirect_stdout(io.StringIO()):
                exec(source, config)
            config.update(MAX_EPOCHS=99, BACKBONE_LR=0.5, PRECISION="16-mixed")
            with contextlib.redirect_stdout(io.StringIO()):
                exec(source, config)
            for key, value in (
                ("optimizer", "AdamW(backbone_lr=3e-5)"), ("max_epochs", 10),
                ("patience", 5), ("stage_policy", "last stage only"),
            ):
                changed = {**record, "training": {**record["training"], key: value}}
                path.write_text(json.dumps(changed))
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Control settings differ"):
                    exec(source, config)


if __name__ == "__main__":
    unittest.main()
