"""Five-fold helpers: per-label thresholds, write-once records and fold bookkeeping."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from stage2.cv import fold_status, load_fold_record, record_failure, sha256, write_json_once, write_matrix
from stage2.experiment import LABEL_COLUMNS
from stage2.thresholds import FINE_GRID, f1_over_grid, nested_thresholds, per_label_f1, tune_label_thresholds


class PerLabelThresholdTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        self.targets = (rng.uniform(size=(500, 5)) < [0.4, 0.4, 0.05, 0.05, 0.05]).astype(int)
        self.probabilities = np.clip(self.targets * rng.uniform(0.2, 1.0, (500, 5)) + rng.uniform(0, 0.3, (500, 5)), 0, 1)
        self.folds = np.arange(500) % 5

    def test_grid_f1_matches_sklearn(self) -> None:
        for index in range(5):
            fast = f1_over_grid(self.targets[:, index], self.probabilities[:, index], FINE_GRID)
            slow = [f1_score(self.targets[:, index], self.probabilities[:, index] >= t, zero_division=0) for t in FINE_GRID]
            np.testing.assert_allclose(fast, slow)

    def test_no_positives_and_no_predictions_scores_zero_like_sklearn(self) -> None:
        self.assertEqual(f1_over_grid(np.zeros(4), np.zeros(4), np.array([0.5]))[0], 0.0)

    def test_tuned_thresholds_are_never_worse_than_fixed_on_the_same_data(self) -> None:
        tuned = tune_label_thresholds(self.targets, self.probabilities)
        self.assertEqual(tuned.shape, (5,))
        self.assertTrue(np.all(per_label_f1(self.targets, self.probabilities, tuned) >= per_label_f1(self.targets, self.probabilities, 0.25) - 1e-12))

    def test_nested_estimate_uses_one_threshold_set_per_held_out_fold(self) -> None:
        result = nested_thresholds(self.targets, self.probabilities, self.folds)
        self.assertEqual(sorted(result["thresholds_by_held_out_fold"]), [0, 1, 2, 3, 4])
        self.assertEqual(list(result["thresholds"]), list(LABEL_COLUMNS))
        self.assertLessEqual(result["fair_macro_f1"], result["optimistic_macro_f1"] + 1e-12)


class FoldBookkeepingTests(unittest.TestCase):
    def test_records_are_written_once(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "record.json"
            write_json_once(path, {"a": 1})
            with self.assertRaises(FileExistsError):
                write_json_once(path, {"a": 2})
            self.assertEqual(json.loads(path.read_text()), {"a": 1})

    def test_matrix_accepts_per_row_folds(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "oof.csv"
            write_matrix(path, ["a", "b"], np.full((2, 5), 0.5), [3, 4])
            lines = path.read_text().splitlines()
            self.assertEqual(lines[0], "image_id,fold," + ",".join(LABEL_COLUMNS))
            self.assertTrue(lines[2].startswith("b,4,"))
            self.assertFalse((Path(folder) / "oof.csv.partial").exists())

    def test_changed_file_breaks_a_completed_fold(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            model_dir = Path(folder)
            directory = model_dir / "fold2"
            self.assertEqual(fold_status(model_dir, 2), "not started")
            write_matrix(directory / "oof.csv", ["a"], np.full((1, 5), 0.5), 2)
            write_json_once(directory / "fold.json", {"status": "completed", "fold": 2,
                                                      "files": {"oof": {"path": "oof.csv", "sha256": sha256(directory / "oof.csv")}}})
            self.assertEqual(fold_status(model_dir, 2), "completed")
            self.assertEqual(load_fold_record(model_dir, 2)["fold"], 2)
            (directory / "oof.csv").write_text("changed\n")
            self.assertEqual(fold_status(model_dir, 2), "broken")

    def test_failures_leave_a_note_and_propagate(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(KeyError):
                with record_failure(Path(folder), "training", {"fold": 1}):
                    raise KeyError("secret text")
            notes = list(Path(folder).glob("failure_*_training.json"))
            self.assertEqual(len(notes), 1)
            self.assertNotIn("secret text", notes[0].read_text())


if __name__ == "__main__":
    unittest.main()
