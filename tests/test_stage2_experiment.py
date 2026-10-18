"""Small contract tests that do not need pretrained model weights."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from stage2.experiment import LABEL_COLUMNS, form_submission, preflight_images, score_probabilities, split_rows, write_probabilities


class ExperimentContractTests(unittest.TestCase):
    def test_image_preflight_checks_every_file_and_decodes_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (2, 2), "red").save(root / "a.jpg")
            rows = [{"image_id": "a"}, {"image_id": "b"}]
            with self.assertRaisesRegex(FileNotFoundError, "b.jpg"):
                preflight_images(rows, root)
            (root / "b.jpg").write_bytes(b"not a JPEG")
            with self.assertRaisesRegex(ValueError, "b.jpg"):
                preflight_images(rows, root)
            Image.new("RGB", (2, 2), "blue").save(root / "b.jpg")
            preflight_images(rows, root)

    def test_fold_join_uses_image_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train.csv").write_text(
                "image_id,Cat,Dog,Moris,Motya,Biatrix\n"
                "a,1,0,0,0,0\n"
                "b,0,1,0,0,0\n"
                "c,0,0,1,0,0\n"
                "d,0,0,0,1,0\n"
                "e,0,0,0,0,1\n"
            )
            folds = root / "folds.csv"
            folds.write_text("image_id,fold\ne,4\nd,3\nc,2\nb,1\na,0\n")
            training, validation = split_rows(root, folds, 0)
            self.assertEqual([row["image_id"] for row in validation], ["a"])
            self.assertEqual([row["image_id"] for row in training], ["b", "c", "d", "e"])

    def test_metrics_and_submission_keep_label_order(self) -> None:
        targets = np.array([[1, 0, 1, 0, 1], [0, 1, 0, 1, 0]])
        probabilities = np.array([[0.9, 0.1, 0.8, 0.2, 0.7], [0.1, 0.9, 0.2, 0.8, 0.1]])
        metrics = score_probabilities(targets, probabilities, 0.25)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(tuple(metrics["per_label_f1"]), LABEL_COLUMNS)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = root / "sample.csv"
            sample.write_text("image_id,Cat,Dog,Moris,Motya,Biatrix\nb,1,1,1,1,1\na,1,1,1,1,1\n")
            rows = form_submission(sample, ["a", "b"], probabilities, 0.25)
            self.assertEqual([row["image_id"] for row in rows], ["b", "a"])
            self.assertEqual([rows[0][label] for label in LABEL_COLUMNS], [0, 1, 0, 1, 0])
            output = root / "val.csv"
            write_probabilities(output, ["a", "b"], probabilities, 0)
            with output.open(newline="") as file:
                self.assertEqual(next(csv.reader(file)), ["image_id", "fold", *LABEL_COLUMNS])
            with self.assertRaises(FileExistsError):
                write_probabilities(output, ["a", "b"], probabilities, 0)


if __name__ == "__main__":
    unittest.main()
