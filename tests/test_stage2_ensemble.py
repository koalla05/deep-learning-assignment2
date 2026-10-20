"""Contracts for combining saved run probabilities."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from stage2.ensemble import architecture_name, load_finetuned_network, mean_probabilities, predict_flipped_probabilities, read_probabilities, weighted_mean_probabilities, write_test_probabilities
from stage2.experiment import LABEL_COLUMNS, write_probabilities


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "08_ensemble.ipynb"


class EnsembleContractTests(unittest.TestCase):
    def test_weighted_mean_normalizes_weights_and_equal_weights_match_the_mean(self) -> None:
        a, b = np.full((2, 5), 0.2), np.full((2, 5), 0.8)
        np.testing.assert_allclose(weighted_mean_probabilities([a, b], [1, 1]), mean_probabilities([a, b]))
        np.testing.assert_allclose(weighted_mean_probabilities([a, b], [3, 1]), np.full((2, 5), 0.35))
        np.testing.assert_allclose(weighted_mean_probabilities([a, b], [0.3, 0.1]), np.full((2, 5), 0.35))
        np.testing.assert_allclose(weighted_mean_probabilities([a, b], [1, 0]), a)
        for bad in ([1], [1, -1], [0, 0], [1, float("nan")]):
            with self.assertRaises(ValueError):
                weighted_mean_probabilities([a, b], bad)

    def test_probabilities_are_aligned_by_id_not_row_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "val.csv"
            write_probabilities(path, ["a", "b"], np.array([[0.1] * 5, [0.9] * 5]), fold=0)
            probabilities = read_probabilities(path, ["b", "a"])
            np.testing.assert_allclose(probabilities[:, 0], [0.9, 0.1])
            with self.assertRaisesRegex(ValueError, "expected image IDs"):
                read_probabilities(path, ["a", "c"])

    def test_test_probabilities_round_trip_and_refuse_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "member_test" / "member_test.csv"
            values = np.array([[0.2, 0.8, 0.0, 1.0, 0.5]])
            write_test_probabilities(path, ["x"], values)
            np.testing.assert_allclose(read_probabilities(path, ["x"]), values)
            with self.assertRaises(FileExistsError):
                write_test_probabilities(path, ["x"], values)

    def test_mean_needs_two_aligned_members(self) -> None:
        first, second = np.zeros((3, 5)), np.ones((3, 5))
        np.testing.assert_allclose(mean_probabilities([first, second]), np.full((3, 5), 0.5))
        with self.assertRaises(ValueError):
            mean_probabilities([first])
        with self.assertRaises(ValueError):
            mean_probabilities([first, np.ones((2, 5))])

    def test_checkpoint_hash_is_checked_before_building_a_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best.ckpt"
            checkpoint.write_bytes(b"not the recorded checkpoint")
            record = {
                "status": "completed", "model": {"variant": "standard"},
                "data": {"label_columns": list(LABEL_COLUMNS)},
                "artifacts": {"best_checkpoint_sha256": "0" * 64},
            }
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_finetuned_network(record, checkpoint)

    def test_architecture_name_needs_no_network(self) -> None:
        name = "hf-hub:timm/convnext_base.fb_in22k_ft_in1k_384@2b2698d86523262ca8065d0957c7f80d963bb43f"
        self.assertEqual(architecture_name(name), "convnext_base")

    def test_notebook_defaults_do_not_predict_or_write(self) -> None:
        cells = json.loads(NOTEBOOK.read_text())["cells"]
        flags = "".join(next(cell for cell in cells if cell["id"] == "a1f0e002")["source"])
        self.assertIn("BUILD_SUBMISSION = False", flags)
        self.assertIn("SUBMIT_TO_KAGGLE = False", flags)


class EfficientNetNotebookTests(unittest.TestCase):
    def test_defaults_do_not_train_download_or_build_artifacts(self) -> None:
        notebook = NOTEBOOK.with_name("09_efficientnet.ipynb")
        cells = json.loads(notebook.read_text())["cells"]
        source = "".join(next(cell for cell in cells if cell["id"] == "b3f0e002")["source"])
        for expected in (
            "'STAGE2_RUN_TRAINING', '0'", "'STAGE2_ALLOW_WEIGHT_DOWNLOAD', '0'",
            "'STAGE2_BUILD_TEST_PREDICTIONS', '0'", "BUILD_SUBMISSION = False", "SUBMIT_TO_KAGGLE = False",
        ):
            self.assertIn(expected, source)
        self.assertIn("MODEL_NAME = 'tf_efficientnetv2_s.in21k_ft_in1k'", source)
        self.assertNotIn("jft", source)


class EnsembleExp1Exp3NotebookTests(unittest.TestCase):
    def test_defaults_do_not_predict_or_write_and_members_are_exp1_and_exp3(self) -> None:
        cells = json.loads(NOTEBOOK.with_name("10_ensemble_exp1_exp3.ipynb").read_text())["cells"]
        flags = "".join(next(cell for cell in cells if cell["id"] == "a1f0e002")["source"])
        self.assertIn("BUILD_SUBMISSION = False", flags)
        self.assertIn("SUBMIT_TO_KAGGLE = False", flags)
        self.assertIn("'STAGE2_RUN_TEST_INFERENCE', '0'", flags)
        self.assertIn("'exp1':", flags)
        self.assertIn("'exp3':", flags)
        self.assertNotIn("'exp2':", flags)


class FlipPredictionTests(unittest.TestCase):
    def test_predicts_the_mirror_image_in_loader_order(self) -> None:
        import torch

        torch.manual_seed(0)
        network = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(4, 5))
        images = torch.randn(5, 3, 8, 8)
        batches = [(["a", "b"], images[:2], torch.empty(0)), (["c", "d", "e"], images[2:], torch.empty(0))]
        ids, probabilities = predict_flipped_probabilities(network, batches, torch.device("cpu"))
        self.assertEqual(ids, list("abcde"))
        expected = network(images.flip(-1)).sigmoid().detach().numpy()
        np.testing.assert_allclose(probabilities, expected, atol=1e-6)
        self.assertGreater(float(np.abs(probabilities - network(images).sigmoid().detach().numpy()).max()), 1e-4)
        with self.assertRaises(ValueError):
            predict_flipped_probabilities(network, [], torch.device("cpu"))


class BitNotebookTests(unittest.TestCase):
    def test_defaults_do_not_train_download_or_build_artifacts(self) -> None:
        cells = json.loads(NOTEBOOK.with_name("12_bit_resnetv2.ipynb").read_text())["cells"]
        source = "".join(next(cell for cell in cells if cell["id"] == "b3f0e002")["source"])
        for expected in (
            "'STAGE2_RUN_TRAINING', '0'", "'STAGE2_SMOKE_TEST', '1'", "'STAGE2_ALLOW_WEIGHT_DOWNLOAD', '0'",
            "'STAGE2_BUILD_TEST_PREDICTIONS', '0'", "BUILD_SUBMISSION = False", "SUBMIT_TO_KAGGLE = False",
            "MODEL_NAME = 'resnetv2_50x1_bit.goog_in21k_ft_in1k'", "'STAGE2_VALIDATION_FOLD', '0'",
        ):
            self.assertIn(expected, source)
        self.assertNotIn("jft", source.replace("JFT", ""))
        self.assertNotIn("FrozenBatchNormStatistics", "".join("".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"))


class EnsembleThreeMemberNotebookTests(unittest.TestCase):
    def test_defaults_do_not_predict_or_write_and_members_are_exp1_exp3_exp5(self) -> None:
        cells = json.loads(NOTEBOOK.with_name("13_ensemble_exp1_exp3_exp5.ipynb").read_text())["cells"]
        flags = "".join(next(cell for cell in cells if cell["id"] == "a1f0e002")["source"])
        self.assertIn("BUILD_SUBMISSION = False", flags)
        self.assertIn("SUBMIT_TO_KAGGLE = False", flags)
        self.assertIn("'STAGE2_RUN_TEST_INFERENCE', '0'", flags)
        for member in ("'exp1':", "'exp3':", "'exp5':"):
            self.assertIn(member, flags)
        self.assertNotIn("'exp2':", flags)


class EnsembleHrnetNotebookTests(unittest.TestCase):
    def test_members_are_exp1_exp3_exp5_exp07_and_nothing_is_uploaded(self) -> None:
        cells = json.loads(NOTEBOOK.with_name("18_ensemble_exp1_exp3_exp5_exp7.ipynb").read_text())["cells"]
        flags = "".join(next(cell for cell in cells if cell["id"] == "b8e00002")["source"])
        code = "".join("".join(cell["source"]) for cell in cells if cell["cell_type"] == "code")
        self.assertIn("SUBMIT_TO_KAGGLE = False", flags)
        self.assertIn("'STAGE2_RUN_TEST_INFERENCE', '0'", flags)
        for member in ("'exp1':", "'exp3':", "'exp5':", "'exp07':"):
            self.assertIn(member, flags)
        self.assertNotIn("'exp2':", flags)
        self.assertIn("Refusing to overwrite", code)
        self.assertNotIn("kaggle competitions submit", code)


class FullDataNotebookTests(unittest.TestCase):
    def test_trains_all_members_on_all_folds_and_never_uploads(self) -> None:
        cells = json.loads(NOTEBOOK.with_name("14_full_data_final.ipynb").read_text())["cells"]
        config = "".join(next(cell for cell in cells if cell["id"] == "c14e0003")["source"])
        code = "".join("".join(cell["source"]) for cell in cells if cell["cell_type"] == "code")
        self.assertIn("SUBMIT_TO_KAGGLE = False", config)
        self.assertIn("TRAIN_ORDER = ['exp3', 'exp5', 'exp1']", config)
        self.assertIn("'backbone_lr': 5e-5, 'head_lr': 5e-4", config)
        self.assertIn("all_rows = folds_1_to_4 + fold_0", code)
        self.assertIn("MIN_POSITIVES_TO_TUNE = 100", config)
        self.assertNotIn("kaggle competitions submit", code)


if __name__ == "__main__":
    unittest.main()
