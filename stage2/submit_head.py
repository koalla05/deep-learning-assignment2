"""Build a local submission from a completed frozen-backbone head run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from stage2.experiment import LABEL_COLUMNS, ImageDataset, form_submission, read_csv, write_submission
from stage2.train import MultilabelClassifier, predict_probabilities


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true", help="Predict one batch without writing a CSV.")
    parser.add_argument("--batch-size", type=int, default=2)
    arguments = parser.parse_args()
    if arguments.batch_size < 1:
        parser.error("--batch-size must be positive.")

    root = Path(__file__).resolve().parents[1]
    record_path = root / "experiments/exp0/results/convnext_head_fold0_seed42_colab/convnext_head_fold0_seed42_colab.json"
    with record_path.open() as record_file:
        record = json.load(record_file)
    if record["status"] != "completed" or record["model"]["variant"] != "head_only":
        raise ValueError("Expected a completed head-only run.")
    head_path = root / record["artifacts"]["head_checkpoint"]
    if not head_path.is_file():
        head_path = root / "experiments/exp0/checkpoints" / Path(record["artifacts"]["head_checkpoint"]).name
    if not head_path.is_file():
        raise FileNotFoundError(f"Missing trained head: {head_path}")
    submission_path = root / "submissions" / f"{record['run_id']}_025.csv"
    if submission_path.exists() and not arguments.smoke_test:
        raise FileExistsError(f"Refusing to overwrite {submission_path}")

    data_dir = root / "data"
    test_rows = read_csv(data_dir / "test.csv", ("image_id",))
    sample_path = data_dir / "sample_submission.csv"
    sample_rows = read_csv(sample_path, ("image_id", *LABEL_COLUMNS))
    test_ids = [row["image_id"] for row in test_rows]
    sample_ids = [row["image_id"] for row in sample_rows]
    if not test_ids or len(set(test_ids)) != len(test_ids) or len(set(sample_ids)) != len(sample_ids) or set(test_ids) != set(sample_ids):
        raise ValueError("Test and sample IDs must be nonempty, unique, and matching.")
    missing_images = [image_id for image_id in test_ids if not (data_dir / "test" / f"{image_id}.jpg").is_file()]
    if missing_images:
        raise FileNotFoundError(f"Missing {len(missing_images)} test images; first ID: {missing_images[0]}")

    os.environ["HF_HUB_OFFLINE"] = "1"
    import timm

    torch.manual_seed(record["seed"])
    network = timm.create_model(record["model"]["name"], pretrained=True, num_classes=len(LABEL_COLUMNS))
    network.get_classifier().load_state_dict(torch.load(head_path, map_location="cpu", weights_only=True), strict=True)
    threshold = record["metrics"]["threshold"]
    module = MultilabelClassifier(network, learning_rate=0.0, weight_decay=0.0, threshold=threshold, frozen_backbone=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    prediction_rows = test_rows[:arguments.batch_size] if arguments.smoke_test else test_rows
    dataset = ImageDataset(prediction_rows, data_dir / "test", record["data"]["image_size"], record["data"]["mean"], record["data"]["std"])
    loader = DataLoader(dataset, batch_size=arguments.batch_size, shuffle=False, num_workers=0)
    print(f"Run: {record['run_id']}; device: {device}; images: {len(dataset)}; threshold: {threshold}", flush=True)
    predicted_ids, probabilities = predict_probabilities(module, loader, device)
    assert predicted_ids == [row["image_id"] for row in prediction_rows]
    assert probabilities.shape == (len(prediction_rows), len(LABEL_COLUMNS))
    assert np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()
    if arguments.smoke_test:
        print("Smoke test passed; no submission written or uploaded.")
        return
    submission_rows = form_submission(sample_path, predicted_ids, probabilities, threshold)
    write_submission(submission_path, submission_rows)
    print("Positive predictions:", {label: sum(row[label] for row in submission_rows) for label in LABEL_COLUMNS})
    print(f"Saved {len(submission_rows)} rows: {submission_path}")
    print("Nothing uploaded to Kaggle.")


if __name__ == "__main__":
    main()
