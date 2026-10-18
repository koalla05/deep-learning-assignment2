"""Make a local Kaggle CSV from a completed Stage 2 fine-tuning checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from stage2.ensemble import load_finetuned_network
from stage2.experiment import LABEL_COLUMNS, ImageDataset, form_submission, read_csv, write_submission


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, help="Predict a few images for a smoke test; do not write a CSV.")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Batch size and smoke-test limit must be positive.")

    root = Path(__file__).resolve().parents[1]
    with args.record.open() as source:
        record = json.load(source)
    if args.output.exists() and args.limit is None:
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    data_dir = root / "data"
    test_rows = read_csv(data_dir / "test.csv", ("image_id",))
    sample_path = data_dir / "sample_submission.csv"
    sample_rows = read_csv(sample_path, ("image_id", *LABEL_COLUMNS))
    test_ids = [row["image_id"] for row in test_rows]
    sample_ids = [row["image_id"] for row in sample_rows]
    if not test_ids or len(set(test_ids)) != len(test_ids) or len(set(sample_ids)) != len(sample_ids) or set(test_ids) != set(sample_ids):
        raise ValueError("Test and sample submission IDs must be unique and identical.")

    os.environ["HF_HUB_OFFLINE"] = "1"
    network = load_finetuned_network(record, args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    network.to(device).eval()

    rows = test_rows[:args.limit] if args.limit is not None else test_rows
    dataset = ImageDataset(rows, data_dir / "test", record["data"]["image_size"], record["data"]["mean"], record["data"]["std"])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    predicted_ids: list[str] = []
    probabilities: list[np.ndarray] = []
    print(f"{record['run_id']}: {len(rows)} images on {device}; threshold {record['metrics']['threshold']}", flush=True)
    with torch.inference_mode():
        for batch_index, (ids, images, _) in enumerate(loader, start=1):
            predicted_ids.extend(ids)
            probabilities.append(network(images.to(device)).sigmoid().cpu().numpy())
            if batch_index % 100 == 0:
                print(f"Predicted {len(predicted_ids)}/{len(rows)}", flush=True)
    if predicted_ids != [row["image_id"] for row in rows]:
        raise ValueError("Prediction order changed during loading.")
    scores = np.concatenate(probabilities)
    if scores.shape != (len(rows), len(LABEL_COLUMNS)) or not np.isfinite(scores).all():
        raise ValueError("Invalid prediction matrix.")
    if args.limit is not None:
        print("Smoke test passed; no CSV written.")
        return

    submission = form_submission(sample_path, predicted_ids, scores, record["metrics"]["threshold"])
    write_submission(args.output, submission)
    print("Positive predictions:", {label: sum(row[label] for row in submission) for label in LABEL_COLUMNS})
    print(f"Saved {len(submission)} rows: {args.output}")
    print("Nothing uploaded to Kaggle.")


if __name__ == "__main__":
    main()
