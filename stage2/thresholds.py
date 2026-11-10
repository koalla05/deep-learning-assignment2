"""Choose per-label decision thresholds from held-out predictions, only where the data can support it."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import f1_score

from stage2.experiment import LABEL_COLUMNS

DEFAULT_GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)


@dataclass(frozen=True)
class ThresholdChoice:
    label: str
    positives: int
    informative: int
    threshold: float
    tuned: bool
    f1_default: float
    f1_chosen: float
    plateau: tuple[float, float]
    bootstrap_interval: tuple[float, float] | None


def best_plateau(targets: np.ndarray, probabilities: np.ndarray, grid: np.ndarray = DEFAULT_GRID) -> np.ndarray:
    """All grid thresholds that reach the best F1; with few examples this range is wide."""
    scores = np.array([f1_score(targets, probabilities >= threshold, zero_division=0) for threshold in grid])
    return grid[np.isclose(scores, scores.max())]


def choose_thresholds(
    targets: np.ndarray, probabilities: np.ndarray, default: float = 0.25, min_positives: int = 100,
    grid: np.ndarray = DEFAULT_GRID, bootstrap: int = 200, seed: int = 0,
) -> list[ThresholdChoice]:
    """Tune a label only if it has enough positives; otherwise keep the default.

    The tuned threshold is the middle of the best-F1 plateau, which is less sensitive
    to a single example than either end of it. The bootstrap interval shows how much
    that choice moves when the held-out images are resampled.
    """
    if targets.shape != probabilities.shape or targets.ndim != 2 or targets.shape[1] != len(LABEL_COLUMNS):
        raise ValueError("Targets and probabilities must both have shape (images, 5).")
    if not 0 < default < 1 or min_positives < 1:
        raise ValueError("Expected a default threshold in (0, 1) and a positive minimum count.")
    rng = np.random.default_rng(seed)
    choices = []
    for index, label in enumerate(LABEL_COLUMNS):
        y, p = targets[:, index], probabilities[:, index]
        positives = int(y.sum())
        plateau = best_plateau(y, p, grid)
        tuned = positives >= min_positives
        threshold = float(np.median(plateau)) if tuned else default
        interval = None
        if tuned and bootstrap:
            picks = []
            for _ in range(bootstrap):
                rows = rng.integers(0, len(y), len(y))
                picks.append(float(np.median(best_plateau(y[rows], p[rows], grid))))
            interval = (float(np.percentile(picks, 5)), float(np.percentile(picks, 95)))
        choices.append(ThresholdChoice(
            label=label, positives=positives, informative=int(((p > grid[0]) & (p < grid[-1])).sum()),
            threshold=threshold, tuned=tuned,
            f1_default=float(f1_score(y, p >= default, zero_division=0)),
            f1_chosen=float(f1_score(y, p >= threshold, zero_division=0)),
            plateau=(float(plateau.min()), float(plateau.max())), bootstrap_interval=interval,
        ))
    return choices


# Per-label thresholds from five-fold out-of-fold predictions.
#
# With all five folds there are 41-56 photos of each named pet instead of 8-10 on
# fold 0, enough to give every label its own threshold. Choosing a threshold on the
# same predictions we score is optimistic, so `nested_thresholds` also reports a fair
# estimate: each fold is scored with thresholds chosen on the other four folds only.

FINE_GRID = np.round(np.arange(0.02, 0.981, 0.01), 2)


def f1_over_grid(targets: np.ndarray, probabilities: np.ndarray, grid: np.ndarray = FINE_GRID) -> np.ndarray:
    """F1 of one label at every grid threshold (prediction is `probability >= threshold`).

    Same values as sklearn's f1_score with zero_division=0, but vectorized, so an
    ensemble search can afford it many times.
    """
    if targets.ndim != 1 or targets.shape != probabilities.shape:
        raise ValueError("Expected one label: 1-D targets and probabilities of the same length.")
    predicted = probabilities[:, None] >= grid[None, :]
    positive = targets.astype(bool)[:, None]
    true_positives = (predicted & positive).sum(axis=0)
    false_positives = predicted.sum(axis=0) - true_positives
    false_negatives = int(positive.sum()) - true_positives
    denominator = 2 * true_positives + false_positives + false_negatives
    return np.where(denominator > 0, 2 * true_positives / np.maximum(denominator, 1), 0.0)


def tune_label_thresholds(targets: np.ndarray, probabilities: np.ndarray, grid: np.ndarray = FINE_GRID) -> np.ndarray:
    """One threshold per label: the middle of that label's best-F1 plateau."""
    if targets.shape != probabilities.shape or targets.ndim != 2 or targets.shape[1] != len(LABEL_COLUMNS):
        raise ValueError("Targets and probabilities must both have shape (images, 5).")
    thresholds = []
    for index in range(len(LABEL_COLUMNS)):
        scores = f1_over_grid(targets[:, index], probabilities[:, index], grid)
        plateau = grid[np.isclose(scores, scores.max())]
        thresholds.append(float(np.median(plateau)))
    return np.asarray(thresholds)


def per_label_f1(targets: np.ndarray, probabilities: np.ndarray, thresholds) -> np.ndarray:
    """F1 of every label at its own threshold (a scalar threshold is used for all five)."""
    thresholds = np.broadcast_to(np.asarray(thresholds, dtype=float), (len(LABEL_COLUMNS),))
    return np.array([
        f1_over_grid(targets[:, index], probabilities[:, index], thresholds[index:index + 1])[0]
        for index in range(len(LABEL_COLUMNS))
    ])


def nested_thresholds(
    targets: np.ndarray, probabilities: np.ndarray, folds: np.ndarray, grid: np.ndarray = FINE_GRID,
) -> dict:
    """Fair per-label-threshold score: tune on four folds, score the fifth, pool the five.

    Returns the thresholds chosen for each held-out fold (their spread shows how stable
    the choice is), the fair per-label and macro F1 of the pooled decisions, and the
    thresholds tuned on all folds, which are the ones applied to the test set.
    """
    if folds.shape != (len(targets),):
        raise ValueError("Expected one fold number per image.")
    decisions = np.zeros(targets.shape, dtype=bool)
    by_fold = {}
    for fold in np.unique(folds):
        held_out = folds == fold
        chosen = tune_label_thresholds(targets[~held_out], probabilities[~held_out], grid)
        by_fold[int(fold)] = chosen.tolist()
        decisions[held_out] = probabilities[held_out] >= chosen
    fair = per_label_f1(targets, decisions.astype(float), 0.5)
    final = tune_label_thresholds(targets, probabilities, grid)
    optimistic = per_label_f1(targets, probabilities, final)
    return {
        "thresholds": dict(zip(LABEL_COLUMNS, final.tolist())),
        "thresholds_by_held_out_fold": {fold: dict(zip(LABEL_COLUMNS, values)) for fold, values in by_fold.items()},
        "fair_macro_f1": float(fair.mean()),
        "fair_per_label_f1": dict(zip(LABEL_COLUMNS, fair.tolist())),
        "optimistic_macro_f1": float(optimistic.mean()),
        "optimistic_per_label_f1": dict(zip(LABEL_COLUMNS, optimistic.tolist())),
    }
