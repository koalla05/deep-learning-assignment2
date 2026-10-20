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
