"""Per-cell empirical separability (orientation-agnostic AUC) and the AUC-vs-w curve."""

from __future__ import annotations

import numpy as np

from cfgid.detector.empirical import auc


def cell_auc(h0_scores: np.ndarray, h1_scores: np.ndarray) -> float:
    """Orientation-agnostic separability of H1 vs H0 image-level scores (0.5 = inseparable)."""
    h0 = np.asarray(h0_scores).reshape(-1)
    h1 = np.asarray(h1_scores).reshape(-1)
    if h0.size == 0 or h1.size == 0:
        raise ValueError("cell_auc requires non-empty H0 and H1 scores")
    scores = np.concatenate([h1, h0])
    labels = np.concatenate([np.ones(h1.size), np.zeros(h0.size)])
    a = auc(scores, labels)
    return max(a, 1.0 - a)


def auc_curve(scores_by_w: dict[int, np.ndarray], *, h0_w: int = 1) -> dict[int, float]:
    """AUC(w vs H0) for every w (the H0 entry is ~0.5 by construction)."""
    if h0_w not in scores_by_w:
        raise ValueError(f"h0_w={h0_w} not present in scores_by_w")
    h0 = scores_by_w[h0_w]
    return {w: cell_auc(h0, scores_by_w[w]) for w in sorted(scores_by_w)}
