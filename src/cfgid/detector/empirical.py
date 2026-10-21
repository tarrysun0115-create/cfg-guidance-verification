"""Empirical ROC / AUC primitives (the shared AUC implementation of the repo).

``auc`` is the Mann-Whitney rank statistic (ties handled by average ranks): the probability a
random H1 score exceeds a random H0 score, ties counted 1/2 (the AUC convention of Appendix C).
It is orientation-SENSITIVE; callers that want sign-free separability take max(AUC, 1 - AUC)
explicitly. ``roc_curve`` returns the empirical FPR/TPR sweep.
These feed the per-cell separability used by ``align`` and the rho-sufficiency check.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from cfgid.numerics import as_finite_float32


def _split_by_label(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Validate finiteness via as_finite_float32 (rejects NaN/Inf), but RANK on the ORIGINAL float64
    # values: narrowing to float32 before rankdata/argsort collapses genuinely-distinct float64
    # scores below the float32 ULP into ties, biasing AUC toward 0.5. The validation pass is just a
    # NaN/Inf guard; the splits stay float64 (symmetric across rho/baselines, so the comparison
    # direction is unchanged; genuinely-equal float64 values still tie).
    as_finite_float32("scores", scores)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(labels).reshape(-1)
    if s.shape != y.shape:
        raise ValueError(f"scores {s.shape} and labels {y.shape} must have the same length")
    uniq = set(np.unique(y).tolist())
    if not uniq <= {0, 1}:
        raise ValueError(f"labels must be in {{0, 1}}, got {sorted(uniq)}")
    return s[y == 1], s[y == 0]


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC = P(score(H1) > score(H0)), ties counted as 0.5. labels in {0, 1}."""
    pos, neg = _split_by_label(scores, labels)
    n_pos, n_neg = pos.size, neg.size
    if n_pos == 0 or n_neg == 0:
        raise ValueError("auc requires at least one positive and one negative label")
    ranks = rankdata(np.concatenate([pos, neg]))
    rank_sum_pos = float(np.sum(ranks[:n_pos]))
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def roc_curve(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Empirical (fpr, tpr) over descending-score thresholds; both arrays start at (0, 0)."""
    pos, neg = _split_by_label(scores, labels)
    n_pos, n_neg = pos.size, neg.size
    if n_pos == 0 or n_neg == 0:
        raise ValueError("roc_curve requires at least one positive and one negative label")
    scores_all = np.concatenate([pos, neg])
    order = np.argsort(-scores_all, kind="mergesort")
    s_sorted = scores_all[order]
    y = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    # Collapse tied scores into a SINGLE threshold step: at a threshold equal to a shared score, all
    # samples at that score cross together, so the ROC must advance TPR and FPR jointly (a diagonal
    # segment). Emitting one point per sample instead ranks the concatenated positives ahead of the
    # tied negatives and makes the curve over-optimistic -- trapz(fpr, tpr) would then disagree with
    # the tie-correct Mann-Whitney `auc`. Keep only the last index of each distinct score so the two
    # agree on ties (routine for quantized/clipped forensic scores).
    last_of_score = np.concatenate([s_sorted[1:] != s_sorted[:-1], [True]])
    tpr = np.concatenate([[0.0], tp[last_of_score] / n_pos])
    fpr = np.concatenate([[0.0], fp[last_of_score] / n_neg])
    return fpr.astype(np.float64), tpr.astype(np.float64)
