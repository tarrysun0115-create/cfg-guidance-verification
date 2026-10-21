"""Lightweight rho-sufficiency check: does the full inversion residual beat the scalar rho?

If a shallow detector on the FULL inversion residual (``residual_feature_v1``) separates H0/H1
materially better than the scalar ``rho`` does, compressing to rho threw away signal -> the gate
returns ``inconclusive`` (not pass). This is a cheap check, not the channel ablation of Table X
(``scripts/analysis/feature_comparison.py``). The shallow detector is Fisher LDA with a 1e-3
ridge (no sklearn); ``fisher_lda_direction`` / ``project_lda`` are also used, fit-split / eval-split
isolated, by ``cfgid.sweep.evidence`` (the "Rho sufficiency" block of the decision memo).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cfgid.detector.empirical import auc
from cfgid.numerics import as_finite_float32


def fisher_lda_direction(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Fit the Fisher LDA direction on (N, D) features. labels in {0, 1}. Returns (D,) float64."""
    x = as_finite_float32("features", features)
    if x.ndim != 2:
        raise ValueError(f"features must be (N, D); got {x.shape}")
    y = np.asarray(labels).reshape(-1)
    if y.shape[0] != x.shape[0]:
        raise ValueError("features and labels length mismatch")
    x0, x1 = x[y == 0], x[y == 1]
    if x0.shape[0] < 2 or x1.shape[0] < 2:
        raise ValueError("Fisher LDA needs >=2 samples per class")
    mu0, mu1 = x0.mean(axis=0), x1.mean(axis=0)
    # Pooled within-class covariance with ridge regularization for stability.
    cov = (np.cov(x0, rowvar=False) + np.cov(x1, rowvar=False)) / 2.0
    cov = np.atleast_2d(cov) + 1e-3 * np.eye(x.shape[1], dtype=np.float64)
    return np.linalg.solve(cov, (mu1 - mu0).astype(np.float64))


def project_lda(features: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Project (N, D) features onto a fitted LDA direction -> (N,) float64 scores."""
    x = as_finite_float32("features", features)
    if x.ndim != 2 or x.shape[1] != direction.shape[0]:
        raise ValueError(f"features {x.shape} incompatible with direction {direction.shape}")
    # np.errstate: silence numpy's known spurious "divide by zero in matmul" FP warning; the result
    # is then explicitly checked for finiteness (a real NaN/Inf would still be caught).
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        scores = np.ascontiguousarray(x, dtype=np.float64) @ direction
    if not np.all(np.isfinite(scores)):
        raise ValueError("Fisher LDA produced non-finite scores")
    return scores.astype(np.float64)


def fisher_lda_scores(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Fit + project in one step (in-sample). Split-isolated callers fit/project separately."""
    return project_lda(features, fisher_lda_direction(features, labels))


@dataclass(frozen=True)
class RhoSufficiency:
    rho_auc: float
    full_residual_auc: float

    @property
    def auc_gap(self) -> float:
        """Full-residual AUC minus rho AUC; large positive => rho is NOT sufficient."""
        return self.full_residual_auc - self.rho_auc

    def is_sufficient(self, max_gap: float) -> bool:
        return self.auc_gap <= max_gap


def rho_sufficiency_gap(
    rho_scores: np.ndarray,
    residual_features: np.ndarray,
    labels: np.ndarray,
) -> RhoSufficiency:
    """Compare scalar-rho AUC against a shallow full-residual detector AUC on the same labels."""
    y = np.asarray(labels).reshape(-1)
    rho_a = auc(rho_scores, y)
    full_scores = fisher_lda_scores(residual_features, y)
    full_a = auc(full_scores, y)
    # AUC is invariant to score orientation only up to reflection; take the separating orientation.
    full_a = max(full_a, 1.0 - full_a)
    rho_a = max(rho_a, 1.0 - rho_a)
    return RhoSufficiency(rho_auc=float(rho_a), full_residual_auc=float(full_a))
