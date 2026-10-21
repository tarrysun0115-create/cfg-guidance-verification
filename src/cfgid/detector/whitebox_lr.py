"""Controllable-FPR white-box likelihood-ratio detector for the scalar rho channel.

The forensic analyst is white-box: it knows the Gaussian-location channel
``rho | w ~ N(m0 + mu(w), v)`` per image (Eq. (9); H0: w=1, mean m0; H1: w=w0, mean m0 + mu). The
upstream caller owns the texture convention used to predict ``mu`` (for the measured-Etex path,
that convention must not multiply plugin ``g_sigma`` again). For a Gaussian channel with known
variance the log-likelihood ratio (Eq. (23); sign-aligned form Eq. (24)) is increasing in the
sign-aligned rho, so after ``N_eff``-image averaging (Eq. (14)) the Neyman-Pearson test simply
thresholds the aggregated mean.

ANALYTIC FPR: the decision threshold is set from the H0 fit-split moments alone (predictive -- no
H1 labels) to hit a target false-positive rate ``alpha``:

    tau(alpha, N_eff) = m0 + Phi^{-1}(1 - alpha) * sqrt(v / N_eff),

so for a Gaussian H0 with KNOWN ``m0`` and ``v`` the FPR is alpha by construction and the analyst
dials it. Here ``m0`` and ``v`` are estimated from the fit split, so even under the Gaussian model
the realised FPR equals alpha only up to estimation error. The detector's analytic power at
guidance ``w`` is

    power(w) = Phi( sqrt(N_eff)*mu(w)/sqrt(v) - Phi^{-1}(1-alpha) ).

Any test's ``power - FPR`` is at most the TV between its H0 and H1 laws, so at or below w* (where
the Gaussian rho-channel TV <= delta, Corollary 1) this detector's ``power - alpha`` cannot exceed
delta; above w* it can. This is a statement about the rho channel only.

SIGN-AWARE: the CFG-response sign of rho is decoder-specific (Section IV-E4). The detector operates
on the channel-aligned statistic ``sign*rho`` so H1 always raises the aggregated mean, where
``sign in {+1,-1}`` is the cell's channel direction (anchored on the fit split, like ``mu``). Sign
alignment leaves the variance (var(sign*rho) = var(rho)) and the STANDARDIZED cutoff
``Phi^{-1}(1 - alpha)`` unchanged; the raw threshold ``tau`` is not sign-invariant, because ``m0``
is estimated on the aligned scores. Alignment only points the one-sided test at the correct tail
(otherwise a cell where CFG lowers rho would be tested in the wrong direction and show ~zero
power). The realised FPR is a separate question. The analytic threshold is GAUSSIAN, and a
non-Gaussian H0 can put more or less than ``alpha`` beyond it whether or not H0 is symmetric (a
unit-variance uniform H0, for example, puts about 0.025 beyond the alpha=0.05 cutoff). rho is a
log-ratio and is empirically skewed (non-Gaussian, Section III-D); for ``sign=-1`` the Gaussian
threshold probes the short tail and the realised FPR can drift below ``alpha``. Averaging over
N_eff images moves the aggregated mean toward a Gaussian (CLT), which shrinks but does not remove
this gap. The held-out ``fpr_controlled`` flag (measured on test-split H0) is therefore an
EMPIRICAL check of this module's thresholds, not a finite-sample guarantee; callers must gate any
controllable-FPR statement about this detector on ``fpr_controlled``.

The paper's finite-sample FPR control comes instead from distribution-free conformal thresholds
calibrated on the independent CAL split (Section IV-D); this module backs the diagnostic scripts
``scripts/analysis/run_whitebox_lr.py`` and ``cross_decoder.py``.

Pure numpy/scipy; consumes persisted per-image ``rho_summary`` (no re-inversion). The threshold uses
only H0 statistics; H1 is used solely to MEASURE power for validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, comb, floor, lgamma, log
from operator import index

import numpy as np
from scipy.stats import norm


def _positive_index(name: str, value: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    try:
        out = index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from exc
    if out < 1:
        raise ValueError(f"{name} must be >= 1, got {out}")
    return out


def fpr_threshold(h0_mean: float, h0_var: float, alpha: float, n_eff: float) -> float:
    """Analytic one-sided threshold on the N_eff-aggregated mean rho for a target FPR ``alpha``."""
    if not np.isfinite(h0_mean):
        raise ValueError(f"h0_mean must be finite, got {h0_mean}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not (np.isfinite(h0_var) and h0_var > 0.0):
        raise ValueError(f"h0_var must be positive and finite, got {h0_var}")
    if not (np.isfinite(n_eff) and n_eff > 0.0):
        raise ValueError(f"n_eff must be positive, got {n_eff}")
    out = float(h0_mean + norm.ppf(1.0 - alpha) * np.sqrt(h0_var / n_eff))
    if not np.isfinite(out):  # e.g. an extreme alpha pushes Phi^{-1}(1-alpha) to +/-inf
        raise ValueError(f"fpr_threshold is non-finite for alpha={alpha}, n_eff={n_eff}")
    return out


def analytic_power(mu: float, h0_var: float, alpha: float, n_eff: float) -> float:
    """Phi( sqrt(N_eff)*mu/sqrt(v) - Phi^{-1}(1-alpha) ): the LR power at signal shift ``mu``."""
    if not np.isfinite(mu):
        raise ValueError(f"mu must be finite, got {mu}")
    if mu < 0.0:
        raise ValueError(f"mu must be non-negative for the high-rho one-sided detector, got {mu}")
    if not (np.isfinite(h0_var) and h0_var > 0.0):
        raise ValueError(f"h0_var must be positive and finite, got {h0_var}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not (np.isfinite(n_eff) and n_eff > 0.0):
        raise ValueError(f"n_eff must be positive, got {n_eff}")
    z = np.sqrt(n_eff) * mu / np.sqrt(h0_var)
    return float(norm.cdf(z - norm.ppf(1.0 - alpha)))


def _group_means(scores: np.ndarray, k: int, n_groups: int, rng: np.random.Generator) -> np.ndarray:
    """``n_groups`` bootstrap means of K independent draws (the N_eff=K aggregated statistic)."""
    k = _positive_index("k", k)
    n_groups = _positive_index("n_groups", n_groups)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if s.size == 0:
        raise ValueError("_group_means needs non-empty scores")
    if not np.all(np.isfinite(s)):
        raise ValueError("_group_means scores must be finite")
    return s[rng.integers(0, s.size, size=(n_groups, k))].mean(axis=1)


def _rate_interval(alpha: float, fpr_tol: float) -> tuple[float, float]:
    upper = alpha + fpr_tol
    if upper <= 0.0:
        return 1.0, 0.0
    return max(0.0, alpha - fpr_tol), upper


def _positive_uniform_rate_can_land_within_tol(
    *, support_count: int, alpha: float, fpr_tol: float
) -> bool:
    """Whether a positive m/support_count rate can land in the target interval."""
    support_count = _positive_index("support_count", support_count)
    lower, upper = _rate_interval(alpha, fpr_tol)
    if lower > upper:
        return False
    width = upper - lower
    if width > 0.0 and log(width) + log(support_count) >= 0.0:
        # A closed interval whose scaled width is at least one always contains an integer.
        return True
    hi_f = upper * support_count
    lo_f = lower * support_count
    if not (np.isfinite(hi_f) and np.isfinite(lo_f)):
        return False
    lo = max(1, ceil(lo_f - 1.0e-12))
    hi = floor(hi_f + 1.0e-12)
    return lo <= hi


def _positive_tail_prob_can_land(tail_probs: np.ndarray, *, alpha: float, fpr_tol: float) -> bool:
    lower, upper = _rate_interval(alpha, fpr_tol)
    if lower > upper:
        return False
    probs = np.asarray(tail_probs, dtype=np.float64).reshape(-1)
    if probs.size == 0:
        return False
    if not np.all(np.isfinite(probs)):
        raise ValueError("tail probabilities must be finite")
    positive = probs[probs > 0.0]
    if positive.size == 0:
        return False
    return bool(np.any((lower - 1.0e-12 <= positive) & (positive <= upper + 1.0e-12)))


def _float64_exact_integer_keys(values: np.ndarray) -> list[int]:
    """Integer keys for exact ordering/equality of finite float64 values."""
    ratios: list[tuple[int, int]] = []
    max_exp = 0
    for value in values:
        numerator, denominator = float(value).as_integer_ratio()
        if denominator <= 0 or denominator & (denominator - 1):
            raise ValueError(f"float64 value has non-binary denominator {denominator}")
        exp = denominator.bit_length() - 1
        ratios.append((numerator, exp))
        max_exp = max(max_exp, exp)
    return [numerator << (max_exp - exp) for numerator, exp in ratios]


def _tail_grid_guaranteed_to_land_from_max_jump(
    max_atom_mass: float, *, alpha: float, fpr_tol: float
) -> bool:
    """Conservative sufficient check from an upper bound on any one tail-grid jump.

    As a threshold sweeps upward over a finite bootstrap distribution, the upper-tail probability
    falls from 1 to 0 by jumps equal to the atom masses at aggregate levels. If every jump is no
    wider than the requested interval, the sweep cannot jump over that interval without landing in
    it. For intervals starting at zero, it is enough that the final positive tail (the top atom) is
    no larger than the upper endpoint.
    """
    lower, upper = _rate_interval(alpha, fpr_tol)
    if lower > upper:
        return False
    atom = float(max_atom_mass)
    if not (np.isfinite(atom) and 0.0 < atom <= 1.0):
        raise ValueError(f"max_atom_mass must be in (0, 1], got {max_atom_mass}")
    if upper >= 1.0:
        return lower <= 1.0
    if lower == 0.0:
        return atom <= upper + 1.0e-12
    width = upper - lower
    if width <= 0.0:
        return False
    return atom <= width + 1.0e-12


def _exact_bootstrap_tail_can_land(
    h0_test: np.ndarray,
    k: int,
    *,
    alpha: float,
    fpr_tol: float,
    max_ordered_tuples: int = 1_000_000,
) -> bool | None:
    """Exact positive bootstrap tail grid when the unique-value K-product is small enough."""
    scores = np.asarray(h0_test, dtype=np.float64).reshape(-1)
    if scores.size == 0 or not np.all(np.isfinite(scores)):
        raise ValueError("h0_test must be non-empty and finite")
    k = _positive_index("k", k)
    values, counts = np.unique(scores, return_counts=True)
    support_size = values.size**k
    if support_size > max_ordered_tuples:
        return None
    probs = counts.astype(np.float64) / float(scores.size)
    if values.size == 1:
        return _positive_tail_prob_can_land(
            np.array([1.0], dtype=np.float64), alpha=alpha, fpr_tol=fpr_tol
        )

    value_keys = _float64_exact_integer_keys(values)
    prob_values = [float(prob) for prob in probs]
    dist: dict[int, float] = {0: 1.0}
    for _ in range(k):
        next_dist: dict[int, float] = {}
        for partial_key, partial_mass in dist.items():
            for value_key, prob in zip(value_keys, prob_values, strict=True):
                key = partial_key + value_key
                next_dist[key] = next_dist.get(key, 0.0) + partial_mass * prob
        dist = next_dist

    # The keys are exact sums of the input float64 values after scaling to a common power-of-two
    # denominator. This preserves true distinct levels even when they are only a few ULPs apart,
    # while still merging permutations and additive coincidences into their exact aggregate level.
    level_masses = np.array([dist[key] for key in sorted(dist)], dtype=np.float64)
    mass_total = float(np.sum(level_masses, dtype=np.float64))
    if not np.all(np.isfinite(level_masses)) or not np.isfinite(mass_total):
        raise ValueError("exact bootstrap produced non-finite masses")
    if not np.isclose(mass_total, 1.0, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError(f"exact bootstrap mass must sum to 1, got {mass_total}")
    tail = np.cumsum(level_masses[::-1], dtype=np.float64)
    return _positive_tail_prob_can_land(tail, alpha=alpha, fpr_tol=fpr_tol)


def _rank_sum_tail_can_land(
    h0_test: np.ndarray,
    k: int,
    *,
    alpha: float,
    fpr_tol: float,
) -> bool:
    """Conservative sufficient check for large bootstrap supports.

    Exact enumeration above is authoritative. When it is too large, do NOT treat the rank-sum tail
    probabilities as an achievable grid for the real scores: uneven score spacings can make that a
    false positive. For distinct pools, use the arithmetic-progression rank-sum distribution only as
    a worst-case upper bound on a single aggregate-level atom; if every possible tail jump is small
    enough, the true tail sweep must land in the requested interval. For tied pools, fall back to
    the always-safe one-step atom bound unless the exact unique-value path handled it already.
    """
    scores = np.asarray(h0_test, dtype=np.float64).reshape(-1)
    if scores.size == 0 or not np.all(np.isfinite(scores)):
        raise ValueError("h0_test must be non-empty and finite")
    k = _positive_index("k", k)
    _values, counts = np.unique(scores, return_counts=True)
    probs = counts.astype(np.float64) / float(scores.size)
    max_atom_bound = float(np.max(probs))
    if np.all(counts == 1):
        n = probs.size
        dist = np.array([1.0], dtype=np.float64)
        for _ in range(k):
            prefix = np.concatenate(([0.0], np.cumsum(dist, dtype=np.float64)))
            new_len = dist.size + n - 1
            idx = np.arange(new_len)
            lo = np.maximum(0, idx - n + 1)
            hi = np.minimum(dist.size - 1, idx)
            dist = (prefix[hi + 1] - prefix[lo]) / float(n)
        if not np.all(np.isfinite(dist)):
            raise ValueError("rank-sum bootstrap resolution produced non-finite probabilities")
        max_atom_bound = float(np.max(dist))
    return _tail_grid_guaranteed_to_land_from_max_jump(max_atom_bound, alpha=alpha, fpr_tol=fpr_tol)


def empirical_fpr_tpr(
    h0: np.ndarray,
    h1: np.ndarray,
    threshold: float,
    *,
    k: int,
    n_groups: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Empirical (FPR, TPR) of the threshold rule on K-aggregated H0/H1 means (held-out test)."""
    if not np.isfinite(threshold):
        raise ValueError(f"threshold must be finite, got {threshold}")
    g0 = _group_means(h0, k, n_groups, rng)
    g1 = _group_means(h1, k, n_groups, rng)
    fpr = float(np.mean(g0 > threshold))
    tpr = float(np.mean(g1 > threshold))
    return fpr, tpr


def _fpr_resolution_resolved(
    *,
    alpha: float,
    n_h0_test: int,
    n_eff: int,
    n_groups: int,
    fpr_tol: float,
    min_resolved_count: float,
    h0_test: np.ndarray | None = None,
) -> bool:
    """Whether the held-out H0 bootstrap + finite MC estimate can resolve target FPR alpha."""
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    n_eff = _positive_index("n_eff", n_eff)
    n_groups = _positive_index("n_groups", n_groups)
    if isinstance(n_h0_test, bool):
        raise ValueError(f"n_h0_test must be an integer, got {n_h0_test!r}")
    try:
        n_h0 = index(n_h0_test)
    except TypeError as exc:
        raise ValueError(f"n_h0_test must be an integer, got {n_h0_test!r}") from exc
    if not (np.isfinite(fpr_tol) and fpr_tol >= 0.0):
        raise ValueError(f"fpr_tol must be non-negative and finite, got {fpr_tol}")
    if not (np.isfinite(min_resolved_count) and min_resolved_count > 0.0):
        raise ValueError(
            f"min_resolved_count must be positive and finite, got {min_resolved_count}"
        )
    if n_h0 < 1:
        return False
    if h0_test is not None:
        h0_scores = np.asarray(h0_test, dtype=np.float64).reshape(-1)
        if h0_scores.size != n_h0:
            raise ValueError(f"h0_test size ({h0_scores.size}) must match n_h0_test ({n_h0})")
        if not np.all(np.isfinite(h0_scores)):
            raise ValueError("h0_test must contain only finite scores")
    else:
        h0_scores = None

    # The ordered bootstrap support has n**K equiprobable tuples, but the thresholded statistic is
    # the aggregate mean: permutations of the same multiset collapse to the SAME threshold level.
    # The number of distinct levels is therefore at most C(n+K-1,K). This count is a proxy floor:
    # keep it for fallback/no-score paths and zero-adjacent intervals, where tiny top-tail slivers
    # should not make alpha look resolved; for nonzero exact intervals, the enumerated tail masses
    # below are authoritative and must not be pre-empted by this coarser multiset count.
    multiset_support = comb(n_h0 + n_eff - 1, n_eff)
    log_multiset_support = lgamma(n_h0 + n_eff) - lgamma(n_eff + 1) - lgamma(n_h0)
    log_min_count = log(min_resolved_count)
    lower, _upper = _rate_interval(alpha, fpr_tol)

    # The reported fpr_emp is itself a finite Monte-Carlo estimate over n_groups sampled aggregates.
    # If alpha*n_groups is below the same count floor, fpr_emp==0 is no longer a meaningful failure
    # of FPR control; it is just an under-resolved measurement.
    if log(alpha) + log(float(n_groups)) < log_min_count:
        return False

    if h0_scores is not None:
        if log(alpha) + n_eff * log(float(n_h0)) < log_min_count:
            return False
        if lower == 0.0 and log(alpha) + log_multiset_support < log_min_count:
            return False
        exact_can_land = _exact_bootstrap_tail_can_land(
            h0_scores, n_eff, alpha=alpha, fpr_tol=fpr_tol
        )
        if exact_can_land is None:
            if log(alpha) + log_multiset_support < log_min_count:
                return False
            bootstrap_can_land = _rank_sum_tail_can_land(
                h0_scores, n_eff, alpha=alpha, fpr_tol=fpr_tol
            )
        else:
            bootstrap_can_land = exact_can_land
    else:
        if log(alpha) + log_multiset_support < log_min_count:
            return False
        bootstrap_can_land = _positive_uniform_rate_can_land_within_tol(
            support_count=multiset_support, alpha=alpha, fpr_tol=fpr_tol
        )

    return bootstrap_can_land and _positive_uniform_rate_can_land_within_tol(
        support_count=n_groups, alpha=alpha, fpr_tol=fpr_tol
    )


@dataclass(frozen=True)
class WhiteBoxLRResult:
    """Per-(cell, w, alpha, N_eff) controllable-FPR LR detector evaluation."""

    cell: str
    w: int
    sign: float  # channel direction (+1 CFG raises rho, -1 CFG lowers rho); fit-anchored
    alpha: float
    n_eff: int
    threshold: float  # on the sign-aligned statistic sign*rho
    mu_predictive: float  # |aligned| signal shift estimated upstream (>= 0)
    n_h0_test: int  # held-out H0 pool size (the FPR resolution is ~1/n_h0_test at n_eff=1)
    fpr_emp: float  # measured FPR on held-out H0 (should ~= alpha: the CONTROL)
    tpr_emp: float  # measured power on held-out H1
    power_analytic: float  # Phi(sqrt(N)*mu/sqrt(v) - z_alpha)
    fpr_resolved: bool  # alpha resolvable by pool support + finite MC? (else control is vacuous)
    fpr_controlled: bool  # fpr_resolved AND fpr_emp>0 AND |fpr_emp - alpha| within tolerance
    power_matches: bool  # |tpr_emp - power_analytic| within tolerance


def evaluate_cell(
    *,
    cell: str,
    w: int,
    h0_fit: np.ndarray,
    h0_test: np.ndarray,
    h1_test: np.ndarray,
    mu_predictive: float,
    sign: float = 1.0,
    alpha: float = 0.01,
    n_eff: int = 1,
    n_groups: int = 8000,
    fpr_tol: float = 0.01,
    power_tol: float = 0.05,
    min_resolved_count: float = 5.0,
    seed: int = 0,
) -> WhiteBoxLRResult:
    """Set the threshold from H0 (fit split) for target ``alpha``; measure FPR/power on test.

    ``mu_predictive`` is the |aligned| signal shift (>= 0; estimated upstream without H1 labels,
    under that caller's texture convention). ``sign in {+1,-1}`` is the channel direction
    (fit-anchored): the test runs on the aligned statistic ``sign*rho`` so H1 raises the aggregated
    mean. The threshold uses ONLY h0_fit (predictive). The variance and the standardized cutoff are
    sign-invariant; the raw threshold is not, because ``m0`` is estimated on the aligned scores.
    The test split measures the realised FPR (an empirical check of the Gaussian threshold, not a
    finite-sample guarantee) and the power (vs the analytic prediction).

    RESOLUTION GUARD: the empirical FPR is a finite bootstrap estimate over the held-out H0 pool.
    For ``K=n_eff`` aggregation, ordered tuples collapse to at most ``C(n_h0_test+K-1,K)``
    distinct mean levels, but those levels have non-uniform masses. Resolution therefore checks the
    multiset count floor, the actual/proxy positive tail-probability grid, AND the finite
    ``n_groups`` Monte-Carlo count. ``fpr_controlled`` requires ``fpr_resolved`` so an unresolved
    alpha is reported False, not vacuously True.
    """
    if sign not in (1.0, -1.0):
        raise ValueError(f"sign must be +1.0 or -1.0, got {sign}")
    n_eff = _positive_index("n_eff", n_eff)
    if not (np.isfinite(fpr_tol) and fpr_tol >= 0.0):
        raise ValueError(f"fpr_tol must be non-negative and finite, got {fpr_tol}")
    if not (np.isfinite(power_tol) and power_tol >= 0.0):
        raise ValueError(f"power_tol must be non-negative and finite, got {power_tol}")
    rng = np.random.default_rng(seed)
    # Align each split to the channel direction so the one-sided high-statistic test is correct.
    h0_fit = sign * np.asarray(h0_fit, dtype=np.float64).reshape(-1)
    h0_test = sign * np.asarray(h0_test, dtype=np.float64).reshape(-1)
    h1_test = sign * np.asarray(h1_test, dtype=np.float64).reshape(-1)
    if h0_fit.size < 2:
        raise ValueError("h0_fit needs at least 2 samples to estimate variance with ddof=1")
    if not np.all(np.isfinite(h0_fit)):
        raise ValueError("h0_fit must contain only finite scores")
    # H0 moments come from the fit split only; neither H1 nor the test split enters tau.
    m0 = float(np.mean(h0_fit))
    v = float(np.var(h0_fit, ddof=1))  # sign-invariant
    threshold = fpr_threshold(m0, v, alpha, n_eff)
    fpr_emp, tpr_emp = empirical_fpr_tpr(
        h0_test, h1_test, threshold, k=n_eff, n_groups=n_groups, rng=rng
    )
    power_an = analytic_power(mu_predictive, v, alpha, n_eff)
    n_h0_test = int(h0_test.size)
    fpr_resolved = _fpr_resolution_resolved(
        alpha=alpha,
        n_h0_test=n_h0_test,
        n_eff=n_eff,
        n_groups=n_groups,
        fpr_tol=fpr_tol,
        min_resolved_count=min_resolved_count,
        h0_test=h0_test,
    )
    return WhiteBoxLRResult(
        cell=cell,
        w=w,
        sign=float(sign),
        alpha=alpha,
        n_eff=n_eff,
        threshold=threshold,
        mu_predictive=mu_predictive,
        n_h0_test=n_h0_test,
        fpr_emp=fpr_emp,
        tpr_emp=tpr_emp,
        power_analytic=power_an,
        fpr_resolved=fpr_resolved,
        # fpr_emp > 0 is REQUIRED: a realised FPR of EXACTLY 0 means the threshold cleared every
        # aggregated H0 point (the skewed short-tail / inverted-channel degeneracy) -- it is never
        # "controlled at alpha>0", and the bootstrap-support resolution check alone misses it
        # (a tiny short-tailed pool can have huge support yet fpr_emp pinned at 0).
        fpr_controlled=bool(fpr_resolved and fpr_emp > 0.0 and abs(fpr_emp - alpha) <= fpr_tol),
        power_matches=bool(abs(tpr_emp - power_an) <= power_tol),
    )
