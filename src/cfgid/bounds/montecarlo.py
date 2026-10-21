"""Monte-Carlo validation of the analytic delta-deniability TV (zero GPU, seconds).

Produces: the Monte-Carlo TV check of Section III-E / Appendix B: 64 checks (the 8 guidance
levels of the p0 grid x N_eff in {1, 16, 64, 256} x {scalar, full-covariance} form), 2x10^5
draws per class and point, preset tolerance 0.02; the summary line prints the worst
|analytic - simulated| TV (0.0029 with the shipped ``configs/p0.yaml``).

Inputs: ``configs/p0.yaml`` (``grid.w`` and ``bounds.*``) only; nominal unit channel parameters,
no data, CPU only.
Outputs: console only (one line per check + a PASS/FAIL summary; exit status 1 on any drift).
Usage: python -m cfgid.bounds.montecarlo

Method: simulate the N_eff-aggregated sufficient statistic for H0 (w=1) and H1 (w), estimate TV
as the TPR - FPR gap of the Bayes-optimal equal-variance midpoint threshold (which attains TV for
two equal-variance Gaussians), and compare to ``wstar_p0.channel_tv`` (scalar) or
``wstar_cov.channel_tv_cov`` (a synthetic equicorrelated 4-coordinate channel). Seeds are fixed
per point: ``seed + 100 * i + j`` for the i-th N_eff and the j-th w (base seed 0).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from cfgid.bounds.wstar_cov import CovChannel, channel_tv_cov, mahalanobis_sq
from cfgid.bounds.wstar_p0 import WStarParams, channel_tv


@dataclass(frozen=True)
class MonteCarloCheck:
    w: float
    n_eff: float
    analytic_tv: float
    simulated_tv: float

    @property
    def abs_err(self) -> float:
        return abs(self.analytic_tv - self.simulated_tv)


def simulate_tv(
    w: float, e_tex: float, params: WStarParams, *, n_eff: float, n_samples: int, seed: int = 0
) -> float:
    """MC estimate of TV(H0=w=1, H1=w) for the N_eff-aggregated sufficient statistic."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if not np.isfinite(w):
        raise ValueError(f"w must be finite, got {w}")
    if e_tex <= 0 or not np.isfinite(e_tex):
        raise ValueError(f"e_tex must be positive and finite, got {e_tex}")
    if not (np.isfinite(n_eff) and n_eff > 0):
        raise ValueError(f"n_eff must be positive, got {n_eff}")
    mu = params.g_sigma * (w - 1.0) * e_tex
    if mu == 0.0:
        return 0.0
    sd = float(np.sqrt(params.v / n_eff))  # aggregated-statistic std
    rng = np.random.default_rng(seed)
    s0 = rng.normal(0.0, sd, n_samples)
    s1 = rng.normal(mu, sd, n_samples)
    thr = mu / 2.0  # Bayes-optimal threshold for equal-variance Gaussians
    direction = 1.0 if mu > 0 else -1.0
    p1 = float(np.mean(direction * (s1 - thr) > 0))
    p0 = float(np.mean(direction * (s0 - thr) > 0))
    return abs(p1 - p0)


def check_grid(
    params: WStarParams,
    e_tex: float,
    w_values: tuple[float, ...],
    n_eff_values: tuple[float, ...],
    *,
    n_samples: int,
    seed: int = 0,
) -> list[MonteCarloCheck]:
    out: list[MonteCarloCheck] = []
    for i, n_eff in enumerate(n_eff_values):
        for j, w in enumerate(w_values):
            out.append(
                MonteCarloCheck(
                    w=w,
                    n_eff=n_eff,
                    analytic_tv=channel_tv(w, e_tex, params, n_eff=n_eff),
                    simulated_tv=simulate_tv(
                        w, e_tex, params, n_eff=n_eff, n_samples=n_samples, seed=seed + 100 * i + j
                    ),
                )
            )
    return out


def simulate_tv_cov(
    w: float, channel: CovChannel, *, n_eff: int, n_samples: int, seed: int = 0
) -> float:
    """MC estimate of TV(H0=w=1, H1=w) for the N_eff-aggregated vector channel.

    Draws full P-dimensional vectors from N(0, N_eff*Sigma) and N(N_eff*(w-1)*d, N_eff*Sigma),
    projects onto the likelihood-ratio direction ``Sigma^{-1} d``, and thresholds at the
    Bayes-optimal midpoint — genuinely exercising Sigma, its inverse, and the projection (not just
    the scalar reduction). ``n_eff`` is integral (the grid uses 1/16/64/256).
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if not (isinstance(n_eff, (int, np.integer)) and n_eff >= 1):
        raise ValueError(f"n_eff must be a positive integer for the cov MC, got {n_eff}")
    if not np.isfinite(w):
        raise ValueError(f"w must be finite, got {w}")
    d = np.asarray(channel.d, dtype=np.float64).reshape(-1)
    cov = np.asarray(channel.cov, dtype=np.float64)
    m = mahalanobis_sq(d, cov)  # d^T Sigma^-1 d
    if w == 1.0:
        return 0.0
    lr_dir = cho_solve(cho_factor(cov, lower=True), d)  # Sigma^{-1} d
    rng = np.random.default_rng(seed)
    # Sample the aggregated N_eff-sum vector ~ N(mean, N_eff*Sigma) via explicit Cholesky (avoids
    # numpy multivariate_normal's spurious internal BLAS warnings). cov(z @ L^T)=L L^T=Sigma.
    chol = np.linalg.cholesky(cov) * float(np.sqrt(n_eff))
    mean1 = float(n_eff) * (w - 1.0) * d
    p = d.size
    # Inputs are provably finite (Gaussian draws x a finite Cholesky factor); some BLAS backends
    # (macOS Accelerate) raise SPURIOUS FP flags (divide/invalid/overflow) on large matmuls.
    # Suppress those known-false flags locally, then assert finiteness so a REAL NaN still raises.
    with np.errstate(all="ignore"):
        s0 = rng.standard_normal((n_samples, p)) @ chol.T
        s1 = mean1 + rng.standard_normal((n_samples, p)) @ chol.T
        t0 = s0 @ lr_dir
        t1 = s1 @ lr_dir
    if not (np.all(np.isfinite(t0)) and np.all(np.isfinite(t1))):
        raise ValueError("cov MC produced non-finite projected statistics")
    thr = float(n_eff) * (w - 1.0) * m / 2.0  # Bayes midpoint in projected coordinates
    direction = 1.0 if (w - 1.0) > 0 else -1.0
    p1 = float(np.mean(direction * (t1 - thr) > 0))
    p0 = float(np.mean(direction * (t0 - thr) > 0))
    return abs(p1 - p0)


def check_grid_cov(
    channel: CovChannel,
    w_values: tuple[float, ...],
    n_eff_values: tuple[int, ...],
    *,
    n_samples: int,
    seed: int = 0,
) -> list[MonteCarloCheck]:
    out: list[MonteCarloCheck] = []
    for i, n_eff in enumerate(n_eff_values):
        for j, w in enumerate(w_values):
            out.append(
                MonteCarloCheck(
                    w=w,
                    n_eff=float(n_eff),
                    analytic_tv=channel_tv_cov(w, channel, n_eff=float(n_eff)),
                    simulated_tv=simulate_tv_cov(
                        w, channel, n_eff=n_eff, n_samples=n_samples, seed=seed + 100 * i + j
                    ),
                )
            )
    return out


def main(argv: list[str] | None = None) -> int:
    from cfgid.p0config import load_p0_config

    cfg = load_p0_config()
    params = WStarParams(g_sigma=1.0, v=1.0)  # nominal self-check parameters
    e_tex = 1.0
    w_grid = tuple(float(w) for w in cfg.grid.w)
    checks = check_grid(
        params,
        e_tex,
        w_grid,
        tuple(float(n) for n in cfg.bounds.n_eff_grid),
        n_samples=cfg.bounds.montecarlo_samples,
    )
    # Covariance-channel self-check: a synthetic correlated 4-patch channel exercises Sigma^{-1}.
    p = 4
    cov = 0.4 * np.ones((p, p)) + 0.6 * np.eye(p)  # equicorrelated, well-conditioned
    cov_channel = CovChannel(d=np.full(p, 0.5), cov=cov)
    cov_checks = check_grid_cov(
        cov_channel,
        w_grid,
        tuple(int(n) for n in cfg.bounds.n_eff_grid),
        n_samples=cfg.bounds.montecarlo_samples,
    )
    tol = cfg.bounds.montecarlo_tol
    worst = 0.0
    for label, group in (("scalar", checks), ("cov", cov_checks)):
        for c in group:
            worst = max(worst, c.abs_err)
            flag = "ok" if c.abs_err <= tol else "DRIFT"
            print(
                f"[{flag:5}] {label:6} w={c.w:>3} n_eff={c.n_eff:>5} "
                f"analytic_TV={c.analytic_tv:.4f} mc_TV={c.simulated_tv:.4f} err={c.abs_err:.4f}"
            )
    ok = worst <= tol
    print(f"\nmontecarlo: {'PASS' if ok else 'FAIL'} (worst abs err {worst:.4f}, tol {tol})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
