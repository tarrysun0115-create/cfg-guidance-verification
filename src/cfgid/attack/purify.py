"""Minimal-L2 purification: penalized Adam on a DIFFERENTIABLE surrogate of rho.

The forensic channel rho is computed by dual-branch inversion; this code never backpropagates
through the diffusion chain (frozen-backbone inference + inversion only). So the attack does NOT
differentiate rho directly. Instead it optimizes a cheap DIFFERENTIABLE surrogate ``s(image)``
(``cfgid.attack.rho_surrogate``: a linear fit of rho on physical image statistics). The attacked
images have rho below the cell's H0 mean, so the attack RAISES ``s`` (hence, to the extent the
surrogate tracks rho, rho itself) toward that level with minimal L2 distortion; an optional
per-step L2-ball projection turns it into projected gradient descent under a distortion budget.

The real rho effect is then verified by a SEPARATE FORWARD inversion (no gradient); the stored
run of Section IV-E7 used 300 Adam steps, learning rate 0.02, four restarts and penalty weight
200 (Appendix B).

This module is the generic operator (pluggable surrogate); the experiment that wires the surrogate
fit + the forward-inversion verification lives in ``cfgid.attack.deniability_experiment`` and
``scripts/analysis/run_deniability_attack.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite

import torch


@dataclass(frozen=True)
class PurifyResult:
    """Per-image result of a minimal-L2 purification."""

    x_adv: torch.Tensor  # purified image(s), same shape as input, in [0,1]
    l2: torch.Tensor  # per-image L2 distortion ||x_adv - x||_2, shape (B,)
    surrogate: torch.Tensor  # per-image final surrogate value, shape (B,)
    target: float
    reached: torch.Tensor  # per-image bool: surrogate >= target, shape (B,)
    steps: int


def _surrogate_scores(values: torch.Tensor, batch_size: int) -> torch.Tensor:
    scores = values.reshape(-1)
    if scores.numel() != batch_size:
        raise ValueError(
            f"surrogate must return one scalar per image, got {scores.numel()} "
            f"for batch {batch_size}"
        )
    return scores


def _restart_noise_like(
    x: torch.Tensor, *, scale: float, generator: torch.Generator
) -> torch.Tensor:
    # Generate restart noise from a CPU generator, then move it to the optimisation device. This
    # avoids relying on backend-local Generator support (notably MPS) while preserving seeded
    # restart determinism; restart 0 still uses exact zeros and does not touch this path.
    noise = torch.randn(x.shape, generator=generator, device=torch.device("cpu"), dtype=x.dtype)
    return (scale * noise).to(device=x.device)


def min_l2_purify(
    x: torch.Tensor,
    surrogate: Callable[[torch.Tensor], torch.Tensor],
    *,
    target: float,
    steps: int = 300,
    lr: float = 0.02,
    constraint_weight: float = 500.0,
    max_l2: float | None = None,
    tol: float = 1e-3,
    rtol: float = 0.0,
    restarts: int = 1,
    restart_init_scale: float = 0.03,
    seed: int = 0,
) -> PurifyResult:
    """Minimally perturb ``x`` in [0,1] so ``surrogate(x_adv) >= target`` (raise rho to a target).

    ``x``: (C,H,W) or (B,C,H,W) float image(s) in [0,1]. ``surrogate``: a DIFFERENTIABLE callable
    mapping a batched (B,C,H,W) tensor to per-image scalars (B,) on the rho scale (rho-like).
    Penalised objective minimised by Adam on the perturbation ``delta``:

        L = constraint_weight * mean( relu(target - s(x+delta))^2 ) + mean( delta^2 )

    the relu term drives the surrogate up to the rho target; the delta^2 term keeps the
    distortion minimal (the min-L2 part). ``reached`` is ``s >= target - max(tol, rtol*|target|)``:
    a soft-penalty optimum sits just below the hard target, and that gap SCALES with the surrogate's
    magnitude, so the absolute ``tol`` default (1e-3) only fits O(1)-scale surrogates -- for a
    surrogate on a larger scale (e.g. raw high-frequency energy) pass a RELATIVE ``rtol`` (or
    normalise the surrogate to unit scale) so ``reached`` is not silently misreported. ``max_l2``
    optionally caps the per-image perturbation to an L2 ball (a distortion budget): if the
    target is unreachable within the budget, ``reached`` is False for that image. ``max_l2`` is
    enforced as a TRUE projected-gradient constraint (not a single post-hoc projection), so the
    result is a constrained optimum OF THIS first-order solver -- but ``reached=False`` is only
    NECESSARY-not-sufficient evidence of infeasibility: it conflates a genuinely infeasible budget
    with mere Adam non-convergence (lr / steps / init). A bound-matching experiment MUST rule out
    non-convergence (restarts, more steps, inspect the achieved gap target-s) before reading
    ``reached=False`` as evidence that a bound is tight.

    ``constraint_weight`` trades reaching the target against minimal L2: too low UNDERSHOOTS the
    target, too high OVERSHOOTS the true minimal distortion. It must be tuned for the surrogate's
    scale (or normalise the surrogate to unit scale); the calling bound-matching experiment should
    report the achieved surrogate gap (target - s) alongside ``l2`` as a quality diagnostic, not
    treat ``l2`` as the minimal distortion unconditionally.

    No diffusion-chain gradient is taken: the surrogate is a plain differentiable image statistic.
    """
    if isinstance(steps, bool) or steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    lr_value = float(lr)
    constraint_weight_value = float(constraint_weight)
    if not (isfinite(lr_value) and lr_value > 0.0):
        raise ValueError(f"lr must be positive and finite, got {lr}")
    if not (isfinite(constraint_weight_value) and constraint_weight_value >= 0.0):
        raise ValueError(
            f"constraint_weight must be non-negative and finite, got {constraint_weight}"
        )
    max_l2_value: float | None = None
    if max_l2 is not None:
        max_l2_value = float(max_l2)
    if max_l2_value is not None and not (isfinite(max_l2_value) and max_l2_value > 0.0):
        raise ValueError(f"max_l2 must be positive when set, got {max_l2}")
    target_value = float(target)
    if not isfinite(target_value):
        raise ValueError(f"target must be finite, got {target}")
    if not (isfinite(tol) and tol >= 0.0):
        raise ValueError(f"tol must be finite and non-negative, got {tol}")
    if not (isfinite(rtol) and rtol >= 0.0):
        raise ValueError(f"rtol must be finite and non-negative, got {rtol}")
    if isinstance(restarts, bool) or not isinstance(restarts, int) or restarts < 1:
        raise ValueError(f"restarts must be a positive integer, got {restarts!r}")
    if not (isfinite(restart_init_scale) and restart_init_scale >= 0.0):
        raise ValueError(f"restart_init_scale must be finite and >= 0, got {restart_init_scale}")
    squeezed = x.ndim == 3
    if squeezed:
        x = x.unsqueeze(0)
    if x.ndim != 4:
        raise ValueError(f"x must be (C,H,W) or (B,C,H,W), got shape {tuple(x.shape)}")
    if not torch.is_floating_point(x):
        raise ValueError("x must be a floating point tensor")
    if not torch.isfinite(x).all():
        raise ValueError("x must be finite")
    if float(x.min()) < 0.0 or float(x.max()) > 1.0:
        # No silent clamp of the baseline: l2 is measured against x, so a clamped input would make
        # the reported distortion relative to an altered baseline, not the user's actual image.
        raise ValueError("x must be in [0, 1] (no silent clamping of the baseline)")
    x = x.detach()
    batch_size = x.shape[0]
    view = (-1, *([1] * (x.ndim - 1)))  # broadcast a per-image scalar over (B,C,H,W)
    reached_tol = max(float(tol), float(rtol) * abs(target_value))

    def _optimise(delta_init: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        delta = delta_init.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([delta], lr=lr_value)
        for _ in range(steps):
            opt.zero_grad()
            x_adv = (x + delta).clamp(0.0, 1.0)
            s = _surrogate_scores(surrogate(x_adv), batch_size)
            loss = (
                constraint_weight_value * torch.relu(target_value - s).pow(2).mean()
                + delta.pow(2).mean()
            )
            loss.backward()
            opt.step()
            if max_l2_value is not None:
                # TRUE projected-gradient: project delta into the per-image L2 ball EVERY step
                # (clamp shrinks ||x_adv - x|| <= ||delta||, so it stays in budget). A single
                # post-hoc projection of the unconstrained optimum is suboptimal, so its
                # reached=False would NOT rigorously prove the target is unreachable in budget.
                with torch.no_grad():
                    norm = delta.flatten(1).norm(dim=1)
                    delta.mul_((max_l2_value / norm.clamp_min(1e-12)).clamp(max=1.0).view(view))
        with torch.no_grad():
            x_adv = (x + delta).clamp(0.0, 1.0)
            s = _surrogate_scores(surrogate(x_adv), batch_size)
            l2 = (x_adv - x).flatten(1).norm(dim=1)
        return x_adv.detach(), s.detach(), l2.detach()

    # MULTI-RESTART for a robust minimal-L2: restart 0 starts at delta=0 (deterministic), restarts
    # 1..K-1 add a small SEEDED random kick to escape the local optimum a single Adam run lands in
    # (the source of the cross-device l2 divergence). PER IMAGE keep the candidate with the smallest
    # L2 among those that REACH the target; else keep the one closest to it (smallest gap).
    gen: torch.Generator | None = None
    best_x_adv = best_s = best_l2 = None
    best_reached: torch.Tensor | None = None
    best_gap: torch.Tensor | None = None
    for r in range(restarts):
        if r == 0 or restart_init_scale == 0.0:
            delta_init = torch.zeros_like(x)
        else:
            if gen is None:
                gen = torch.Generator(device=torch.device("cpu")).manual_seed(int(seed))
            delta_init = _restart_noise_like(x, scale=restart_init_scale, generator=gen)
        x_adv, s, l2 = _optimise(delta_init)
        reached = s >= target_value - reached_tol
        gap = (target_value - s).clamp_min(0.0)
        if best_gap is None:
            best_x_adv, best_s, best_l2 = x_adv, s, l2
            best_reached, best_gap = reached, gap
        else:
            assert (
                best_x_adv is not None
                and best_s is not None
                and best_l2 is not None
                and best_reached is not None
            )
            # Per-image ordering: a reached candidate beats an unreached one; reached-vs-reached
            # picks lower L2; unreached-vs-unreached picks smaller positive target gap. This avoids
            # a fixed BIG sentinel, which can overflow lower-precision dtypes and lose gap ordering.
            improve = (
                (reached & ~best_reached)
                | (reached & best_reached & (l2 < best_l2))
                | (~reached & ~best_reached & (gap < best_gap))
            )
            best_x_adv = torch.where(improve.view(view), x_adv, best_x_adv)
            best_s = torch.where(improve, s, best_s)
            best_l2 = torch.where(improve, l2, best_l2)
            best_reached = torch.where(improve, reached, best_reached)
            best_gap = torch.where(improve, gap, best_gap)

    assert best_x_adv is not None and best_s is not None and best_l2 is not None
    x_adv, s, l2 = best_x_adv, best_s, best_l2
    reached = s >= target_value - reached_tol
    if not (torch.isfinite(x_adv).all() and torch.isfinite(s).all() and torch.isfinite(l2).all()):
        raise ValueError("purification produced non-finite output")
    if squeezed:
        x_adv = x_adv.squeeze(0)
    return PurifyResult(
        x_adv=x_adv, l2=l2, surrogate=s, target=target_value, reached=reached, steps=steps
    )
