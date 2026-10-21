"""Surrogate-attack engine: purify toward the cell's null mean and verify the REAL rho effect.

This is the testable engine for the surrogate purification attack of Section IV-E7 (settings in
Appendix B; the runnable CLI with the real inversion backend lives in
``scripts/analysis/run_deniability_attack.py``). The flow per image:

  1. fit a differentiable rho-surrogate on the cell's evaluation (TEST) rows (``rho_surrogate``;
     done by the CLI),
  2. raise the surrogate by the targeted rho gap with minimal L2 (``attack.purify.min_l2_purify``)
     -- the attacked H1 images lie below the cell's H0 mean rho, so the attack raises rho toward it,
  3. round-trip the purified image through an 8-bit PNG (the exact bytes the analyst would invert),
  4. measure the REAL rho of the purified image by a SEPARATE forward inversion (injected
     ``RhoMeasurer``; NO diffusion-chain gradient -- the surrogate is the only differentiable part),
  5. compare the achieved real-rho shift + L2 distortion against the targeted shift and, when
     supplied, the model-implied mean-shift gap ``required_rho_shift`` of
     ``bounds.deniability_budget``.

The per-image ``surrogate_reached`` / ``deniable`` flags are pooled by
``scripts/analysis/attack_tradeoff.py`` into the two fractions reported in Section IV-E7. The field
name ``deniable`` (and the pooled ``frac_deniable``) is kept for the stored outputs, but it only
records that the rho target was reached: the image's real rho rose by at least the targeted gap,
i.e. up to the cell's empirical H0 mean. No post-attack distribution or TV is estimated, so the
flag carries no per-image undetectability guarantee.

The backend is injected as a ``RhoMeasurer`` so the engine is fully unit-testable with a synthetic
rho function; the real run wires ``get_device()`` + the inversion backend in the CLI. No diffusion/
inversion code is imported here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from cfgid.attack.purify import min_l2_purify
from cfgid.attack.rho_surrogate import RhoSurrogate

# (image_hwc float [0,1], prompt, seed) -> scalar real rho of that exact image (forward inversion).
RhoMeasurer = Callable[[np.ndarray, str, int], float]


def quantize_png_roundtrip(image_hwc: np.ndarray) -> np.ndarray:
    """Exact float image represented by the 8-bit PNG of ``image_hwc`` (matches run_p0._save_png).

    l2 and the inverted rho must be measured against the SAME bytes the analyst would see, so the
    attack's distortion is reported relative to the quantised image actually persisted, not the
    pre-quantisation float optimum.
    """
    arr = np.asarray(image_hwc, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"image must be (H,W,3), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("image must be finite")
    u8 = np.clip(np.rint(arr * 255.0), 0.0, 255.0).astype(np.uint8)
    return u8.astype(np.float32) / 255.0


@dataclass(frozen=True)
class AttackRecord:
    """Per-image surrogate-attack result."""

    image_id: str
    l2: float  # purified-vs-original L2 distortion (on the PNG-quantised image)
    rho_before: float  # real rho of the original image
    rho_after: float  # real rho of the purified image (forward inversion)
    surrogate_before: float
    surrogate_after: float
    target_surrogate: float
    surrogate_reached: bool
    required_rho_shift: float  # the rho gap the attack TARGETS (empirical: H0_mean_rho - image_rho)
    achieved_rho_shift: float  # measured rho_after - rho_before (signed; should be > 0 = raised)
    # "rho target reached": achieved_rho_shift >= required_rho_shift (real rho reached the empirical
    # null mean). The stored field name is kept; it is not a per-image undetectability or TV result.
    deniable: bool
    # The bound's model-implied mean-shift gap (Gaussian-location, shared-variance model; measured-
    # Etex path: max(0, E_tex_meas*(w0-w*))), for the distortion-vs-w*-gap matching. None when no
    # bound params are supplied (the empirical target still drives the attack). Kept None (not NaN)
    # so the JSON stays allow_nan=False safe.
    bound_required_rho_shift: float | None = None
    # Convergence quality: the residual surrogate gap max(0, target - surrogate_after) on the
    # quantised bytes. ~0 = the optimiser reached the target; large = non-converged / infeasible, so
    # this image's l2 is NOT a trustworthy minimal distortion. summarize reports it per cell.
    surrogate_gap: float = 0.0


def _to_bchw(image_hwc: np.ndarray) -> torch.Tensor:
    t = torch.tensor(np.asarray(image_hwc, dtype=np.float32))
    return t.permute(2, 0, 1).unsqueeze(0)


def _project_quantized_to_l2(
    original_hwc: np.ndarray, candidate_hwc: np.ndarray, *, max_l2: float
) -> np.ndarray:
    """Largest 8-bit point on the original->candidate segment whose measured L2 is in budget."""
    current_l2 = float(np.linalg.norm((candidate_hwc - original_hwc).reshape(-1)))
    if current_l2 <= max_l2 + 1e-6:
        return candidate_hwc
    delta = candidate_hwc - original_hwc
    best = original_hwc
    lo, hi = 0.0, 1.0
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        cand = quantize_png_roundtrip(original_hwc + mid * delta)
        cand_l2 = float(np.linalg.norm((cand - original_hwc).reshape(-1)))
        if cand_l2 <= max_l2 + 1e-6:
            best = cand
            lo = mid
        else:
            hi = mid
    return best


def attack_image(
    image_hwc: np.ndarray,
    surrogate: RhoSurrogate,
    *,
    image_id: str,
    prompt: str,
    seed: int,
    measurer: RhoMeasurer,
    required_rho_shift: float,
    rho_before: float | None = None,
    bound_required_rho_shift: float | None = None,
    max_l2: float | None = None,
    device: torch.device | str | None = None,
    steps: int = 300,
    lr: float = 0.02,
    constraint_weight: float = 200.0,
    tol: float = 1e-3,
    rtol: float = 0.02,
    restarts: int = 1,
) -> AttackRecord:
    """Purify one image to raise its rho-surrogate by ``required_rho_shift``, then verify real rho.

    ``required_rho_shift`` is the rho gap the attack TARGETS (the empirical gap to the cell's null
    mean, ``H0_mean_rho - image_rho``); the surrogate target is
    ``surrogate_before + required_rho_shift`` (the surrogate is fit to approximate rho, so a unit of
    surrogate is a unit of rho). ``bound_required_rho_shift`` (optional) is the bound's
    model-implied mean-shift gap, recorded for the distortion-vs-w*-gap matching (None = no bound
    params). ``rho_before`` may be passed (e.g. the persisted rho_summary) to avoid a redundant
    baseline inversion. The returned ``deniable`` flag means only that the rho target was reached.
    """
    if not (np.isfinite(required_rho_shift) and required_rho_shift >= 0.0):
        raise ValueError(f"required_rho_shift must be finite and >= 0, got {required_rho_shift}")
    if bound_required_rho_shift is not None and not (
        np.isfinite(bound_required_rho_shift) and bound_required_rho_shift >= 0.0
    ):
        raise ValueError(
            f"bound_required_rho_shift must be None or finite >= 0, got {bound_required_rho_shift}"
        )
    x_hwc = quantize_png_roundtrip(image_hwc)  # measure everything against the persisted bytes
    # The differentiable optimisation (surrogate forward+backward) runs on ``device`` (CUDA/MPS on a
    # real run via get_device(); the band feature set's pyramid convs are too slow on CPU). The PNG
    # round-trip, l2, and the forward-inversion measurer stay on numpy/CPU bytes -- only the
    # optimisation moves; results are pulled back to CPU. No diffusion grad is taken either way.
    x = _to_bchw(x_hwc) if device is None else _to_bchw(x_hwc).to(device)
    with torch.no_grad():
        s_before = float(surrogate(x).item())
    # Surrogate target: raise s by the targeted rho gap (the surrogate is fit in rho units).
    target = s_before + float(required_rho_shift)

    res = min_l2_purify(
        x.squeeze(0),
        surrogate,
        target=target,
        steps=steps,
        lr=lr,
        constraint_weight=constraint_weight,
        max_l2=max_l2,
        tol=tol,
        rtol=rtol,
        restarts=restarts,
        seed=seed,
    )
    adv_hwc = quantize_png_roundtrip(res.x_adv.detach().to("cpu").permute(1, 2, 0).numpy())
    if max_l2 is not None:
        adv_hwc = _project_quantized_to_l2(x_hwc, adv_hwc, max_l2=float(max_l2))
    adv = _to_bchw(adv_hwc) if device is None else _to_bchw(adv_hwc).to(device)
    with torch.no_grad():
        s_after = float(surrogate(adv).item())
    l2 = float(np.linalg.norm((adv_hwc - x_hwc).reshape(-1)))

    rho0 = float(measurer(x_hwc, prompt, seed)) if rho_before is None else float(rho_before)
    rho1 = float(measurer(adv_hwc, prompt, seed))
    finite_values = (s_before, s_after, target, l2, rho0, rho1)
    if not all(np.isfinite(v) for v in finite_values):
        raise ValueError("attack produced a non-finite surrogate, rho, target, or l2")
    achieved = rho1 - rho0
    reached_tol = max(float(tol), float(rtol) * abs(target))
    return AttackRecord(
        image_id=image_id,
        l2=l2,
        rho_before=rho0,
        rho_after=rho1,
        surrogate_before=s_before,
        surrogate_after=s_after,
        target_surrogate=target,
        surrogate_reached=bool(s_after >= target - reached_tol),
        required_rho_shift=float(required_rho_shift),
        achieved_rho_shift=achieved,
        # stored field name kept: True = rho target reached (no per-image undetectability claim)
        deniable=bool(achieved >= required_rho_shift),
        bound_required_rho_shift=(
            None if bound_required_rho_shift is None else float(bound_required_rho_shift)
        ),
        surrogate_gap=float(max(0.0, target - s_after)),
    )


def summarize_attack(records: list[AttackRecord]) -> dict:
    """Aggregate per-image records into the per-cell attack summary (distortion + crossing)."""
    if not records:
        return {"n": 0}
    l2 = np.array([r.l2 for r in records], dtype=np.float64)
    req = np.array([r.required_rho_shift for r in records], dtype=np.float64)
    ach = np.array([r.achieved_rho_shift for r in records], dtype=np.float64)
    if not (np.all(np.isfinite(l2)) and np.all(np.isfinite(req)) and np.all(np.isfinite(ach))):
        raise ValueError("attack records contain non-finite numeric values")
    deniable = np.array([r.deniable for r in records], dtype=bool)

    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0

    gap = np.array([r.surrogate_gap for r in records], dtype=np.float64)
    reached = np.array([r.surrogate_reached for r in records], dtype=bool)
    if not np.all(np.isfinite(gap)):
        raise ValueError("attack records contain non-finite surrogate_gap")
    out = {
        "n": len(records),
        "mean_l2": float(l2.mean()),
        "median_l2": float(np.median(l2)),
        "mean_required_rho_shift": float(req.mean()),
        "mean_achieved_rho_shift": float(ach.mean()),
        # fraction whose real rho reached the target (stored key name kept)
        "frac_deniable": float(deniable.mean()),
        "frac_rho_raised": float((ach > 0).mean()),
        # Convergence quality: frac that reached the surrogate target + the mean residual gap. A low
        # frac_surrogate_reached / large gap flags non-converged cells whose mean_l2 is NOT a
        # trustworthy minimal distortion (read the bound-matching only where the attack converged).
        "frac_surrogate_reached": float(reached.mean()),
        "mean_surrogate_gap": float(gap.mean()),
        "corr_l2_vs_required_shift": _corr(l2, req),
    }
    # Bound matching: present only when EVERY record carries the analytic bound gap (the stored
    # Section IV-E7 run used the empirical target only, so these keys are absent there).
    # corr_l2_vs_bound = does the minimal distortion scale with the bound's predicted budget;
    # corr_empirical_vs_bound = does the bound predict the per-image empirical gap.
    bound = [r.bound_required_rho_shift for r in records]
    if all(b is not None for b in bound):
        bnd = np.array(bound, dtype=np.float64)
        if not np.all(np.isfinite(bnd)):
            raise ValueError("attack records contain non-finite bound_required_rho_shift")
        out["mean_bound_required_rho_shift"] = float(bnd.mean())
        out["corr_l2_vs_bound_required"] = _corr(l2, bnd)
        out["corr_empirical_vs_bound_gap"] = _corr(req, bnd)
    return out
