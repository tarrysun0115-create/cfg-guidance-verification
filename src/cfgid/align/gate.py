"""Main-grid alignment gate: three-state decision from PREREGISTERED criteria (p0.yaml gate.*).

Pure decision logic over precomputed alignment / negative-control / sufficiency / sigma inputs.
  pass        — predictive-track alignment is significant AND rho beats every nuisance AND rho ~
                the full residual AND calibrated sigma evidence is same-direction across decoders.
  fail        — the knee does NOT increase with w* (slope <= min): the scalar-channel / sigma
                model is rejected.
  inconclusive— a positive trend exists but a gate condition is unmet (e.g. the slope CI includes
                0, |corr| is small, a nuisance baseline matches rho, or sigma evidence is only
                model-level).
The decision and its reasons are written to ``outputs/p0/decision_memo.md`` by
``cfgid.sweep.analyze``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

from cfgid.p0config import GateConfig


@dataclass(frozen=True)
class GateInputs:
    slope: float  # knee-vs-w* OLS slope
    slope_ci_lo: float
    slope_ci_hi: float
    corr: float
    predictive_track: bool  # was the regression on the predictive (unlabeled) g_sigma track?
    rho_beats_nuisance: bool  # rho AUC >= best nuisance AUC - margin, in every cell
    rho_sufficient: bool  # full-residual AUC - rho AUC <= max gap
    sigma_evidence_real: bool  # complete calibrated sigma_eff (not nominal fallback / missing)
    sigma_direction_ok: bool  # knee same-direction with sigma across the two decoders


@dataclass(frozen=True)
class GateResult:
    decision: str  # "pass" | "fail" | "inconclusive"
    reasons: list[str] = field(default_factory=list)


def evaluate_gate(inputs: GateInputs, cfg: GateConfig) -> GateResult:
    numeric_inputs = {
        "slope": inputs.slope,
        "slope_ci_lo": inputs.slope_ci_lo,
        "slope_ci_hi": inputs.slope_ci_hi,
        "corr": inputs.corr,
    }
    bad = [name for name, value in numeric_inputs.items() if not isfinite(value)]
    if bad:
        return GateResult(
            "inconclusive",
            [f"non-finite gate statistic(s): {', '.join(bad)}; rerun or inspect artifacts"],
        )

    # Missing/incomplete sigma calibration means some or all of the w* x-axis was computed with an
    # ordering-only nominal fallback or with calibrated blur levels dropped. That can be useful for
    # a diagnostic plot, but it must not drive a PASS or FAIL decision. (A scalarization downgrade
    # -- effective-only sigma -- does NOT block: the gate is rescale-invariant, testing knee-vs-w*
    # direction/monotonicity; the downgrade is surfaced as a verdict CAVEAT in the memo instead.)
    if not inputs.sigma_evidence_real:
        return GateResult(
            "inconclusive",
            [
                "sigma calibration artifact missing or incomplete; nominal/missing sigma_eff "
                "fallback cannot support a decision"
            ],
        )

    if inputs.slope <= cfg.min_knee_wstar_slope:
        return GateResult(
            "fail",
            [
                f"knee-vs-w* slope {inputs.slope:.4f} <= {cfg.min_knee_wstar_slope}: no positive "
                "alignment (scalar-channel / sigma assumption busted)"
            ],
        )

    reasons: list[str] = []
    if not (inputs.slope_ci_lo > 0.0):
        reasons.append(
            f"slope CI [{inputs.slope_ci_lo:.4f}, {inputs.slope_ci_hi:.4f}] includes 0; add samples"
        )
    if abs(inputs.corr) < cfg.min_correlation:
        reasons.append(f"|corr| {abs(inputs.corr):.3f} < {cfg.min_correlation}")
    if cfg.predictive_only_pass and not inputs.predictive_track:
        reasons.append("alignment holds only on the calibrated track (predictive track required)")
    if not inputs.rho_beats_nuisance:
        reasons.append("a nuisance baseline matches/beats rho below w* (negative control failed)")
    if not inputs.rho_sufficient:
        reasons.append("full residual separates materially better than rho (sufficiency gap)")
    if cfg.sigma_two_decoder_same_direction and not inputs.sigma_direction_ok:
        reasons.append(
            "sigma probe not same-direction across decoders (recorded as model-level only)"
        )

    if reasons:
        return GateResult("inconclusive", reasons)
    return GateResult(
        "pass",
        [
            "predictive-track knee-vs-w* alignment significant; rho beats nuisance baselines; "
            "rho ~ full residual; sigma probe consistent across decoders"
        ],
    )
