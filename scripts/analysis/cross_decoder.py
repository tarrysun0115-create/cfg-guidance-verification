"""Per-decoder AUC and cross-decoder FPR-transfer matrix for the sign-aligned rho channel.

Produces: no printed paper number directly (shared helper plus a standalone diagnostic).
``candidate_readouts.py`` and ``pair_readout_battery.py`` call ``run`` once per candidate readout;
the ``cross_decoder.json`` written here is a diagnostic and is not printed in the paper.

Inputs: only ``outputs/p0/<decoder>/p0_table.json`` (from run_p0). CPU, numpy only; no re-inversion.

Outputs: ``outputs/p0/cross_decoder.json`` and ``outputs/figures/f6_cross_decoder_fpr.png``.

Usage: python scripts/analysis/cross_decoder.py [--features-root outputs/p0] [--alpha 0.01]

On the favorable subset (cells passing the held-out gate, Section IV-E) it asks two decoder-level
questions that an AUC transfer matrix cannot answer:

  * Does a target-FPR threshold calibrated on one decoder retain target agreement on another?
    AUC is threshold-free, so the meaningful cross-decoder object is the FPR-transfer matrix at a
    fixed target alpha. Its finite null pools support a descriptive target-agreement criterion,
    not one-sided control.
  * Does EACH decoder carry signal ON ITS OWN, or is the pooled AUC carried by one decoder? The
    per-decoder single-image AUC answers this and is reported separately.

Construction (NO re-inversion; reuses the pooled selection and sign alignment EXACTLY): load
p0_table.json, ``_group_rows_by_cell``, ``collect_alive_cell_records(cells, alive_t=ALIVE_T)``, then
GROUP the alive records by ``record.decoder``. For decoder D, pool ``h0_D = concat(sign*zt0_eval)``
and ``h1_D = concat(sign*zt1_eval)`` over D's alive records (the same TEST-eval-half, fit-anchored
sign the pooled module uses), so each decoder's pool is on the common positive-shift z-scale.

  * PER-DECODER AUC: ``auc(concat(h1_D,h0_D), labels)`` -- does decoder D separate H1/H0 alone.
  * FPR-TRANSFER matrix at target ``alpha``: train on A, test on B. The threshold is set from A's
    eval-H0 ONLY (``whitebox_lr.fpr_threshold(mean(h0_A), var(h0_A), alpha, n_eff=1)`` --
    predictive, no H1 labels), then realised on B via ``whitebox_lr.empirical_fpr_tpr(h0_B, h1_B,
    threshold, k=1, ...)``. The DIAGONAL is within-decoder target agreement; the OFF-diagonal is
    cross-decoder transfer (does A's threshold over/under-fire on B because the H0 location/scale
    differs across decoders).
  * LEAVE-ONE-DECODER-OUT: threshold from the POOLED other decoders' eval-H0, realised on the
    held-out decoder -- the "calibrate on the decoders you have, face a new one" check.

Strict-JSON out, finite-guarded; a degenerate (<2-sample) decoder pool is skipped with a note rather
than crashing, and any undefined transfer entry is emitted as JSON null (never NaN/Inf). The figure
is a convenience: the JSON data product is produced whether or not matplotlib is available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cfgid import viz
from cfgid.detector import whitebox_lr
from cfgid.detector.empirical import auc
from cfgid.detector.whitebox_lr import _fpr_resolution_resolved
from cfgid.paths import outputs_dir
from cfgid.sweep.neff_scaling import _group_rows_by_cell, _load_p0_rows
from cfgid.sweep.pooled_neff_scaling import CellRecord, collect_alive_cell_records

# The pooled favorable-subset gate threshold; kept here so this and the pooled module pick one set.
ALIVE_T = 1.5
# A decoder pool needs at least this many H0 and H1 eval images to define a threshold / measure FPR.
MIN_POOL = 2
# Default |fpr_emp - alpha| tolerance for the composite target-agreement flag.
DEFAULT_FPR_TOL = 0.02

# Architecture family for descriptive within- versus cross-family target-error summaries.
# sd_unet (UNet + CLIP), dit (transformer + T5/mT5), flow (rectified flow; outside the DDIM
# grid of the paper).
_DECODER_ARCHITECTURE: dict[str, str] = {
    "sd15": "sd_unet",
    "sd21": "sd_unet",
    "sdxl": "sd_unet",
    "playground25": "sd_unet",
    "pixart_sigma": "dit",
    "hunyuandit": "dit",
    "flux1-dev": "flow",
}


def architecture_family(decoder: str) -> str:
    """Architecture family of a decoder (sd_unet / dit / flow); 'other' if unmapped."""
    return _DECODER_ARCHITECTURE.get(decoder, "other")


def per_decoder_pools(records: list[CellRecord]) -> dict[str, dict[str, np.ndarray]]:
    """Group alive records by decoder and build each decoder's SIGN-ALIGNED eval H0/H1 pool.

    For decoder ``D`` the pool concatenates ``sign * zt0_eval`` / ``sign * zt1_eval`` over D's alive
    records -- the same fit-anchored sign the pooled module flips each cell by, so every decoder's
    pool lives on the common positive-shift z-scale (H1 above H0). Decoders are returned in sorted
    key order so the transfer matrix rows/cols are deterministic.
    """
    by_decoder: dict[str, list[CellRecord]] = {}
    for r in records:
        by_decoder.setdefault(str(r.decoder), []).append(r)
    pools: dict[str, dict[str, np.ndarray]] = {}
    for decoder in sorted(by_decoder):
        recs = by_decoder[decoder]
        h0 = np.concatenate(
            [r.sign * np.asarray(r.zt0_eval, dtype=np.float64).reshape(-1) for r in recs]
        )
        h1 = np.concatenate(
            [r.sign * np.asarray(r.zt1_eval, dtype=np.float64).reshape(-1) for r in recs]
        )
        if not (np.all(np.isfinite(h0)) and np.all(np.isfinite(h1))):
            raise ValueError(f"decoder {decoder!r} pool contains non-finite eval z-scores")
        pools[decoder] = {"h0": h0, "h1": h1}
    return pools


def per_decoder_auc(pools: dict[str, dict[str, np.ndarray]]) -> list[dict]:
    """Single-image sign-aligned AUC per decoder (does each decoder separate H1/H0 alone).

    A decoder whose pool has <MIN_POOL H0 or H1 images cannot define an AUC; it is reported with a
    null AUC rather than crashing (so the data product stays strict-JSON and finite).
    """
    out: list[dict] = []
    for decoder in sorted(pools):
        h0 = pools[decoder]["h0"]
        h1 = pools[decoder]["h1"]
        n0, n1 = int(h0.size), int(h1.size)
        if n0 < MIN_POOL or n1 < MIN_POOL:
            out.append({"decoder": decoder, "n_h0": n0, "n_h1": n1, "auc": None})
            continue
        scores = np.concatenate([h1, h0])
        labels = np.concatenate([np.ones(n1, int), np.zeros(n0, int)])
        out.append({"decoder": decoder, "n_h0": n0, "n_h1": n1, "auc": float(auc(scores, labels))})
    return out


def _threshold_from_h0(h0: np.ndarray, alpha: float) -> float | None:
    """Analytic target-alpha threshold from a pooled eval-H0 (predictive); None if undefined.

    Returns None when the pool is too small to estimate a variance (ddof=1 needs >=2) or its eval-H0
    variance is non-positive (a degenerate constant pool) -- both make a Gaussian threshold
    undefined, which we surface as JSON null rather than letting fpr_threshold raise.
    """
    h0 = np.asarray(h0, dtype=np.float64).reshape(-1)
    if h0.size < MIN_POOL:
        return None
    m0 = float(np.mean(h0))
    v0 = float(np.var(h0, ddof=1))
    if not (np.isfinite(m0) and np.isfinite(v0) and v0 > 0.0):
        return None
    # An extreme alpha can drive norm.ppf(1-alpha) to +/-inf; honour the "None if undefined"
    # contract rather than letting a non-finite threshold (or a fpr_threshold ValueError) escape.
    try:
        thr = whitebox_lr.fpr_threshold(m0, v0, alpha, n_eff=1)
    except ValueError:
        return None
    return float(thr) if np.isfinite(thr) else None


def _realize(
    h0: np.ndarray, h1: np.ndarray, threshold: float, *, n_groups: int, seed: int
) -> tuple[float | None, float | None]:
    """Realise (FPR, TPR) of ``threshold`` on a test decoder pool (k=1); None if pool too small."""
    h0 = np.asarray(h0, dtype=np.float64).reshape(-1)
    h1 = np.asarray(h1, dtype=np.float64).reshape(-1)
    if h0.size < MIN_POOL or h1.size < MIN_POOL:
        return None, None
    fpr, tpr = whitebox_lr.empirical_fpr_tpr(
        h0, h1, threshold, k=1, n_groups=n_groups, rng=np.random.default_rng(seed)
    )
    return float(fpr), float(tpr)


def _resolved_for_pool(h0: np.ndarray, alpha: float, *, n_groups: int, fpr_tol: float) -> bool:
    """Whether a test decoder's eval-H0 pool + MC can RESOLVE the target alpha (k=1).

    False for a sub-MIN_POOL pool; otherwise delegates to whitebox_lr._fpr_resolution_resolved so
    the composite target-agreement criterion matches detector_eval.py and
    whitebox_lr.evaluate_cell.
    """
    h0 = np.asarray(h0, dtype=np.float64).reshape(-1)
    if h0.size < MIN_POOL:
        return False
    return _fpr_resolution_resolved(
        alpha=alpha,
        n_h0_test=int(h0.size),
        n_eff=1,
        n_groups=n_groups,
        fpr_tol=fpr_tol,
        min_resolved_count=5.0,
        h0_test=h0,
    )


def fpr_transfer_matrix(
    pools: dict[str, dict[str, np.ndarray]],
    alpha: float,
    *,
    fpr_tol: float = DEFAULT_FPR_TOL,
    n_groups: int = 20000,
    seed: int = 0,
) -> dict:
    """Train-decoder x test-decoder FPR (and TPR) transfer at target ``alpha`` (k=1, predictive).

    ``transfer_fpr[i][j]`` is the realised FPR when the threshold is calibrated on
    ``decoders[i]`` eval-H0 and applied to ``decoders[j]``; the diagonal is within-decoder target
    agreement, the off-diagonal is cross-decoder transfer. A degenerate train pool (no threshold)
    or test pool (<MIN_POOL) yields a JSON-null entry rather than NaN. The per-train-decoder
    threshold is computed ONCE and reused across all test decoders so the matrix is consistent.

    ``transfer_target_consistent[i][j]`` is a composite target-agreement flag (as in
    detector_eval.py): the test decoder's pool can resolve alpha AND ``transfer_fpr[i][j]`` is
    positive and within ``fpr_tol`` of alpha. It is not a one-sided-control certificate. The
    ``transfer_controlled`` key is retained as an identical compatibility alias. Thus an
    inverted-channel diagonal 0.000 fails the criterion (never a vacuous "good 0").
    ``transfer_resolved[j]`` is the per-test-decoder resolution flag.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    decoders = sorted(pools)
    thresholds = {d: _threshold_from_h0(pools[d]["h0"], alpha) for d in decoders}
    resolved = {
        d: _resolved_for_pool(pools[d]["h0"], alpha, n_groups=n_groups, fpr_tol=fpr_tol)
        for d in decoders
    }
    transfer_fpr: list[list[float | None]] = []
    transfer_tpr: list[list[float | None]] = []
    transfer_controlled: list[list[bool]] = []
    for i, train in enumerate(decoders):
        thr = thresholds[train]
        fpr_row: list[float | None] = []
        tpr_row: list[float | None] = []
        ctrl_row: list[bool] = []
        for j, test in enumerate(decoders):
            if thr is None:
                fpr_row.append(None)
                tpr_row.append(None)
                ctrl_row.append(False)
                continue
            # Distinct seed per (train, test) so the bootstrap MC noise is not shared across cells.
            fpr, tpr = _realize(
                pools[test]["h0"],
                pools[test]["h1"],
                thr,
                n_groups=n_groups,
                seed=seed + i * len(decoders) + j,
            )
            fpr_row.append(fpr)
            tpr_row.append(tpr)
            ctrl_row.append(
                bool(
                    resolved[test] and fpr is not None and fpr > 0.0 and abs(fpr - alpha) <= fpr_tol
                )
            )
        transfer_fpr.append(fpr_row)
        transfer_tpr.append(tpr_row)
        transfer_controlled.append(ctrl_row)
    return {
        "decoders": decoders,
        "transfer_fpr": transfer_fpr,
        "transfer_tpr": transfer_tpr,
        "transfer_target_consistent": transfer_controlled,
        "transfer_controlled": transfer_controlled,
        "transfer_resolved": [bool(resolved[d]) for d in decoders],
    }


def architecture_transfer_summary(transfer: dict, alpha: float) -> dict:
    """Aggregate the OFF-diagonal FPR-transfer entries into WITHIN- vs CROSS-architecture groups.

    For every ordered pair of distinct decoders this bins ``|transfer_fpr[i][j] - alpha|`` and the
    ``transfer_target_consistent`` composite flag by whether the pair shares an architecture
    family, then reports descriptive mean/median target error and criterion fraction. Similar
    cross- and within-family errors do not establish transfer or control when destination pools
    are unresolved. Strict-JSON safe: a bin with no defined entry is JSON null.
    """
    decoders: list[str] = transfer["decoders"]
    fpr: list[list[float | None]] = transfer["transfer_fpr"]
    target_consistent: list[list[bool]] = transfer["transfer_target_consistent"]
    fam = [architecture_family(d) for d in decoders]
    within_err: list[float] = []
    cross_err: list[float] = []
    within_ctrl: list[bool] = []
    cross_ctrl: list[bool] = []
    for i in range(len(decoders)):
        for j in range(len(decoders)):
            if i == j:
                continue  # off-diagonal only: the diagonal is target agreement, not transfer
            same = fam[i] == fam[j]
            entry = fpr[i][j]
            if entry is not None:
                (within_err if same else cross_err).append(abs(float(entry) - alpha))
            (within_ctrl if same else cross_ctrl).append(bool(target_consistent[i][j]))

    def _summ(errs: list[float], ctrls: list[bool]) -> dict:
        return {
            "n_pairs": len(ctrls),
            "n_defined": len(errs),
            "mean_abs_fpr_err": float(np.mean(errs)) if errs else None,
            "median_abs_fpr_err": float(np.median(errs)) if errs else None,
            "frac_target_consistent": float(np.mean(ctrls)) if ctrls else None,
            # Compatibility alias for existing artifacts and downstream readers.
            "frac_controlled": float(np.mean(ctrls)) if ctrls else None,
        }

    return {
        "alpha": float(alpha),
        "families": dict(zip(decoders, fam, strict=True)),
        "within_architecture": _summ(within_err, within_ctrl),
        "cross_architecture": _summ(cross_err, cross_ctrl),
        "note": (
            "descriptive within- versus cross-architecture target errors; similarity does not "
            "establish transfer or one-sided FPR control, especially when target pools are "
            "unresolved"
        ),
    }


def leave_one_architecture_out(
    pools: dict[str, dict[str, np.ndarray]],
    alpha: float,
    *,
    fpr_tol: float = DEFAULT_FPR_TOL,
    n_groups: int = 20000,
    seed: int = 0,
) -> list[dict]:
    """Calibrate on ALL decoders of OTHER architecture families; realise on each held-out family.

    A descriptive cross-architecture check: pool the eval-H0 of every decoder NOT in the
    held-out architecture family, set ONE threshold, and measure the realised FPR on the held-out
    family's pooled eval images. (Cf. leave_one_decoder_out, but the held-out unit is an
    architecture. ``target_consistent`` records the composite target-agreement criterion, not
    one-sided control or proof of transfer; legacy ``controlled`` is an identical compatibility
    alias. A family with no other-architecture decoders is skipped.
    """
    by_family: dict[str, list[str]] = {}
    for d in sorted(pools):
        by_family.setdefault(architecture_family(d), []).append(d)
    out: list[dict] = []
    for held, _members in sorted(by_family.items()):
        others = [d for d in sorted(pools) if architecture_family(d) != held]
        held_decoders = by_family[held]
        if not others:
            out.append(
                {
                    "held_out_architecture": held,
                    "n_calib_decoders": 0,
                    "fpr": None,
                    "target_consistent": False,
                    "controlled": False,
                    "note": "no other-architecture decoders to calibrate on",
                }
            )
            continue
        calib_h0 = np.concatenate([pools[d]["h0"] for d in others])
        held_h0 = np.concatenate([pools[d]["h0"] for d in held_decoders])
        held_h1 = np.concatenate([pools[d]["h1"] for d in held_decoders])
        thr = _threshold_from_h0(calib_h0, alpha)
        if thr is None or held_h0.size < MIN_POOL or held_h1.size < MIN_POOL:
            out.append(
                {
                    "held_out_architecture": held,
                    "n_calib_decoders": len(others),
                    "fpr": None,
                    "target_consistent": False,
                    "controlled": False,
                    "note": "undefined threshold or held-out pool too small",
                }
            )
            continue
        fpr, _tpr = _realize(held_h0, held_h1, thr, n_groups=n_groups, seed=seed)
        resolved = _resolved_for_pool(held_h0, alpha, n_groups=n_groups, fpr_tol=fpr_tol)
        target_consistent = bool(
            resolved and fpr is not None and fpr > 0.0 and abs(fpr - alpha) <= fpr_tol
        )
        out.append(
            {
                "held_out_architecture": held,
                "held_out_decoders": held_decoders,
                "calib_decoders": others,
                "n_calib_decoders": len(others),
                "fpr": fpr,
                "target_consistent": target_consistent,
                "controlled": target_consistent,
            }
        )
    return out


def _off_diagonal_note(off: list[float], alpha: float) -> str:
    """Direction-aware summary of the off-diagonal (cross-decoder) FPR values vs target ``alpha``.

    An entry ABOVE alpha over-fires; BELOW alpha under-fires. The sign-inverted rho channel's short
    tail makes UNDER-firing the dominant cross-decoder regime (whitebox_lr), so an unconditional
    "over-fire" label would be backwards there -- this reports the worst of each direction present.
    """
    over = [v for v in off if v > alpha]
    under = [v for v in off if v < alpha]
    parts: list[str] = []
    if over:
        parts.append(f"worst over-fire {max(over):.3f}")
    if under:
        parts.append(f"worst under-fire {min(under):.3f}")
    if not parts:  # every off-diagonal sits essentially at alpha
        parts.append(f"all ~= alpha={alpha:g}")
    return (
        f"off-diagonal (cross-decoder) FPR in [{min(off):.3f}, {max(off):.3f}] vs "
        f"alpha={alpha:g}; " + ", ".join(parts) + "."
    )


def leave_one_decoder_out(
    pools: dict[str, dict[str, np.ndarray]],
    alpha: float,
    *,
    fpr_tol: float = DEFAULT_FPR_TOL,
    n_groups: int = 20000,
    seed: int = 0,
) -> list[dict]:
    """Calibrate the threshold on the POOLED other decoders' eval-H0, realise on the held-out one.

    The deployable "calibrate on the decoders you have, then face a new decoder" check. Needs >=2
    decoders to hold one out; with a single decoder the list is empty. A held-out decoder with no
    defined threshold (degenerate pooled-other H0) or too few held-out images yields null fpr/tpr.
    ``target_consistent`` is the composite criterion (held-out pool resolves alpha AND FPR is
    positive and within tolerance); it is not one-sided control. Legacy ``controlled`` is an
    identical compatibility alias.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    decoders = sorted(pools)
    out: list[dict] = []
    if len(decoders) < 2:
        return out
    for idx, held in enumerate(decoders):
        others_h0 = [pools[d]["h0"] for d in decoders if d != held]
        pooled_h0 = np.concatenate([np.asarray(a, dtype=np.float64).reshape(-1) for a in others_h0])
        thr = _threshold_from_h0(pooled_h0, alpha)
        if thr is None:
            out.append(
                {
                    "held_out": held,
                    "fpr": None,
                    "tpr": None,
                    "target_consistent": False,
                    "controlled": False,
                }
            )
            continue
        fpr, tpr = _realize(
            pools[held]["h0"], pools[held]["h1"], thr, n_groups=n_groups, seed=seed + 1000 + idx
        )
        resolved = _resolved_for_pool(pools[held]["h0"], alpha, n_groups=n_groups, fpr_tol=fpr_tol)
        target_consistent = bool(
            resolved and fpr is not None and fpr > 0.0 and abs(fpr - alpha) <= fpr_tol
        )
        out.append(
            {
                "held_out": held,
                "fpr": fpr,
                "tpr": tpr,
                "target_consistent": target_consistent,
                "controlled": target_consistent,
            }
        )
    return out


def _diagonal_offdiagonal(transfer: dict) -> tuple[list[float], list[float]]:
    """Collect the finite diagonal and off-diagonal FPR entries for the summary."""
    decoders = transfer["decoders"]
    fpr = transfer["transfer_fpr"]
    diag: list[float] = []
    off: list[float] = []
    for i in range(len(decoders)):
        for j in range(len(decoders)):
            v = fpr[i][j]
            if v is None:
                continue
            (diag if i == j else off).append(float(v))
    return diag, off


def _figure(transfer: dict, alpha: float) -> None:
    """Heatmap of the train x test FPR-transfer matrix, annotated, centered at alpha. JSON-first:
    guarded on matplotlib availability so the data product is produced regardless."""
    plt = viz.pyplot()
    if plt is None:
        return
    decoders = transfer["decoders"]
    n = len(decoders)
    if n == 0:
        return
    # NaN-free float grid for imshow; missing entries are masked (drawn light gray) and left
    # un-annotated so a degenerate transfer cell is visually distinct from a measured 0.0.
    grid = np.full((n, n), np.nan, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            v = transfer["transfer_fpr"][i][j]
            if v is not None:
                grid[i, j] = float(v)
    masked = np.ma.masked_invalid(grid)
    finite = masked.compressed()
    # Diverging colormap centered at alpha: |fpr-alpha| spread sets the symmetric range so under-
    # and over-firing read as opposite hues; fall back to a small span if everything equals alpha.
    spread = float(np.max(np.abs(finite - alpha))) if finite.size else alpha
    spread = spread if spread > 0.0 else max(alpha, 1e-3)
    fig, ax = plt.subplots(figsize=(1.6 + 1.1 * n, 1.4 + 1.0 * n))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("0.85")
    im = ax.imshow(masked, cmap=cmap, vmin=alpha - spread, vmax=alpha + spread, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(decoders)
    ax.set_yticklabels(decoders)
    ax.set_xlabel("test decoder")
    ax.set_ylabel("train decoder (threshold calibrated on)")
    for i in range(n):
        for j in range(n):
            v = transfer["transfer_fpr"][i][j]
            txt = "n/a" if v is None else f"{v:.3f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color="black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"realised FPR (target alpha={alpha:g})")
    ax.set_title(
        rf"cross-decoder FPR transfer ($N_{{\mathrm{{eff}}}}=1$, target $\alpha$={alpha:g})"
    )
    viz.save_figure(fig, "f6_cross_decoder_fpr")


def assess(
    cells: dict,
    *,
    alpha: float,
    alive_t: float,
    fpr_tol: float,
    n_groups: int,
    seed: int,
) -> dict:
    """Build the per-decoder AUC + FPR-transfer matrix + leave-one-decoder-out payload."""
    records = collect_alive_cell_records(cells, alive_t=alive_t)
    pools = per_decoder_pools(records)
    decoders = sorted(pools)
    notes: list[str] = []
    degenerate = [
        d
        for d in decoders
        if int(pools[d]["h0"].size) < MIN_POOL or int(pools[d]["h1"].size) < MIN_POOL
    ]
    if degenerate:
        notes.append(
            f"degenerate decoder pool(s) {degenerate} (<{MIN_POOL} H0 or H1 eval images): null "
            f"per-decoder AUC; null as a TEST column for ANY of them (a realised FPR/TPR needs "
            f"both >= {MIN_POOL} H0 and >= {MIN_POOL} H1); additionally null as a TRAIN row for a "
            f"<{MIN_POOL}-H0 one (no threshold without >= {MIN_POOL} H0)."
        )
    per_decoder = per_decoder_auc(pools)
    transfer = fpr_transfer_matrix(pools, alpha, fpr_tol=fpr_tol, n_groups=n_groups, seed=seed)
    loo = leave_one_decoder_out(pools, alpha, fpr_tol=fpr_tol, n_groups=n_groups, seed=seed)
    arch_transfer = architecture_transfer_summary(transfer, alpha)
    loao = leave_one_architecture_out(pools, alpha, fpr_tol=fpr_tol, n_groups=n_groups, seed=seed)
    diag, off = _diagonal_offdiagonal(transfer)
    if diag:
        notes.append(
            f"diagonal (within-decoder) FPR in [{min(diag):.3f}, {max(diag):.3f}] vs target "
            f"alpha={alpha:g}."
        )
    if off:
        notes.append(_off_diagonal_note(off, alpha))
    n_target_consistent = sum(int(c) for row in transfer["transfer_target_consistent"] for c in row)
    notes.append(
        f"{n_target_consistent} of {len(decoders) ** 2} transfer entries meet the composite target-"
        f"agreement criterion (fpr_resolved AND |fpr-alpha|<={fpr_tol:g}); this is not one-sided "
        "control. A 0.000 diagonal on the inverted (sign-) channel fails the criterion because it "
        "is short-tail degeneracy, not a calibrated operating point."
    )
    if len(decoders) < 2:
        notes.append(
            f"only {len(decoders)} decoder with alive cells; cross-decoder transfer is vacuous."
        )
    return {
        "alpha": alpha,
        "alive_t": alive_t,
        "fpr_tol": float(fpr_tol),
        "n_groups": int(n_groups),
        "decoders": decoders,
        "n_alive_cells": len(records),
        "alive_cells": [r.label() for r in records],
        "per_decoder": per_decoder,
        "transfer_fpr": transfer["transfer_fpr"],
        "transfer_tpr": transfer["transfer_tpr"],
        "transfer_target_consistent": transfer["transfer_target_consistent"],
        "transfer_controlled": transfer["transfer_controlled"],
        "transfer_resolved": transfer["transfer_resolved"],
        "leave_one_out": loo,
        "architecture_transfer": arch_transfer,
        "leave_one_architecture_out": loao,
        "notes": " ".join(notes) if notes else "no cross-decoder notes.",
    }


def run(
    features_root: Path,
    out_dir: Path,
    *,
    alpha: float,
    alive_t: float,
    fpr_tol: float,
    n_groups: int,
    seed: int,
) -> int:
    rows: list[dict] = []
    table_paths = sorted(features_root.glob("*/p0_table.json"))
    for table_path in table_paths:
        rows.extend(_load_p0_rows(table_path))
    if not rows:
        if not table_paths:
            print(f"[skip] no <decoder>/p0_table.json files under {features_root}")
        else:
            print(
                f"[skip] {len(table_paths)} p0_table.json file(s) under {features_root} "
                "contain no rows"
            )
        return 0
    cells = _group_rows_by_cell(rows)
    res = assess(cells, alpha=alpha, alive_t=alive_t, fpr_tol=fpr_tol, n_groups=n_groups, seed=seed)
    if not res["decoders"]:
        print(f"[skip] no decoder passed the alive gate (alive_t={alive_t}); nothing to transfer.")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cross_decoder.json"
    out_path.write_text(json.dumps(res, indent=2, allow_nan=False), encoding="utf-8")

    _figure({"decoders": res["decoders"], "transfer_fpr": res["transfer_fpr"]}, alpha)

    print(
        f"\nCross-decoder transfer ({res['n_alive_cells']} alive cells, "
        f"{len(res['decoders'])} decoders, alpha={alpha:g}, alive_t={alive_t}):"
    )
    print("  per-decoder single-image AUC:")
    for pd in res["per_decoder"]:
        a = "n/a" if pd["auc"] is None else f"{pd['auc']:.3f}"
        print(f"    {pd['decoder']:6s}  H0={pd['n_h0']:3d} H1={pd['n_h1']:3d}  AUC={a}")
    print(f"  FPR-transfer matrix (rows=train, cols=test, target alpha={alpha:g}):")
    header = "          " + "  ".join(f"{d:>7s}" for d in res["decoders"])
    print(header)
    for i, train in enumerate(res["decoders"]):
        cells_str = "  ".join(
            "    n/a" if v is None else f"{v:7.3f}" for v in res["transfer_fpr"][i]
        )
        print(f"    {train:6s}  {cells_str}")
    print("  leave-one-decoder-out (threshold from pooled OTHER decoders):")
    for lo in res["leave_one_out"]:
        f = "n/a" if lo["fpr"] is None else f"{lo['fpr']:.3f}"
        t = "n/a" if lo["tpr"] is None else f"{lo['tpr']:.3f}"
        verdict = "meets-criterion" if lo["target_consistent"] else "fails-criterion"
        print(f"    held_out={lo['held_out']:6s}  FPR={f}  TPR={t}  ({verdict})")
    print(f"  notes: {res['notes']}")
    print(f"-> {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cross_decoder")
    ap.add_argument(
        "--features-root", default=None, help="dir of <decoder>/p0_table.json (default: outputs/p0)"
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="target FPR for the transfer matrix (default 0.01)",
    )
    ap.add_argument(
        "--alive-t", type=float, default=ALIVE_T, help="held-out alive gate (pooled default 1.5)"
    )
    ap.add_argument(
        "--fpr-tol",
        type=float,
        default=DEFAULT_FPR_TOL,
        help="|fpr_emp - alpha| tolerance for the target-agreement criterion (default 0.02)",
    )
    ap.add_argument("--n-groups", type=int, default=20000, help="bootstrap groups for FPR/TPR")
    args = ap.parse_args(argv)
    features_root = Path(args.features_root) if args.features_root else (outputs_dir() / "p0")
    out_dir = outputs_dir() / "p0"
    return run(
        features_root,
        out_dir,
        alpha=args.alpha,
        alive_t=args.alive_t,
        fpr_tol=args.fpr_tol,
        n_groups=args.n_groups,
        seed=0,
    )


if __name__ == "__main__":
    raise SystemExit(main())
