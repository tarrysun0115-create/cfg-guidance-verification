"""Sign-aligned rho vs CLIP text alignment on the favorable subset's evaluation images.

Produces: no printed paper number directly (shared helper). Its functions (``collect_alive_eval``,
``compare_rho_clip``, ``_cell_value_struct``, ``_eval_rows``) are reused by
``neutral_pool_allcell.py`` (the CLIP row of Table VIII, Section IV-E1), ``pair_distinctness.py``,
``forensic_baselines.py``, ``rho_mechanism.py`` and ``deep_probe.py``, so all of them score the
identical favorable-subset images. The standalone ``clip_vs_rho.json`` it writes is a diagnostic and
is not printed in the paper.

Inputs: ``outputs/p0/<decoder>/p0_table.json``, the main-grid PNGs it references (not
shipped; written by the generation run), and the local CLIP checkpoint
(``checkpoints/<evidence.clip_local_dir>`` from ``configs/p0.yaml``). No re-inversion; a GPU only
speeds up CLIP.

Outputs: ``outputs/p0/clip_vs_rho.json``.

Usage: python scripts/analysis/clip_vs_rho.py [--features-root outputs/p0] [--clip-dir DIR]

Favorable-subset selection (mirrors ``cfgid.sweep.pooled_neff_scaling``, Section IV-E): per cell
(decoder, sigma_blur) pick the H1 guidance w with the largest |FIT AUC - 0.5|, take the sign from
FIT, standardize TEST rho by the FIT H0 mean/std, split TEST positionally into a gate half and an
evaluation half, and admit the cell when the gate half's Welch |t| >= ``--alive-t`` (1.5) with a
shift agreeing with the FIT sign. On the pooled evaluation halves it reports:
  - aligned-rho AUC and CLIP AUC (head to head);
  - the CLIP-RESIDUALIZED aligned-rho AUC: regress rho_z on the CLIP score (label-free linear
    decorrelation), take the residual, and score it, with a label-permutation p-value
    (``--n-perm`` draws, seed 0). A residual that still separates H1/H0 means rho carries
    guidance signal beyond the CLIP semantic cue.
"""

from __future__ import annotations

import argparse
import json
from operator import index
from pathlib import Path

import numpy as np

from cfgid.baselines.clip_align import ClipUnavailableError, clip_alignment_scores, clip_available
from cfgid.detector.empirical import auc
from cfgid.paths import checkpoints_dir, outputs_dir
from cfgid.sweep.neff_scaling import (
    _Cell,
    _load_p0_rows,
    _row_finite_float,
    _row_finite_int,
    filter_core_none_qf_rows,
)
from cfgid.sweep.pooled_neff_scaling import (
    _best_abs_signal_w,
    _split_gate_eval,
    _standardize,
    _two_sample_shift_t,
)


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    try:
        out = index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from exc
    if out < 1:
        raise ValueError(f"{name} must be >= 1, got {out}")
    return out


def _cell_value_struct(cell_rows: list[dict]) -> _Cell:
    """Build the value-only _Cell (rho_summary by w/split) the pooled selection expects."""
    c = _Cell()
    for r in cell_rows:
        rho = _row_finite_float(r, "rho_summary")
        w = _row_finite_int(r, "w")
        bucket = (
            "fit" if r.get("split") == "train" else ("test" if r.get("split") == "test" else None)
        )
        if bucket is None:
            continue
        label = str(r.get("label"))
        if label == "h0":
            c.h0[bucket].append(rho)
        elif label == "h1":
            c.h1.setdefault(w, {"fit": [], "test": []})[bucket].append(rho)
        else:
            raise ValueError("row label must be one of ('h0', 'h1')")
    return c


def _row_decoder(row: dict) -> str:
    if "decoder" not in row:
        raise ValueError("row field 'decoder' is required")
    return str(row["decoder"])


def _eval_rows(cell_rows: list[dict], w: int, label: str) -> list[dict]:
    """TEST-eval-half rows for one (cell, w, label), matching pooled's positional split.

    For H0 we do NOT filter on w: the pooled collector pools EVERY test-H0 row regardless of w
    (``_group_rows_by_cell`` ignores w for H0; run_p0 emits H0 at w=1). Mirroring that exactly makes
    the eval set structurally identical to the pooled module's instead of only identical under the
    current schema — so a stray test-H0 row with w!=1 can never silently desync the two selections.
    """
    test = [
        r
        for r in cell_rows
        if str(r["label"]) == label
        and r.get("split") == "test"
        and (label == "h0" or _row_finite_int(r, "w") == w)
    ]
    mid = len(test) // 2
    return test[mid:]  # second half == eval (gate is the first half)


def collect_alive_eval(rows_by_decoder: dict[str, tuple[Path, list[dict]]], *, alive_t: float):
    """Mirror pooled selection; return (image_paths, prompts, aligned_rho_z, labels, alive)."""
    paths: list[Path] = []
    prompts: list[str] = []
    rho_z: list[float] = []
    labels: list[int] = []
    alive: list[str] = []
    cell_groups: list[tuple[str, str, Path, float, list[dict]]] = []
    for decoder, (decoder_dir, raw_rows) in rows_by_decoder.items():
        # Route through the shared filter so this alive set matches the pooled collector
        # EXACTLY: drop non-core (FLUX) rows + fail loud on a compressed-QF axis.
        rows = filter_core_none_qf_rows(raw_rows)
        by_cell: dict[float, list[dict]] = {}
        decoder_name = str(decoder)
        for r in rows:
            row_decoder = _row_decoder(r)
            if row_decoder != decoder_name:
                raise ValueError(
                    f"row decoder {row_decoder!r} does not match rows_by_decoder key "
                    f"{decoder_name!r}"
                )
            by_cell.setdefault(_row_finite_float(r, "sigma_blur"), []).append(r)
        for sb, cell_rows in by_cell.items():
            cell_groups.append(
                (f"{decoder_name}/sb={sb:g}", decoder_name, decoder_dir, sb, cell_rows)
            )
    for key, _decoder, decoder_dir, _sb, cell_rows in sorted(cell_groups, key=lambda item: item[0]):
        cell = _cell_value_struct(cell_rows)
        # min_n=8 + the >=4 gate/eval guard match cfgid.sweep.pooled_neff_scaling exactly, so
        # the alive set here is identical to the pooled module's (locked by a cross-check test).
        w = _best_abs_signal_w(cell, min_n=8)
        if w is None:
            continue
        h0_fit = np.asarray(cell.h0["fit"], dtype=np.float64)
        h1_fit = np.asarray(cell.h1[w]["fit"], dtype=np.float64)
        m0, s0 = float(np.mean(h0_fit)), float(np.std(h0_fit, ddof=1))
        if not (np.isfinite(s0) and s0 > 0):
            continue
        # Orientation from FIT only: sign of the mean standardized H1 shift.
        s = float(np.sign(np.mean(_standardize(h1_fit, m0, s0))))
        if s == 0.0:
            continue
        h0_test = np.asarray(cell.h0["test"], dtype=np.float64)
        h1_test = np.asarray(cell.h1[w]["test"], dtype=np.float64)
        zt0_gate, zt0_eval = _split_gate_eval(_standardize(h0_test, m0, s0))
        zt1_gate, zt1_eval = _split_gate_eval(_standardize(h1_test, m0, s0))
        if min(zt0_gate.size, zt0_eval.size, zt1_gate.size, zt1_eval.size) < 4:
            continue
        # Gate on the first TEST half (Welch t, sign must agree with FIT); evaluate on the second.
        shift, t = _two_sample_shift_t(zt0_gate, zt1_gate)
        if not np.isfinite(t) or abs(t) < alive_t or float(np.sign(shift)) != s:
            continue
        alive.append(f"{key}/w={w}/sign={'+' if s > 0 else '-'}")
        for label, lab in (("h0", 0), ("h1", 1)):
            for r in _eval_rows(cell_rows, w, label):
                paths.append(decoder_dir / str(r["image"]))
                prompts.append(str(r["prompt_text"]))
                rho_z.append(s * (_row_finite_float(r, "rho_summary") - m0) / s0)
                labels.append(lab)
    return paths, prompts, np.array(rho_z), np.array(labels, dtype=int), tuple(alive)


def _residualize(x: np.ndarray, on: np.ndarray) -> np.ndarray:
    """Linear (label-free) decorrelation: residual of x after regressing on `on`."""
    a, b = np.polyfit(on, x, 1)[::-1] if on.std() > 0 else (float(np.mean(x)), 0.0)
    return x - (a + b * on)


def compare_rho_clip(
    rho_z: np.ndarray, clip: np.ndarray, labels: np.ndarray, *, n_perm: int, seed: int
) -> dict:
    """Head-to-head AUC + the CLIP-residualized rho AUC + its label-permutation p-value."""
    n_perm = _positive_int("n_perm", n_perm)
    rho_z = np.asarray(rho_z, dtype=np.float64).reshape(-1)
    clip = np.asarray(clip, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    if rho_z.shape != clip.shape or rho_z.shape != labels.shape:
        raise ValueError(
            f"rho_z, clip, and labels must have the same shape, got "
            f"{rho_z.shape}, {clip.shape}, {labels.shape}"
        )
    rho_auc = float(auc(rho_z, labels))
    clip_auc = float(auc(clip, labels))
    resid = _residualize(rho_z, clip)  # rho with the CLIP component removed (label-free)
    resid_auc = float(auc(resid, labels))
    rng = np.random.default_rng(seed)
    perm = np.array([auc(resid, rng.permutation(labels)) for _ in range(n_perm)])
    # One-sided permutation p with the +1 correction (floor 1 / (n_perm + 1)).
    resid_perm_p = float((1.0 + np.sum(perm >= resid_auc)) / (n_perm + 1.0))
    return {
        "aligned_rho_auc": rho_auc,
        "clip_auc": clip_auc,
        "clip_residualized_rho_auc": resid_auc,
        "clip_residualized_rho_perm_p": resid_perm_p,
        "rho_beats_clip": bool(rho_auc > clip_auc),
        "rho_adds_beyond_clip": bool(resid_perm_p < 0.05 and resid_auc > 0.5),
    }


def run(
    features_root: Path, clip_dir: Path, out_dir: Path, *, alive_t: float, n_perm: int, seed: int
) -> int:
    if not clip_available(clip_dir):
        print(
            f"[skip] CLIP checkpoint missing at {clip_dir}; cannot run the rho-vs-CLIP comparison."
        )
        return 0
    rows_by_decoder: dict[str, tuple[Path, list[dict]]] = {}
    for table_path in sorted(features_root.glob("*/p0_table.json")):
        rows_by_decoder[table_path.parent.name] = (table_path.parent, _load_p0_rows(table_path))
    if not rows_by_decoder:
        print(f"[skip] no <decoder>/p0_table.json under {features_root}")
        return 0
    paths, prompts, rho_z, labels, alive = collect_alive_eval(rows_by_decoder, alive_t=alive_t)
    if labels.size < 8 or labels.sum() == 0 or labels.sum() == labels.size:
        print(f"[skip] too few / single-class pooled eval images ({labels.size}); no comparison.")
        return 0
    try:
        from cfgid.device import get_device

        clip = clip_alignment_scores(paths, prompts, clip_dir=clip_dir, device=get_device()).astype(
            np.float64
        )
    except ClipUnavailableError as exc:
        print(f"[skip] CLIP checkpoint unusable at {clip_dir}: {exc}")
        return 0

    cmp = compare_rho_clip(rho_z, clip, labels, n_perm=n_perm, seed=seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_alive_cells": len(alive),
        "alive_cells": list(alive),
        "n_eval_images": int(labels.size),
        **cmp,
    }
    (out_dir / "clip_vs_rho.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        f"\nCLIP vs sign-aligned rho ({len(alive)} alive cells, {labels.size} held-out eval imgs):"
    )
    print(
        f"  aligned-rho AUC = {cmp['aligned_rho_auc']:.3f}    CLIP AUC = {cmp['clip_auc']:.3f}    "
        f"({'rho BEATS CLIP' if cmp['rho_beats_clip'] else 'CLIP >= rho'})"
    )
    print(
        f"  CLIP-residualized rho AUC = {cmp['clip_residualized_rho_auc']:.3f}  "
        f"permutation p = {cmp['clip_residualized_rho_perm_p']:.4f}"
    )
    if payload["rho_adds_beyond_clip"]:
        print(
            "  => rho carries CFG signal BEYOND CLIP: the residual separation is not only the"
        )
        print("     CLIP semantic confound (read together with the head-to-head AUCs above).")
    else:
        print(
            "  => rho does NOT add significant signal beyond CLIP -> aligned-rho detection may be"
        )
        print("     the CLIP confound on this pool.")
    print(f"-> {out_dir / 'clip_vs_rho.json'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="clip_vs_rho")
    ap.add_argument(
        "--features-root", default=None, help="dir of <decoder>/p0_table.json (default: outputs/p0)"
    )
    ap.add_argument(
        "--clip-dir",
        default=None,
        help="CLIP checkpoint dir (default: checkpoints/<evidence.clip_local_dir>)",
    )
    ap.add_argument("--alive-t", type=float, default=1.5)
    ap.add_argument("--n-perm", type=int, default=2000)
    args = ap.parse_args(argv)
    features_root = Path(args.features_root) if args.features_root else (outputs_dir() / "p0")
    if args.clip_dir:
        clip_dir = Path(args.clip_dir)
    else:
        from cfgid.p0config import load_p0_config

        clip_dir = checkpoints_dir() / load_p0_config().evidence.clip_local_dir
    return run(
        features_root,
        clip_dir,
        outputs_dir() / "p0",
        alive_t=args.alive_t,
        n_perm=args.n_perm,
        seed=0,
    )


if __name__ == "__main__":
    raise SystemExit(main())
