"""Orientation x magnitude candidate readouts over the stored residual field (server helper).

Produces: no printed paper number by itself; its own JSON is a diagnostic.
Its per-image feature builder ``_per_image_features`` (with the rho-reproduction gate), the
synthetic smoke tree ``_smoke_tree``, ``_sign_pattern``, ``_transfer_summary`` and the sibling
preflight are reused by pair_readout_battery.py (Table I pair rows, Table III) and
prompt_mismatch_pair.py (Table IX).

Question: can a candidate keep ``diff_energy``'s raw single-image detection power while carrying
an orientation, and how does each candidate's calibrated FPR threshold transfer across decoders?
Pure reanalysis of the SERVER feature store (``residual_v1.dat`` + ``compact_rho.npy``) and
``p0_table.json``; NO re-inversion.

Candidates (per-image scalars, all under the identical E_tex-mask + winsorize + mean template of
Eq. (4) / Eq. (8)):

* ``rho``             -- the paper channel rho(x) (baseline control; every image must reproduce
                         the stored ``rho_summary`` within RHO_REPRODUCTION_TOL, or the run
                         refuses to emit numbers).
* ``diff_energy``     -- unsigned per-patch ||r_cond - r_uncond|| magnitude, i.e. the pair's m(x)
                         coordinate (raw-power reference).
* ``signed_diff``     -- per-patch sign(rho_p) x ||dr_p|| before the template (a per-patch
                         product of sign and magnitude).
* ``img_signed_diff`` -- image-level sign(rho(x)) x diff_energy(x): the sign is taken from the
                         image-level rho instead of each patch.
* ``fusion_zsum``     -- fixed combination with no labels beyond FIT; per cell, z-score rho(x)
                         and diff_energy(x) on their own FIT-H0 and sum after FIT-sign alignment
                         (defined under the fixed w=20 protocol; reported for that variant only).

Per candidate it reports (a) the selection-free anchor fields via the unmodified
``fixed_h1_allcell`` machinery (AUC1 / macro / sqrt-K exponent / K-grid / TPR@1%,5%), (b) the
cross-decoder FPR-transfer matrix via the unmodified ``cross_decoder`` machinery, and (c) the
per-cell FIT-sign pattern (does the candidate's response direction vary by decoder at all).

Mechanism: per-image candidate values are written as VIRTUAL ``p0_table.json`` clones (only
``rho_summary`` replaced) under a work dir, and the existing, already-validated runners are
invoked on those clones unchanged. Paper figures are protected by no-op'ing ``viz.save_figure``
for the duration of the reused runs.

Inputs: outputs/p0/<decoder>/{p0_table.json, residual_v1.dat, compact_*.npy, metadata.json} --
the server-only feature store (``signed_diff`` needs the per-patch fields).
Outputs: outputs/p0/candidate_readouts/{candidate_readouts.json, fixed_h1_<candidate>.json,
<candidate>/cross_decoder.json}.
Usage (server):
    uv run --no-sync python scripts/analysis/candidate_readouts.py
Self-test (local, synthetic store; validates the full plumbing end to end):
    uv run --no-sync python scripts/analysis/candidate_readouts.py --smoke
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from cfgid import viz
from cfgid.featurestore import read_compact, read_residual_v1
from cfgid.numerics import finite_float
from cfgid.paths import outputs_dir
from cfgid.sweep.evidence import _require_row_store_alignment, _residual_energy_features
from cfgid.sweep.neff_scaling import _group_rows_by_cell, _load_p0_rows

RHO_REPRODUCTION_TOL = 1e-3  # same gate as feature_comparison: candidate 'rho' must match stored
FUSION_H1_W = 20  # the FIT-sign protocol under which fusion_zsum is defined
CANDIDATES = ("rho", "diff_energy", "signed_diff", "img_signed_diff", "fusion_zsum")


def _load_sibling(name: str):
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"{name}_reuse", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_FH = _load_sibling("fixed_h1_allcell")
_XD = _load_sibling("cross_decoder")
_FC = _load_sibling("feature_comparison")  # reuse _image_summary + store validation helpers


def _preflight_sibling_versions() -> None:
    """Fail fast (BEFORE the expensive store pass) if a sibling script is a stale version.

    The fixed_h1 machinery must support the generalized H0 set (``h0_ws``) and per-variant
    (h0_ws, h1_ws) pairs; an older copy silently yields '[w20] no cells' and then crashes in
    ``collect_fixed_h1_records`` (e.g. when a server's scripts/ directory lags the repository).
    """
    import inspect

    problems: list[str] = []
    if "h0_ws" not in inspect.signature(_FH.collect_fixed_h1_records).parameters:
        problems.append("fixed_h1_allcell.collect_fixed_h1_records lacks the 'h0_ws' parameter")
    if "seed" not in inspect.signature(_FH.run).parameters:
        problems.append("fixed_h1_allcell.run lacks the 'seed' parameter")
    sample = next(iter(_FH.DEFAULT_VARIANTS.values()))
    if not (
        isinstance(sample, tuple) and len(sample) == 2 and isinstance(sample[0], tuple)
    ):
        problems.append("fixed_h1_allcell.DEFAULT_VARIANTS values are not (h0_ws, h1_ws) pairs")
    if problems:
        raise SystemExit(
            "[abort] stale sibling script(s) on this machine:\n  - "
            + "\n  - ".join(problems)
            + "\nSync the current scripts/analysis/fixed_h1_allcell.py from the repo and re-run."
        )


_preflight_sibling_versions()


def _per_image_features(dec_dir: Path, rows: list[dict]) -> dict[str, dict[str, float]]:
    """Per-image scalars for the store-template candidates of one decoder dir.

    Returns name -> {image_id -> value} for rho / diff_energy / signed_diff / img_signed_diff.
    Fails loud if the store is absent, misaligned, or the recomputed rho does not reproduce the
    stored ``rho_summary`` within RHO_REPRODUCTION_TOL.
    """
    if not (dec_dir / "residual_v1.dat").exists() or not (dec_dir / "compact_rho.npy").exists():
        raise SystemExit(
            f"[abort] {dec_dir.name}: feature store missing (residual_v1.dat + compact_rho.npy). "
            "This script is a SERVER reanalysis; run it where the store lives."
        )
    batch, meta = read_compact(dec_dir)
    residual, _ = read_residual_v1(dec_dir)
    _require_row_store_alignment(rows, meta)
    diff_pp = _residual_energy_features(residual)  # (N, P) per-patch ||dr_p||
    rho_pp = batch.rho  # (N, P)
    # Store records pair with table rows by position (image_id order checked above).
    ids = list(meta.image_ids) or [str(r["image_id"]) for r in rows]

    out: dict[str, dict[str, float]] = {name: {} for name in CANDIDATES if name != "fusion_zsum"}
    stored = {str(r["image_id"]): finite_float("rho_summary", r["rho_summary"]) for r in rows}
    worst = 0.0
    for i, iid in enumerate(ids):
        rp = np.asarray(rho_pp[i], dtype=np.float64)
        dp = np.asarray(diff_pp[i], dtype=np.float64)
        et = np.asarray(batch.e_tex[i], dtype=np.float64)
        # Same E_tex mask (q = 0.2) + winsorization ([0.01, 0.99]) + mean for every candidate.
        rho_sum = finite_float("rho", _FC._image_summary(rp, et))
        diff_sum = finite_float("diff_energy", _FC._image_summary(dp, et))
        sgn_sum = finite_float("signed_diff", _FC._image_summary(np.sign(rp) * dp, et))
        img_sign = float(np.sign(rho_sum)) or 1.0  # sign(0) -> +1, never a dropped image
        out["rho"][iid] = rho_sum
        out["diff_energy"][iid] = diff_sum
        out["signed_diff"][iid] = sgn_sum
        out["img_signed_diff"][iid] = img_sign * diff_sum
        if iid in stored:
            worst = max(worst, abs(rho_sum - stored[iid]))
    if worst > RHO_REPRODUCTION_TOL:
        raise SystemExit(
            f"[abort] {dec_dir.name}: recomputed rho deviates from stored rho_summary by "
            f"{worst:.4g} (> {RHO_REPRODUCTION_TOL:g}); refusing to emit numbers."
        )
    return out


def _fusion_values(rows: list[dict], feats: dict[str, dict[str, float]]) -> dict[str, float]:
    """Per-cell FIT-H0 z-scores of rho and diff_energy, FIT-sign aligned, summed.

    Defined under the fixed w=1-vs-w=FUSION_H1_W protocol: per (decoder, sigma_blur) cell, each
    component is standardized on its train-split H0 mean/std and oriented by the sign of its
    train-split H1 (w=FUSION_H1_W) mean. Cells lacking those FIT samples contribute no fused rows.
    """
    by_cell: dict[str, dict[str, list[tuple[str, float, float]]]] = {}
    for r in rows:
        iid = str(r["image_id"])
        if iid not in feats["rho"] or r.get("split") not in ("train",):
            continue
        key = f"{r['decoder']}/sb={float(r['sigma_blur']):g}"
        lab = str(r.get("label"))
        w = int(r["w"])
        slot = "h0" if lab == "h0" else (f"h1_{w}" if lab == "h1" else "")
        if slot in ("h0", f"h1_{FUSION_H1_W}"):
            by_cell.setdefault(key, {}).setdefault(slot, []).append(
                (iid, feats["rho"][iid], feats["diff_energy"][iid])
            )
    stats: dict[str, tuple[float, float, float, float, float, float]] = {}
    for key, slots in by_cell.items():
        h0 = slots.get("h0", [])
        h1 = slots.get(f"h1_{FUSION_H1_W}", [])
        if len(h0) < 8 or len(h1) < 8:
            continue
        # FIT-H0 mean and sample std (ddof=1) per coordinate; orientation = sign of the mean
        # standardized FIT-H1 value (ties -> +1).
        r0 = np.array([v[1] for v in h0])
        d0 = np.array([v[2] for v in h0])
        mr, sr = float(np.mean(r0)), float(np.std(r0, ddof=1))
        md, sd = float(np.mean(d0)), float(np.std(d0, ddof=1))
        if not (np.isfinite(sr) and sr > 0 and np.isfinite(sd) and sd > 0):
            continue
        sgn_r = float(np.sign(np.mean([(v[1] - mr) / sr for v in h1]))) or 1.0
        sgn_d = float(np.sign(np.mean([(v[2] - md) / sd for v in h1]))) or 1.0
        stats[key] = (mr, sr, sgn_r, md, sd, sgn_d)
    fused: dict[str, float] = {}
    for r in rows:
        iid = str(r["image_id"])
        key = f"{r['decoder']}/sb={float(r['sigma_blur']):g}"
        if iid not in feats["rho"] or key not in stats:
            continue
        mr, sr, sgn_r, md, sd, sgn_d = stats[key]
        fused[iid] = float(
            sgn_r * (feats["rho"][iid] - mr) / sr + sgn_d * (feats["diff_energy"][iid] - md) / sd
        )
    return fused


def _write_virtual_tables(
    features_root: Path, work_root: Path
) -> tuple[dict[str, Path], dict[str, int]]:
    """Stage 1: compute all candidate per-image values and write virtual p0_table clones."""
    roots = {name: work_root / name for name in CANDIDATES}
    dropped_fusion = 0
    for dec_dir in sorted(d for d in features_root.glob("*/") if (d / "p0_table.json").exists()):
        rows = _load_p0_rows(dec_dir / "p0_table.json")
        feats = _per_image_features(dec_dir, rows)
        feats["fusion_zsum"] = _fusion_values(rows, feats)
        for name in CANDIDATES:
            fmap = feats[name]
            clone = []
            for r in rows:
                iid = str(r["image_id"])
                if iid not in fmap:
                    continue
                rr = dict(r)
                rr["rho_summary"] = fmap[iid]
                clone.append(rr)
            if name == "fusion_zsum":
                dropped_fusion += len(rows) - len(clone)
            out_dir = roots[name] / dec_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "p0_table.json").write_text(
                json.dumps(clone, allow_nan=False), encoding="utf-8"
            )
        print(f"[stage1] {dec_dir.name}: {len(rows)} rows -> {len(CANDIDATES)} virtual tables")
    return roots, {"fusion_rows_dropped": dropped_fusion}


def _sign_pattern(root: Path) -> dict:
    """Diagnostic: the candidate's per-cell FIT signs (does direction vary by decoder?)."""
    rows: list[dict] = []
    for tp in sorted(root.glob("*/p0_table.json")):
        rows.extend(_load_p0_rows(tp))
    cells = _group_rows_by_cell(rows)
    recs = _FH.collect_fixed_h1_records(cells, (FUSION_H1_W,), min_n=8, h0_ws=(1,))
    per_cell = {r.key: int(r.sign) for r in recs}
    n_pos = sum(1 for s in per_cell.values() if s > 0)
    return {
        "per_cell_fit_sign_w20": per_cell,
        "n_pos": n_pos,
        "n_neg": len(per_cell) - n_pos,
        "uniform_direction": n_pos == 0 or n_pos == len(per_cell),
    }


def _transfer_summary(res_path: Path) -> dict:
    """Condense a cross_decoder.json into the transfer summary numbers."""
    res = json.loads(res_path.read_text(encoding="utf-8"))
    alpha = float(res["alpha"])
    mat = res["transfer_fpr"]
    n_dec = len(res["decoders"])
    within = [mat[i][i] for i in range(n_dec) if mat[i][i] is not None]
    cross = [
        mat[i][j] for i in range(n_dec) for j in range(n_dec) if i != j and mat[i][j] is not None
    ]
    ctrl = res.get("transfer_controlled")
    n_controlled = (
        sum(1 for i in range(n_dec) for j in range(n_dec) if i != j and bool(ctrl[i][j]))
        if ctrl
        else None
    )
    return {
        "decoders": res["decoders"],
        "alpha": alpha,
        "n_cross_entries": len(cross),
        "n_cross_controlled": n_controlled,
        "mean_abs_fpr_err_within": (
            float(np.mean([abs(v - alpha) for v in within])) if within else None
        ),
        "mean_abs_fpr_err_cross": (
            float(np.mean([abs(v - alpha) for v in cross])) if cross else None
        ),
    }


def run(features_root: Path, out_dir: Path, work_root: Path, *, seed: int = 0) -> int:
    """Stage 1 virtual tables, then the reused anchor / transfer runners per candidate."""
    out_dir.mkdir(parents=True, exist_ok=True)
    roots, notes = _write_virtual_tables(features_root, work_root)

    # Protect the paper figures: the reused runners draw f5/f6 assets on their default paths.
    viz_save = viz.save_figure
    viz.save_figure = lambda *a, **k: None  # type: ignore[assignment]
    xd_figure = _XD._figure
    _XD._figure = lambda *a, **k: None
    matrix: dict[str, dict] = {}
    try:
        for name in CANDIDATES:
            variants = {"w20": ((1,), (FUSION_H1_W,))}
            if name != "fusion_zsum":  # fusion is defined under the w20 protocol only
                variants["low3_vs_high10"] = ((1, 2, 3), (10, 14, 20))
            anchors_path = out_dir / f"fixed_h1_{name}.json"
            rc = _FH.run(
                roots[name],
                anchors_path,
                variants=variants,
                n_groups=4000,
                n_perm=500,
                min_n=8,
                seed=seed,
            )
            if rc != 0:
                raise SystemExit(f"[abort] fixed_h1 runner failed for {name!r}")
            xd_dir = out_dir / name
            rc = _XD.run(
                roots[name],
                xd_dir,
                alpha=0.01,
                alive_t=1.5,
                fpr_tol=0.01,
                n_groups=20000,
                seed=seed,
            )
            anchors = json.loads(anchors_path.read_text(encoding="utf-8"))
            entry: dict = {
                "anchors_w20": anchors["variants"]["w20"]["all_cell"],
                "sign_pattern": _sign_pattern(roots[name]),
            }
            if "low3_vs_high10" in anchors["variants"]:
                entry["anchors_low3_vs_high10"] = anchors["variants"]["low3_vs_high10"]["all_cell"]
            xd_json = xd_dir / "cross_decoder.json"
            entry["transfer"] = _transfer_summary(xd_json) if xd_json.exists() else None
            matrix[name] = entry
    finally:
        viz.save_figure = viz_save  # type: ignore[assignment]
        _XD._figure = xd_figure

    out = {
        "analysis": "candidate_readouts",
        "seed": seed,
        "protocol": {
            "anchors": "fixed_h1_allcell machinery, selection-free, n_groups=4000, n_perm=500",
            "transfer": "cross_decoder machinery, alpha=0.01, alive_t=1.5, n_groups=20000",
            "fusion": f"per-cell FIT-H0 z-sum, FIT signs under w={FUSION_H1_W}",
        },
        "notes": notes,
        "candidates": matrix,
    }
    out_path = out_dir / "candidate_readouts.json"
    out_path.write_text(json.dumps(out, indent=2, allow_nan=False), encoding="utf-8")

    print("\n=== candidate readouts (selection-free w20 anchors) ===")
    hdr = (
        f"{'candidate':18s} {'AUC1':>6s} {'macro':>6s} {'exp':>6s} {'K16':>6s} "
        f"{'TPR@1%':>7s} {'TPR@5%':>7s} {'signs +/-':>10s} {'xfer ctrl':>9s}"
    )
    print(hdr)
    for name, e in matrix.items():
        a = e["anchors_w20"]
        sp = e["sign_pattern"]
        tr = e["transfer"]
        k16 = a["auc_emp"][a["k_grid"].index(16)] if 16 in a["k_grid"] else float("nan")
        xf = "n/a" if not tr or tr["n_cross_controlled"] is None else (
            f"{tr['n_cross_controlled']}/{tr['n_cross_entries']}"
        )
        print(
            f"{name:18s} {a['auc1_pooled_fitsign']:6.3f} {a['macro_cell_auc']:6.3f} "
            f"{(a['exponent'] if a['exponent'] is not None else float('nan')):6.3f} {k16:6.3f} "
            f"{a['tpr_at_1pct_fpr_k1']:7.3f} {a['tpr_at_5pct_fpr_k1']:7.3f} "
            f"{sp['n_pos']:>4d}/{sp['n_neg']:<4d} {xf:>9s}"
        )
    print(f"-> {out_path}")
    return 0


# --- smoke self-test: synthetic store, validates the full plumbing locally ----------------------


def _smoke_tree(root: Path, *, seed: int = 0) -> None:
    """Two synthetic decoders x sb {0,16} x w {1,2,3,10,14,20}, store + p0_table per decoder."""
    from cfgid.featurestore.schema import FeatureMetadata
    from cfgid.featurestore.store import CompactBatch, write_compact, write_residual_v1

    rng = np.random.default_rng(seed)
    n_p, dim = 16, 4
    ws = (1, 2, 3, 10, 14, 20)
    n_prompts = 60
    splits = ["train"] * 24 + ["null_adapt"] * 12 + ["calibration"] * 12 + ["test"] * 12
    for dec, dsign in (("deca", +1.0), ("decb", -1.0)):
        rows: list[dict] = []
        rho_all, etex_all, res_all, ids = [], [], [], []
        for sb in (0.0, 16.0):
            atten = 1.0 if sb == 0.0 else 0.4
            for w in ws:
                shift = dsign * 0.9 * atten * np.log1p(max(w - 1, 0)) / np.log1p(19)
                dmag = 1.0 + 1.4 * atten * (max(w - 1, 0) / 19.0)
                for p in range(n_prompts):
                    rho_p = rng.normal(shift, 0.6, size=n_p)
                    res_vec = rng.normal(0.0, 1.0, size=(n_p, dim))
                    res_vec *= (dmag + rng.normal(0, 0.25)) / max(
                        np.linalg.norm(res_vec, axis=1).mean(), 1e-9
                    )
                    e_tex = rng.uniform(0.5, 1.5, size=n_p)
                    iid = f"{dec}_sb{sb:g}_w{w}_p{p}_s0"  # needs the _p<idx>_ shape
                    rho_sum = _FC._image_summary(rho_p, e_tex)
                    rows.append(
                        {
                            "image_id": iid,
                            "decoder": dec,
                            "sigma_blur": sb,
                            "w": w,
                            "label": "h0" if w == 1 else "h1",
                            "split": splits[p % len(splits)],
                            "rho_summary": float(rho_sum),
                        }
                    )
                    ids.append(iid)
                    rho_all.append(rho_p.astype(np.float32))
                    etex_all.append(e_tex.astype(np.float32))
                    res_all.append(res_vec.astype(np.float16))
        dec_dir = root / dec
        dec_dir.mkdir(parents=True, exist_ok=True)
        n = len(ids)
        meta = FeatureMetadata(
            model_name=dec,
            branch_mode="true_cfg_core",
            n_images=n,
            n_patches=n_p,
            patch_grid=(4, 4),
            residual_dim=dim,
            image_ids=ids,
        )
        batch = CompactBatch(
            rho=np.stack(rho_all),
            e_tex=np.stack(etex_all),
            sigma=np.zeros(n, dtype=np.float32),
            qf=np.full(n, np.nan, dtype=np.float32),
        )
        write_compact(dec_dir, batch, meta)
        write_residual_v1(dec_dir, np.stack(res_all).astype(np.float16), meta)
        (dec_dir / "p0_table.json").write_text(json.dumps(rows, allow_nan=False), encoding="utf-8")
    print(f"[smoke] synthetic tree at {root}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="candidate_readouts")
    ap.add_argument(
        "--features-root", default=None, help="dir of <decoder>/ stores (def outputs/p0)"
    )
    ap.add_argument("--out", default=None, help="output dir (def outputs/p0/candidate_readouts)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="local synthetic self-test")
    args = ap.parse_args(argv)
    if args.smoke:
        tmp = Path(tempfile.mkdtemp(prefix="candidate_readouts_smoke_"))
        try:
            _smoke_tree(tmp / "root", seed=args.seed)
            return run(
                tmp / "root", tmp / "out", tmp / "work", seed=args.seed
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    features_root = Path(args.features_root) if args.features_root else (outputs_dir() / "p0")
    out_dir = Path(args.out) if args.out else (outputs_dir() / "p0" / "candidate_readouts")
    work_root = Path(tempfile.mkdtemp(prefix="candidate_readouts_work_"))
    try:
        return run(features_root, out_dir, work_root, seed=args.seed)
    finally:
        shutil.rmtree(work_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
