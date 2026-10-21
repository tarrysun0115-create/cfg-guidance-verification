"""Prompt-mismatch audit of the pair readouts: wrong-prompt re-inversion plus deployed-head audit.

The stored main-grid pass recorded, for the unblurred baseline cell, a scalar
``rho_summary_wrong_prompt`` (rho re-measured with another image's prompt) but no per-patch
residuals, so the magnitude coordinate m(x) under a mismatched prompt cannot be recovered from
the existing artifacts. This script therefore RE-INVERTS the same images with the same permuted
prompt, this time capturing the pair (rho_p, ||dr_p||), and asks whether the pair readouts inherit
rho's prompt conditionality. It is a GPU job: about 700 dual-branch inversions at T=50 across
the five decoders.

Protocol (reconstructed from run_p0's own recipe; no new parameters):

* target rows  = rows carrying ``rho_summary_wrong_prompt`` (run_p0's baseline cell:
  sigma_blur=0, QF = ``evidence.wrong_prompt_subset_qf``, no post-processing), restricted to
  split=test and w in {1, w_high};
* wrong prompt = bank[(pi+1) % n], with the prompt bank reconstructed from the table itself
  (pi parsed from image_id; bank[pi] is asserted equal to every row's prompt_text, so the
  mapping is the one run_p0 used);
* inversion    = backend.dual_branch_residual(stored PNG, wrong prompt, num_steps = configured
  inversion steps, seed = the row's gen_seed, capture_step_frac) ->
  build_inversion_result(configured patch size, denominator floor, log ratio, E_tex method).

Reproduction gate (fail-loud): the recomputed wrong-prompt rho must match the stored
``rho_summary_wrong_prompt`` within --tol (default 1e-3) for EVERY image; the first deviation
aborts the run, so every recorded magnitude coordinate comes from a re-inversion that
reproduced the original wrong-prompt rho.

Audit (deployed orientation): per cell, all FIT quantities come from the CORRECT-prompt
table/store only (per-coordinate FIT-H0 mean/std and FIT sign; the 2-D fusion weights of
pair_readout_battery._fit_stats_2d) and the SAME transforms are applied unchanged to the
wrong-prompt values. Readouts: rho, diff_energy (m), fusion_zsum (T_pair-zsum) and
fusion_lda_full (T_pair-LDA). Output: matched vs mismatched AUC1 and per-image matched/mismatched
score correlations, pooled over decoders (raw score pooling) and per decoder.

Produces:
    Table IX (Section IV-E2): matched AUC1, mismatched AUC1 and per-image correlation for rho,
    T_pair-LDA, T_pair-zsum and m (``prompt_mismatch_pair.json``:
    ``audit/<readout>/{correct_auc,wrong_auc_deployed,pearson_correct_wrong}``), and the
    per-decoder collapse quoted in Section IV-E2 (``audit/per_decoder/<decoder>/<readout>``).

Inputs: ``outputs/p0/<decoder>/p0_table.json``, the stored PNG images, the frozen backbone
weights and a GPU for the re-inversion; the audit additionally needs the server-side residual
feature store (``residual_v1.dat``, ``compact_rho.npy``) for the correct-prompt coordinates.
``--audit-only`` re-runs the audit from the per-image pair records (no GPU).

Outputs: ``outputs/p0/prompt_mismatch_pair/`` (or ``--out``):
``prompt_mismatch_pairs_<decoder>.jsonl`` (per-image wrong-prompt rho and m; checkpointed per
image, a re-run resumes and skips finished ids) and ``prompt_mismatch_pair.json``.

Usage:
    uv run --no-sync python scripts/analysis/prompt_mismatch_pair.py --limit 8   # gate check
    uv run --no-sync python scripts/analysis/prompt_mismatch_pair.py             # full run
    uv run --no-sync python scripts/analysis/prompt_mismatch_pair.py --audit-only
    uv run --no-sync python scripts/analysis/prompt_mismatch_pair.py --smoke
The smoke test (no GPU) exercises row selection, bank reconstruction, resume and the full audit
on a synthetic store; the inversion call itself is run_p0's own, unmodified.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np

from cfgid.paths import outputs_dir
from cfgid.sweep.neff_scaling import _load_p0_rows

W_HIGH_DEFAULT = 20
# run_p0 ids look like "<decoder>_p<pi>_s<seed>_w<w>_..."; the shared smoke tree uses
# "<dec>_sb<sb>_w<w>_p<p>_s0" (index at the tail) -- accept both, fail loud on anything else.
_PI_RE = re.compile(r"_p(\d+)_s\d+(?:_w|$)")


def _load_sibling(name: str):
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"{name}_reuse_g4", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_FM = _load_sibling("candidate_readouts")  # store loading + rho repro gate (+ _FC inside)
_PRB = _load_sibling("pair_readout_battery")  # _fit_stats_2d / _apply_fusion


def _parse_pi(image_id: str) -> int:
    m = _PI_RE.search(image_id)
    if m is None:
        raise ValueError(f"cannot parse prompt index from image_id {image_id!r}")
    return int(m.group(1))


def _prompt_bank(rows: list[dict]) -> list[str]:
    """Reconstruct run_p0's prompt bank from the table; fail loud on any inconsistency."""
    by_pi: dict[int, str] = {}
    for r in rows:
        pi = _parse_pi(str(r["image_id"]))
        text = str(r["prompt_text"])
        if by_pi.setdefault(pi, text) != text:
            raise SystemExit(f"[abort] prompt index {pi} maps to two different prompt texts")
    n = max(by_pi) + 1
    missing = [i for i in range(n) if i not in by_pi]
    if missing:
        raise SystemExit(
            f"[abort] prompt bank incomplete (missing indices {missing[:5]}...); the "
            "(pi+1) mod n wrong-prompt mapping would be wrong -- refusing."
        )
    return [by_pi[i] for i in range(n)]


def _target_rows(rows: list[dict], *, w_high: int, all_rows: bool) -> list[dict]:
    out = []
    for r in rows:
        if r.get("rho_summary_wrong_prompt") is None:
            continue
        if not all_rows and (r.get("split") != "test" or int(r["w"]) not in (1, w_high)):
            continue
        out.append(r)
    return out


def _load_pairs(path: Path) -> dict[str, dict]:
    pairs: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                pairs[str(rec["image_id"])] = rec
    return pairs


def _reinvert_decoder(
    dec_dir: Path,
    rows: list[dict],
    pairs_path: Path,
    *,
    w_high: int,
    tol: float,
    limit: int | None,
    all_rows: bool,
) -> dict:
    """Re-invert the wrong-prompt subset of one decoder; checkpointed + gated per image."""
    from PIL import Image

    from cfgid.channel.features import build_inversion_result
    from cfgid.channel.inversion.registry import get_backend
    from cfgid.channel.rho import image_rho_summary
    from cfgid.config import load_models_config
    from cfgid.device import get_device
    from cfgid.p0config import load_p0_config

    cfg = load_p0_config()
    models = load_models_config()
    decoder = dec_dir.name
    if decoder not in models:
        return {"skipped": f"decoder {decoder!r} not in models config"}
    spec = models[decoder]
    if not spec.is_present():
        return {"skipped": f"weights for {decoder!r} not present"}
    targets = _target_rows(rows, w_high=w_high, all_rows=all_rows)
    if not targets:
        return {"skipped": "no rows carry rho_summary_wrong_prompt"}
    bank = _prompt_bank(rows)
    for r in targets:  # mapping proof: the bank IS the list run_p0 indexed
        if bank[_parse_pi(str(r["image_id"]))] != str(r["prompt_text"]):
            raise SystemExit("[abort] prompt bank does not reproduce a row's prompt_text")
    done = _load_pairs(pairs_path)
    todo = [r for r in targets if str(r["image_id"]) not in done]
    if limit is not None:
        todo = todo[: max(0, limit - len(done))]
    print(
        f"[{decoder}] targets={len(targets)} done={len(done)} todo={len(todo)} "
        f"(bank n={len(bank)}, steps={cfg.patch.inversion_steps})"
    )
    if not todo:
        return {"n_targets": len(targets), "n_done": len(done), "worst_abs_dev": None}

    backend = get_backend(spec, get_device())
    backend.load()
    worst = 0.0
    pairs_path.parent.mkdir(parents=True, exist_ok=True)
    with pairs_path.open("a", encoding="utf-8") as fh:
        for k, r in enumerate(todo):
            iid = str(r["image_id"])
            pi = _parse_pi(iid)
            wrong = bank[(pi + 1) % len(bank)]
            png = dec_dir / str(r["image"])
            if not png.exists():
                raise SystemExit(f"[abort] stored PNG missing: {png}")
            img = np.asarray(Image.open(png).convert("RGB"), dtype=np.float32) / 255.0
            dual = backend.dual_branch_residual(
                img,
                wrong,
                num_steps=cfg.patch.inversion_steps,
                seed=int(r["gen_seed"]),
                capture_step_frac=cfg.patch.capture_step_frac,
            )
            res = build_inversion_result(
                dual,
                patch_size=cfg.patch.patch_size,
                floor=cfg.rho.denom_floor,
                log_ratio=cfg.rho.log_ratio,
                sigma=0.0,  # metadata only; rho/diff do not read it
                e_tex_method=cfg.e_tex.method,
                image_id=f"{iid}_wrongprompt",
            )
            rho_w = image_rho_summary(
                res.rho_patch,
                res.e_tex_patch,
                mask_quantile=cfg.rho.low_etex_mask_quantile,
                winsor_quantiles=cfg.rho.winsor_quantiles,
            )
            stored = float(r["rho_summary_wrong_prompt"])
            dev = abs(rho_w - stored)
            worst = max(worst, dev)
            if dev > tol:
                raise SystemExit(
                    f"[abort] {decoder}/{iid}: recomputed wrong-prompt rho {rho_w:.6f} deviates "
                    f"from stored {stored:.6f} by {dev:.4g} (> {tol:g}). The re-inversion path "
                    "does not reproduce the original pass; NO pair numbers are emitted."
                )
            norms = np.linalg.norm(np.asarray(res.residual, dtype=np.float64), axis=1)
            diff_w = float(_FM._FC._image_summary(norms, np.asarray(res.e_tex_patch)))
            rec = {
                "image_id": iid,
                "decoder": decoder,
                "w": int(r["w"]),
                "label": str(r["label"]),
                "split": str(r["split"]),
                "pi": pi,
                "rho_wrong": float(rho_w),
                "stored_rho_wrong": stored,
                "abs_dev": float(dev),
                "diff_wrong": diff_w,
            }
            fh.write(json.dumps(rec, allow_nan=False) + "\n")
            fh.flush()
            if (k + 1) % 25 == 0 or k + 1 == len(todo):
                print(f"  [{decoder}] {k + 1}/{len(todo)} re-inverted (worst dev {worst:.2e})")
    return {"n_targets": len(targets), "n_done": len(done) + len(todo), "worst_abs_dev": worst}


# --- audit -----------------------------------------------------------------------------------


def _cell_coord_stats(
    rows: list[dict], feats: dict[str, dict[str, float]], name: str, *, w_high: int, min_n: int
) -> dict[str, tuple[float, float, float]]:
    """Per-cell FIT (m0, s0, sign) for one scalar coordinate, from CORRECT-prompt train rows.

    H0 = h0 rows, H1 = h1 rows at ``w_high``; m0/s0 are the FIT-H0 mean and std (ddof=1) and the
    sign is that of the FIT H1-minus-H0 mean shift (0 -> +1). Cells below ``min_n`` per side are
    skipped.
    """
    pools: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        iid = str(r["image_id"])
        if iid not in feats[name] or r.get("split") != "train":
            continue
        lab = str(r.get("label"))
        w = int(r["w"])
        side = "h0" if lab == "h0" else ("h1" if lab == "h1" and w == w_high else "")
        if side:
            pools.setdefault(_PRB._cell_key(r), {}).setdefault(side, []).append(feats[name][iid])
    out: dict[str, tuple[float, float, float]] = {}
    for key, sides in pools.items():
        h0 = np.asarray(sides.get("h0", []), dtype=np.float64)
        h1 = np.asarray(sides.get("h1", []), dtype=np.float64)
        if min(h0.size, h1.size) < min_n:
            continue
        m0, s0 = float(h0.mean()), float(h0.std(ddof=1))
        if not (np.isfinite(s0) and s0 > 0):
            continue
        s = float(np.sign(h1.mean() - m0)) or 1.0
        out[key] = (m0, s0, s)
    return out


def _audit(
    features_root: Path,
    pairs_by_decoder: dict[str, dict[str, dict]],
    *,
    w_high: int,
    min_per_cell: int,
) -> dict:
    """Matched vs mismatched AUC on TEST rows at w=1 vs ``w_high``, per decoder and pooled.

    Each image contributes its correct-prompt score (``cz``) and its mismatched-prompt score
    (``wz``) under the SAME correct-prompt FIT transform, so the two AUCs share labels and rows.
    """
    from scipy.stats import rankdata

    from cfgid.detector.empirical import auc

    def _pearson(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.corrcoef(a, b)[0, 1]) if (np.std(a) > 0 and np.std(b) > 0) else 0.0

    def _spearman(a: np.ndarray, b: np.ndarray) -> float:
        return _pearson(
            np.asarray(rankdata(a), dtype=np.float64), np.asarray(rankdata(b), dtype=np.float64)
        )

    candidates = ("rho", "diff_energy", "fusion_zsum", "fusion_lda_full")
    pooled: dict[str, dict[str, list[float]]] = {
        c: {"cz": [], "wz": [], "lab": []} for c in candidates
    }
    per_decoder: dict[str, dict] = {}
    for decoder, pairs in sorted(pairs_by_decoder.items()):
        dec_dir = features_root / decoder
        rows = _load_p0_rows(dec_dir / "p0_table.json")
        feats = _FM._per_image_features(dec_dir, rows)  # includes the correct-rho repro gate
        st2d = _PRB._fit_stats_2d(rows, feats, (1,), (w_high,))
        fused_c = {
            "fusion_zsum": _PRB._apply_fusion(rows, feats, st2d, "w_zsum"),
            "fusion_lda_full": _PRB._apply_fusion(rows, feats, st2d, "w_full"),
        }
        coord_stats = {
            name: _cell_coord_stats(rows, feats, name, w_high=w_high, min_n=min_per_cell)
            for name in ("rho", "diff_energy")
        }
        dec_pool: dict[str, dict[str, list[float]]] = {
            c: {"cz": [], "wz": [], "lab": []} for c in candidates
        }
        for r in rows:
            iid = str(r["image_id"])
            rec = pairs.get(iid)
            if rec is None or r.get("split") != "test" or int(r["w"]) not in (1, w_high):
                continue
            key = _PRB._cell_key(r)
            lab = 1 if int(r["w"]) == w_high else 0
            # scalar coordinates: deployed per-cell FIT (m0, s0, sign) applied to both sides
            wrong_val = {"rho": float(rec["rho_wrong"]), "diff_energy": float(rec["diff_wrong"])}
            for name in ("rho", "diff_energy"):
                st = coord_stats[name].get(key)
                if st is None or iid not in feats[name]:
                    continue
                m0, s0, s = st
                dec_pool[name]["cz"].append(s * (feats[name][iid] - m0) / s0)
                dec_pool[name]["wz"].append(s * (wrong_val[name] - m0) / s0)
                dec_pool[name]["lab"].append(lab)
            # pair readouts: deployed per-cell 2-D weights applied to both sides
            st2 = st2d.get(key)
            if st2 is None:
                continue
            y_w = np.array([rec["rho_wrong"], rec["diff_wrong"]], dtype=np.float64)
            z_w = (y_w - np.asarray(st2["m"])) / np.asarray(st2["s"])
            for cand, wkey in (("fusion_zsum", "w_zsum"), ("fusion_lda_full", "w_full")):
                if iid not in fused_c[cand]:
                    continue
                dec_pool[cand]["cz"].append(fused_c[cand][iid])
                dec_pool[cand]["wz"].append(float(np.dot(np.asarray(st2[wkey]), z_w)))
                dec_pool[cand]["lab"].append(lab)
        summary: dict[str, dict] = {}
        for cand in candidates:
            cz = np.asarray(dec_pool[cand]["cz"])
            wz = np.asarray(dec_pool[cand]["wz"])
            lab_arr = np.asarray(dec_pool[cand]["lab"], dtype=int)
            if lab_arr.size >= 2 * min_per_cell and 0 < lab_arr.sum() < lab_arr.size:
                summary[cand] = {
                    "n_paired": int(lab_arr.size),
                    "correct_auc": float(auc(cz, lab_arr)),
                    "wrong_auc_deployed": float(auc(wz, lab_arr)),
                    "pearson_correct_wrong": _pearson(cz, wz),
                }
            for slot, vals in dec_pool[cand].items():
                pooled[cand][slot].extend(vals)
        per_decoder[decoder] = summary
    out: dict = {"per_decoder": per_decoder}
    for cand in candidates:
        cz = np.asarray(pooled[cand]["cz"])
        wz = np.asarray(pooled[cand]["wz"])
        lab_arr = np.asarray(pooled[cand]["lab"], dtype=int)
        if lab_arr.size < 2 * min_per_cell or lab_arr.sum() in (0, lab_arr.size):
            out[cand] = {"skipped": f"pooled pair set too small ({lab_arr.size})"}
            continue
        out[cand] = {
            "n_paired": int(lab_arr.size),
            "correct_auc": float(auc(cz, lab_arr)),
            "wrong_auc_deployed": float(auc(wz, lab_arr)),
            "pearson_correct_wrong": _pearson(cz, wz),
            "spearman_correct_wrong": _spearman(cz, wz),
        }
    return out


def run(
    features_root: Path,
    out_dir: Path,
    *,
    w_high: int,
    tol: float,
    limit: int | None,
    decoders: list[str] | None,
    audit_only: bool,
    all_rows: bool,
    min_per_cell: int,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    reinvert_stats: dict[str, dict] = {}
    pairs_by_decoder: dict[str, dict[str, dict]] = {}
    for dec_dir in sorted(d for d in features_root.glob("*/") if (d / "p0_table.json").exists()):
        decoder = dec_dir.name
        if decoders and decoder not in decoders:
            continue
        pairs_path = out_dir / f"prompt_mismatch_pairs_{decoder}.jsonl"
        if not audit_only:
            rows = _load_p0_rows(dec_dir / "p0_table.json")
            reinvert_stats[decoder] = _reinvert_decoder(
                dec_dir,
                rows,
                pairs_path,
                w_high=w_high,
                tol=tol,
                limit=limit,
                all_rows=all_rows,
            )
        pairs = _load_pairs(pairs_path)
        if pairs:
            pairs_by_decoder[decoder] = pairs
    if not pairs_by_decoder:
        print("[skip] no prompt-mismatch pair records anywhere; nothing to audit")
        return 0
    audit = _audit(features_root, pairs_by_decoder, w_high=w_high, min_per_cell=min_per_cell)
    out = {
        "analysis": "prompt_mismatch_pair",
        "protocol": {
            "subset": f"run_p0 wrong-prompt baseline cell; split=test, w in {{1, {w_high}}}",
            "wrong_prompt": "bank[(pi+1) mod n], bank reconstructed+verified from the table",
            "orientation": "deployed: per-cell FIT stats/weights from CORRECT-prompt data only",
            "gate": f"recomputed wrong-prompt rho == stored field within {tol:g} per image",
            "vocabulary": "T_pair-LDA/T_pair-zsum are split-calibrated readouts, never label-free",
        },
        "reinvert": reinvert_stats,
        "audit": audit,
    }
    out_path = out_dir / "prompt_mismatch_pair.json"
    out_path.write_text(json.dumps(out, indent=2, allow_nan=False), encoding="utf-8")
    print("\n=== prompt-mismatch audit (deployed orientation) ===")
    for cand in ("rho", "diff_energy", "fusion_zsum", "fusion_lda_full"):
        e = audit.get(cand, {})
        if "skipped" in e:
            print(f"{cand:18s} SKIPPED: {e['skipped']}")
            continue
        print(
            f"{cand:18s} correct={e['correct_auc']:.3f} -> wrong[deployed]="
            f"{e['wrong_auc_deployed']:.3f}  corr r={e['pearson_correct_wrong']:+.2f} "
            f"(n={e['n_paired']})"
        )
    print(f"-> {out_path}")
    return 0


# --- smoke: audit/bank/resume plumbing on the synthetic store (no GPU, no inversion) -----------


def _smoke(seed: int) -> int:
    tmp = Path(tempfile.mkdtemp(prefix="prompt_mismatch_pair_smoke_"))
    try:
        _FM._smoke_tree(tmp / "root", seed=seed)
        rng = np.random.default_rng(seed)
        for tp in sorted((tmp / "root").glob("*/p0_table.json")):
            rows = json.loads(tp.read_text(encoding="utf-8"))
            for r in rows:
                r["prompt_text"] = f"synthetic prompt {_parse_pi(str(r['image_id']))}"
                r["image"] = f"images/{r['image_id']}.png"
                r["gen_seed"] = 0
                if float(r["sigma_blur"]) == 0.0 and int(r["w"]) in (1, W_HIGH_DEFAULT):
                    r["rho_summary_wrong_prompt"] = float(
                        0.5 * r["rho_summary"] + rng.normal(0.0, 0.05)
                    )
            tp.write_text(json.dumps(rows, allow_nan=False), encoding="utf-8")
            # synthesize the pairs a real run would have written (wrong side = damped + noisy)
            dec = tp.parent.name
            pairs_path = tmp / "out" / f"prompt_mismatch_pairs_{dec}.jsonl"
            pairs_path.parent.mkdir(parents=True, exist_ok=True)
            with pairs_path.open("w", encoding="utf-8") as fh:
                for r in rows:
                    if r.get("rho_summary_wrong_prompt") is None or r.get("split") != "test":
                        continue
                    fh.write(
                        json.dumps(
                            {
                                "image_id": r["image_id"],
                                "decoder": dec,
                                "w": int(r["w"]),
                                "label": r["label"],
                                "split": r["split"],
                                "pi": _parse_pi(str(r["image_id"])),
                                "rho_wrong": float(r["rho_summary_wrong_prompt"]),
                                "stored_rho_wrong": float(r["rho_summary_wrong_prompt"]),
                                "abs_dev": 0.0,
                                "diff_wrong": float(rng.normal(1.0, 0.1)),
                            },
                            allow_nan=False,
                        )
                        + "\n"
                    )
            bank = _prompt_bank([dict(r) for r in rows])
            if not bank or not bank[0].startswith("synthetic prompt"):
                raise SystemExit("[smoke] prompt bank reconstruction failed")
        rc = run(
            tmp / "root",
            tmp / "out",
            w_high=W_HIGH_DEFAULT,
            tol=1e-3,
            limit=None,
            decoders=None,
            audit_only=True,
            all_rows=False,
            min_per_cell=4,
        )
        if rc != 0 or not (tmp / "out" / "prompt_mismatch_pair.json").exists():
            raise SystemExit("[smoke] audit-only run failed")
        print("[smoke] PASS (audit/bank/resume plumbing; inversion path is run_p0's own)")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="prompt_mismatch_pair")
    ap.add_argument(
        "--features-root", default=None, help="dir of <decoder>/ stores (def outputs/p0)"
    )
    ap.add_argument("--out", default=None, help="output dir (def outputs/p0/prompt_mismatch_pair)")
    ap.add_argument("--w-high", type=int, default=W_HIGH_DEFAULT)
    ap.add_argument("--tol", type=float, default=1e-3, help="rho reproduction gate tolerance")
    ap.add_argument("--limit", type=int, default=None, help="cap re-inversions per decoder")
    ap.add_argument("--decoder", action="append", default=None, help="restrict to decoder(s)")
    ap.add_argument("--audit-only", action="store_true", help="skip re-inversion; audit pairs")
    ap.add_argument(
        "--all-target-rows",
        action="store_true",
        help="re-invert every row carrying the field (default: test split, w in {1, w_high})",
    )
    ap.add_argument("--min-per-cell", type=int, default=8)
    ap.add_argument("--smoke", action="store_true", help="local plumbing self-test (no GPU)")
    args = ap.parse_args(argv)
    if args.smoke:
        return _smoke(0)
    features_root = Path(args.features_root) if args.features_root else (outputs_dir() / "p0")
    out_dir = Path(args.out) if args.out else (outputs_dir() / "p0" / "prompt_mismatch_pair")
    return run(
        features_root,
        out_dir,
        w_high=args.w_high,
        tol=args.tol,
        limit=args.limit,
        decoders=args.decoder,
        audit_only=args.audit_only,
        all_rows=args.all_target_rows,
        min_per_cell=args.min_per_cell,
    )


if __name__ == "__main__":
    raise SystemExit(main())
