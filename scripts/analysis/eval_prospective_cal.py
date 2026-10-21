"""Prospective evaluation of the frozen conformal calibration on the NEW fresh batch.

Final stage of the prospective-calibration batch (Table VII, Section IV-D). Runs AFTER
freeze_prospective_cal.py and the generation/inversion stages (fresh_batch_validation.py
--stage predict/run under the SAME batch-2 env). Per decoder and readout it:

  1. re-verifies the freeze: recomputes the SHA-256 of frozen_cal_<dec>.json's canonical
     content, checks that the generation plan frozen_<dec>.json binds that digest
     (``prospective_cal_sha256``), and refits the deployed head from the CURRENT main-grid FIT
     state, requiring agreement with the frozen weights within --head-tol (a drift gate only);
  2. scores the new batch's rows with the frozen head (never refit);
  3. forms non-overlapping mean-of-K groups with the frozen seeds (H0 seed 1, H1 seed 2; K=1
     uses the rows as-is; remainder dropped) and applies the FROZEN CAL thresholds -- no
     quantile is ever taken on the new data;
  4. reports realized group FPR with its exact one-sided Clopper--Pearson 95% upper bound
     (CP95), TPR, and single-image/K-group AUC, next to the frozen same-grid reference numbers.

Produces: Table VII (FPR / CP95 / TPR of rho, T_pair-zsum and T_pair-LDA at K=1 and K=2 on SDXL
and PixArt-Sigma) and the prospective-calibration statements of Section IV-D.
Inputs: outputs/fresh_cal_batch/{frozen_cal_<dec>.json, frozen_<dec>.json,
<dec>_rows_shard*.jsonl}; the drift gate also needs main-grid p0_table.json + the
server-side feature store; CPU.
Outputs: outputs/p0/prospective_cal_eval.json.
Usage (batch-2 env exported, see README.md):
  uv run --no-sync python scripts/analysis/eval_prospective_cal.py

Every console line carries [PROSPECTIVE]: thresholds and rules predate generation, by
construction and by hash. Strict JSON; no torch.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
from pathlib import Path

import numpy as np
from _fresh_batch_support import (
    EXPECTED_CHANNEL,
    PROMPT_OFFSET,
    SEED_BASE,
    W_H0,
    W_H1,
    cp_upper,
    finite_int,
    finite_vector,
    fresh_dir,
    frozen_cal_sha256,
    load_frozen_path,
)
from _readout_common import (
    DEPLOYED_READOUTS,
    fit_deployed,
    load_residual_rows,
    write_json_strict,
)

from cfgid.detector.empirical import auc
from cfgid.numerics import finite_float, loads_strict_json
from cfgid.paths import outputs_dir

CONTRAST = ({20.0}, {1.0})
DECODERS = ("sdxl", "pixart_sigma")
ALPHA = 0.05
K_LIST = (1, 2)
FROZEN_CAL_SCHEMA = "frozen_prospective_cal_v1"
GROUP_RULE = {
    "cal_permutation_seed": 0,
    "fresh_h0_seed": 1,
    "fresh_h1_seed": 2,
    "remainder": "dropped",
    "order_statistic_k": "ceil((G+1)*(1-alpha))",
}


def _load_sibling(name: str):
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"{name}_reuse_epc", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_FTC = _load_sibling("fresh_tail_compare")


def _group_means(x: np.ndarray, k: int, rng: np.random.Generator | None) -> np.ndarray:
    """Non-overlapping mean-of-k groups after one permutation; K=1 returns the rows unchanged."""
    v = np.asarray(x, dtype=np.float64).reshape(-1)
    if k == 1:
        return v
    if rng is not None:
        v = rng.permutation(v)
    g = v.size // k
    if g == 0:
        return np.empty(0)
    return v[: g * k].reshape(g, k).mean(axis=1)


def _object(value: object, context: str) -> dict:
    if not isinstance(value, dict):
        raise SystemExit(f"[abort] {context} must be a JSON object")
    return value


def _exact_keys(value: dict, expected: set[str], context: str) -> None:
    if set(value) != expected:
        raise SystemExit(f"[abort] {context} keys {sorted(value)} != expected {sorted(expected)}")


def _load_frozen_cal(
    path: Path, decoder: str, plan_frozen: dict, fdir: Path, head_tol: float
) -> tuple[dict, str]:
    """Validate frozen_cal_<dec>.json (digest, rules, env, plan binding, weights); return it."""
    if not path.is_file():
        raise SystemExit(f"[abort] {path} missing; run freeze_prospective_cal.py first")
    frozen = _object(loads_strict_json(path.read_text(encoding="utf-8")), str(path))
    _exact_keys(
        frozen,
        {
            "schema",
            "decoder",
            "contrast",
            "alpha",
            "k_list",
            "group_rule",
            "env",
            "fresh_plan",
            "readouts",
            "freeze_sha256",
        },
        str(path),
    )
    recorded = frozen["freeze_sha256"]
    if not isinstance(recorded, str) or recorded != frozen_cal_sha256(frozen):
        raise SystemExit(f"[abort] {decoder}: frozen-calibration sha256 mismatch")
    if frozen["schema"] != FROZEN_CAL_SCHEMA or frozen["decoder"] != decoder:
        raise SystemExit(f"[abort] {decoder}: frozen-calibration schema/decoder mismatch")
    if frozen["contrast"] != "w20_vs_1":
        raise SystemExit(f"[abort] {decoder}: frozen-calibration contrast drifted")
    if finite_float("frozen_cal.alpha", frozen["alpha"]) != ALPHA:
        raise SystemExit(f"[abort] {decoder}: frozen alpha is not {ALPHA}")
    if frozen["k_list"] != list(K_LIST):
        raise SystemExit(f"[abort] {decoder}: frozen K list is not {list(K_LIST)}")
    group_rule = _object(frozen["group_rule"], f"{path}: group_rule")
    if group_rule != GROUP_RULE:
        raise SystemExit(f"[abort] {decoder}: frozen group rule drifted: {group_rule!r}")

    env = _object(frozen["env"], f"{path}: env")
    _exact_keys(
        env,
        {"CFGID_FRESH_DIR", "CFGID_PROMPT_OFFSET", "CFGID_SEED_BASE"},
        f"{path}: env",
    )
    if Path(str(env["CFGID_FRESH_DIR"])).resolve() != fdir.resolve():
        raise SystemExit(f"[abort] {decoder}: CFGID_FRESH_DIR differs from the frozen value")
    if finite_int("frozen env prompt offset", env["CFGID_PROMPT_OFFSET"]) != PROMPT_OFFSET:
        raise SystemExit(f"[abort] {decoder}: CFGID_PROMPT_OFFSET differs from the freeze")
    if finite_int("frozen env seed base", env["CFGID_SEED_BASE"]) != SEED_BASE:
        raise SystemExit(f"[abort] {decoder}: CFGID_SEED_BASE differs from the freeze")

    plan = _object(plan_frozen["plan"], f"{decoder}: frozen generation plan")
    expected_plan = {
        key: plan[key]
        for key in (
            "prompt_offset",
            "seed_base",
            "n_prompts",
            "n_shards",
            "steps",
            "w_list",
        )
    }
    fresh_plan = _object(frozen["fresh_plan"], f"{path}: fresh_plan")
    if fresh_plan != expected_plan:
        raise SystemExit(
            f"[abort] {decoder}: generation plan {expected_plan!r} != frozen-cal plan "
            f"{fresh_plan!r}"
        )
    if plan["steps"] != EXPECTED_CHANNEL["steps"] or plan["w_list"] != [W_H0, W_H1]:
        raise SystemExit(f"[abort] {decoder}: generation channel/contrast plan drifted")
    if plan_frozen.get("prospective_cal_sha256") != recorded:
        raise SystemExit(
            f"[abort] {decoder}: frozen generation plan does not bind this calibration digest"
        )

    readouts = _object(frozen["readouts"], f"{path}: readouts")
    if set(readouts) != set(DEPLOYED_READOUTS):
        raise SystemExit(f"[abort] {decoder}: frozen readout set drifted")
    for readout in DEPLOYED_READOUTS:
        entry = _object(readouts[readout], f"{path}: readouts.{readout}")
        _exact_keys(
            entry,
            {"weights", "n_cal", "thresholds", "same_grid_reference"},
            f"{path}: readouts.{readout}",
        )
        weights = _object(entry["weights"], f"{path}: {readout}.weights")
        _exact_keys(weights, {"m0", "s0", "wvec"}, f"{path}: {readout}.weights")
        finite_vector(f"{readout}.weights.m0", weights["m0"], (2,))
        s0 = finite_vector(f"{readout}.weights.s0", weights["s0"], (2,))
        if np.any(s0 <= 0.0):
            raise SystemExit(f"[abort] {decoder}/{readout}: frozen s0 is not positive")
        finite_vector(f"{readout}.weights.wvec", weights["wvec"], (2,))
        n_cal = finite_int(f"{readout}.n_cal", entry["n_cal"], positive=True)
        thresholds = _object(entry["thresholds"], f"{path}: {readout}.thresholds")
        references = _object(entry["same_grid_reference"], f"{path}: {readout}.same_grid_reference")
        expected_kkeys = {f"k{k}" for k in K_LIST}
        if set(thresholds) != expected_kkeys or set(references) != expected_kkeys:
            raise SystemExit(f"[abort] {decoder}/{readout}: frozen K-key set drifted")
        for k in K_LIST:
            kkey = f"k{k}"
            tinfo = _object(thresholds[kkey], f"{path}: {readout}.{kkey}.threshold")
            _exact_keys(
                tinfo,
                {"threshold", "n_cal_groups"},
                f"{path}: {readout}.{kkey}.threshold",
            )
            finite_float(f"{readout}.{kkey}.threshold", tinfo["threshold"])
            n_groups = finite_int(
                f"{readout}.{kkey}.n_cal_groups", tinfo["n_cal_groups"], positive=True
            )
            if n_groups != n_cal // k:
                raise SystemExit(
                    f"[abort] {decoder}/{readout}/{kkey}: n_cal_groups {n_groups} "
                    f"!= floor({n_cal}/{k})"
                )
            ref = _object(references[kkey], f"{path}: {readout}.{kkey}.reference")
            _exact_keys(
                ref,
                {"fpr", "tpr", "n_test_h0_groups", "n_test_h1_groups"},
                f"{path}: {readout}.{kkey}.reference",
            )
            for rate in ("fpr", "tpr"):
                value = finite_float(f"{readout}.{kkey}.{rate}", ref[rate])
                if not 0.0 <= value <= 1.0:
                    raise SystemExit(f"[abort] {decoder}/{readout}/{kkey}: {rate} outside [0,1]")
            for count in ("n_test_h0_groups", "n_test_h1_groups"):
                finite_int(f"{readout}.{kkey}.{count}", ref[count], positive=True)

    # The generation plan carries the original fresh-pipeline LDA head. It is not used to
    # score this batch, but it must describe the same FIT state as the prospective calibration.
    plan_weights = _object(plan_frozen["weights"], f"{decoder}: frozen generation weights")
    cal_weights = readouts["lda_dep"]["weights"]
    for name in ("m0", "s0", "wvec"):
        got = finite_vector(f"{decoder}: generation weights.{name}", plan_weights[name], (2,))
        want = finite_vector(
            f"{decoder}: calibration lda_dep weights.{name}", cal_weights[name], (2,)
        )
        deviation = float(np.max(np.abs(got - want)))
        if deviation > head_tol:
            raise SystemExit(
                f"[abort] {decoder}: generation/calibration head '{name}' differs by "
                f"{deviation:.3g}; the frozen artifacts do not share one FIT state"
            )
    return frozen, recorded


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--root", default=None, help="main-grid store root (def outputs/p0)")
    ap.add_argument("--head-tol", type=float, default=1e-6)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if not math.isfinite(args.head_tol) or args.head_tol < 0.0:
        raise SystemExit("--head-tol must be finite and non-negative")

    required_env = ("CFGID_FRESH_DIR", "CFGID_PROMPT_OFFSET", "CFGID_SEED_BASE")
    missing_env = [name for name in required_env if not os.environ.get(name)]
    if missing_env:
        raise SystemExit(
            f"[abort] missing required batch-2 env vars: {missing_env}; export the batch-2 env "
            "listed in README.md"
        )
    fdir = fresh_dir()
    root = Path(args.root) if args.root else outputs_dir() / "p0"
    rows = load_residual_rows(root)
    cells: dict[tuple[str, float], list[dict]] = {}
    for r in rows:
        cells.setdefault((r["decoder"], r["sigma_blur"]), []).append(r)
    h1_set, h0_set = CONTRAST

    results: dict[str, dict] = {}
    freeze_digests: dict[str, str] = {}
    for decoder in DECODERS:
        plan_frozen = load_frozen_path(fdir / f"frozen_{decoder}.json", decoder)
        cal_path = fdir / f"frozen_cal_{decoder}.json"
        frozen_cal, freeze_digest = _load_frozen_cal(
            cal_path, decoder, plan_frozen, fdir, args.head_tol
        )
        freeze_digests[decoder] = freeze_digest
        fresh = _FTC._load_fresh_rows(fdir, decoder, plan_frozen)
        cell_rows = cells.get((decoder, 0.0))
        if not cell_rows:
            raise SystemExit(f"[abort] store cell {decoder}/sb=0 missing under {root}")

        per_readout: dict[str, dict] = {}
        for readout in DEPLOYED_READOUTS:
            fr = frozen_cal["readouts"][readout]
            frozen_m0 = finite_vector(f"{decoder}/{readout}.weights.m0", fr["weights"]["m0"], (2,))
            frozen_s0 = finite_vector(f"{decoder}/{readout}.weights.s0", fr["weights"]["s0"], (2,))
            frozen_wvec = finite_vector(
                f"{decoder}/{readout}.weights.wvec", fr["weights"]["wvec"], (2,)
            )
            head = fit_deployed(cell_rows, h1_set, h0_set, readout)
            if head is None:
                raise SystemExit(f"[abort] deployed fit failed: {decoder}/{readout}")
            for name, got, want in (
                ("m0", head.m0, frozen_m0),
                ("s0", head.s0, frozen_s0),
                ("wvec", head.wvec, frozen_wvec),
            ):
                dev = float(np.max(np.abs(np.atleast_1d(got) - want)))
                if dev > args.head_tol:
                    raise SystemExit(
                        f"[abort] {decoder}/{readout}: head '{name}' deviates from the "
                        f"frozen weights by {dev:.3g}; FIT state drifted since the freeze"
                    )
            # The refit above is only a drift gate. Evaluation uses the frozen head itself.
            pack = _FTC._score(fresh, frozen_m0, frozen_s0, frozen_wvec)
            f0 = np.asarray(pack["h0"], dtype=np.float64)
            f1 = np.asarray(pack["h1"], dtype=np.float64)
            if f0.size == 0 or f1.size == 0:
                raise SystemExit(f"[abort] {decoder}/{readout}: empty fresh score side")
            labels = np.concatenate([np.zeros(f0.size, int), np.ones(f1.size, int)])
            entry: dict = {
                "n_fresh_h0": int(f0.size),
                "n_fresh_h1": int(f1.size),
                "auc1": float(auc(np.concatenate([f0, f1]), labels)),
            }
            for k in K_LIST:
                kkey = f"k{k}"
                tinfo = fr["thresholds"][kkey]
                thr = float(tinfo["threshold"])
                h0_seed = int(frozen_cal["group_rule"]["fresh_h0_seed"])
                h1_seed = int(frozen_cal["group_rule"]["fresh_h1_seed"])
                g0 = _group_means(f0, k, None if k == 1 else np.random.default_rng(h0_seed))
                g1 = _group_means(f1, k, None if k == 1 else np.random.default_rng(h1_seed))
                if g0.size == 0 or g1.size == 0:
                    raise SystemExit(
                        f"[abort] {decoder}/{readout}: empty fresh group side at K={k}"
                    )
                # A group is flagged when its mean score exceeds the frozen CAL threshold.
                fp = int(np.sum(g0 > thr))
                tp = int(np.sum(g1 > thr))
                ref = fr["same_grid_reference"][kkey]
                group_labels = np.concatenate([np.zeros(g0.size, int), np.ones(g1.size, int)])
                entry[kkey] = {
                    "threshold_frozen": thr,
                    "auc_k": float(auc(np.concatenate([g0, g1]), group_labels)),
                    "fpr": fp / g0.size if g0.size else None,
                    "fpr_cp95_upper": cp_upper(fp, int(g0.size)) if g0.size else None,
                    "tpr": tp / g1.size if g1.size else None,
                    "n_h0_groups": int(g0.size),
                    "n_h1_groups": int(g1.size),
                    "same_grid_reference_fpr": ref["fpr"],
                    "same_grid_reference_tpr": ref["tpr"],
                }
                print(
                    f"[PROSPECTIVE] {decoder:13s} {readout:9s} K={k} "
                    f"FPR={entry[kkey]['fpr']:.4f} (CP95 {entry[kkey]['fpr_cp95_upper']:.4f}) "
                    f"TPR={entry[kkey]['tpr']:.4f}  [same-grid ref "
                    f"{ref['fpr']:.4f}/{ref['tpr']:.4f}]"
                )
            per_readout[readout] = entry
        results[decoder] = per_readout

    out = Path(args.out) if args.out else outputs_dir() / "p0" / "prospective_cal_eval.json"
    write_json_strict(
        out,
        {
            "note": (
                "PROSPECTIVE calibration transport: heads, thresholds, and grouping rules "
                "were frozen before generation; frozen_cal sha256 is bound into frozen_id "
                "and therefore into every fresh row. No quantity was fitted or selected on "
                "the new data."
            ),
            "alpha": ALPHA,
            "fresh_dir": str(fdir),
            "head_tolerance": args.head_tol,
            "freeze_sha256_by_decoder": freeze_digests,
            "results": results,
        },
    )
    print(f"[PROSPECTIVE] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
