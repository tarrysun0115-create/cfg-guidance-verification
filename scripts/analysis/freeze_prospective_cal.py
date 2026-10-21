"""Freeze the prospective conformal calibration (heads, CAL thresholds, grouping) before generation.

First stage of the prospective-calibration batch behind Table VII (Section IV-D). Writes, per
decoder (sdxl, pixart_sigma), a hash-bound frozen_cal_<dec>.json holding everything the
prospective evaluation (eval_prospective_cal.py) is allowed to use:

  * the deployed heads (rho / zsum_dep / lda_dep) refit from the main-grid FIT split of the
    sigma_blur=0 cell at w=20 vs 1, stored as explicit weights;
  * per-readout conformal thresholds from the main-grid CAL-H0 split at alpha=0.05 for K=1 and
    K=2. CAL scores are permuted once with seed 0 (K=1 keeps the stored row order), cut into
    G = floor(n_cal / K) non-overlapping groups (remainder dropped), and the threshold is the
    k-th smallest CAL group mean with k = ceil((G+1)(1-alpha)) (cal_split_fpr's rule). Under
    exchangeable null groups this bounds the probability that a new null group exceeds the
    threshold by alpha; it is infeasible when k > G, i.e. below 19 groups at alpha=0.05;
  * the group-formation rule for the fresh side (non-overlapping groups; H0 seed 1, H1 seed 2);
  * the batch-2 generation plan (1024 prompts, 2 shards, prompt offset 110000, seed base
    3000000, 50 steps, w in {1, 20}), which ``fresh_batch_validation.py --stage predict`` must
    bind into frozen_<dec>.json before any row is generated;
  * same-grid reference operating points (main-grid TEST groups, seeds 1/2) for later
    predicted-versus-realized reporting;
  * ``freeze_sha256``: SHA-256 over the canonical JSON of all of the above (freeze integrity).

Binding chain: freeze_sha256 -> ``prospective_cal_sha256`` inside frozen_<dec>.json ->
``frozen_id`` -> every generated row. eval_prospective_cal.py re-verifies each link.

REFUSES to run if any frozen_cal_* artifact already exists (a freeze happens once), and REFUSES
if the target fresh dir already contains a plan, generated rows or a summary (that would make the
"prospective" label false).

Produces: the frozen heads and CAL thresholds evaluated in Table VII (Section IV-D) and the
batch-2 generation plan they are bound to.
Inputs: main-grid outputs/p0/<dec>/p0_table.json + the server-side feature
store; CPU.
Outputs: $CFGID_FRESH_DIR/frozen_cal_<dec>.json. The paper's outputs/fresh_cal_batch/ files are
the original freeze; re-running into a scratch CFGID_FRESH_DIR re-derives the same heads and
thresholds (the recorded env path, and hence the digest, differ).
Usage (with the batch-2 env exported; see README.md):
  export CFGID_FRESH_DIR=outputs/fresh_cal_batch
  export CFGID_PROMPT_OFFSET=110000
  export CFGID_SEED_BASE=3000000
  uv run --no-sync python scripts/analysis/freeze_prospective_cal.py

Strict JSON; no torch; no generation happens here.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

import numpy as np
from _fresh_batch_support import (
    EXPECTED_CHANNEL,
    PROMPT_OFFSET,
    SEED_BASE,
    W_H0,
    W_H1,
    fresh_dir,
    frozen_cal_sha256,
)
from _readout_common import (
    DEPLOYED_READOUTS,
    fit_deployed,
    load_residual_rows,
    score_deployed,
    write_json_strict,
)

from cfgid.paths import outputs_dir

CONTRAST = ({20.0}, {1.0})
DECODERS = ("sdxl", "pixart_sigma")
ALPHA = 0.05
K_LIST = (1, 2)
# Preregistered plan of the prospective-calibration (batch-2) fresh batch.
D7_PROMPT_OFFSET = 110_000
D7_SEED_BASE = 3_000_000
D7_N_PROMPTS = 1024
D7_N_SHARDS = 2


def _load_sibling(name: str):
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"{name}_reuse_fpc", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_CSF = _load_sibling("cal_split_fpr")


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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--root", default=None, help="main-grid store root (def outputs/p0)")
    args = ap.parse_args(argv)

    required_env = ("CFGID_FRESH_DIR", "CFGID_PROMPT_OFFSET", "CFGID_SEED_BASE")
    missing_env = [name for name in required_env if not os.environ.get(name)]
    if missing_env:
        raise SystemExit(
            f"[abort] missing required batch-2 env vars: {missing_env}; export all three "
            "values listed in README.md before freezing"
        )
    if PROMPT_OFFSET != D7_PROMPT_OFFSET or SEED_BASE != D7_SEED_BASE:
        raise SystemExit(
            "[abort] batch-2 prompt/seed offsets differ from the preregistered values "
            f"({PROMPT_OFFSET}/{SEED_BASE} != {D7_PROMPT_OFFSET}/{D7_SEED_BASE})"
        )
    fdir = fresh_dir()
    legacy_dir = outputs_dir() / "fresh_batch"
    if fdir.resolve() == legacy_dir.resolve():
        raise SystemExit(f"[abort] CFGID_FRESH_DIR resolves to the first batch: {legacy_dir}")
    fdir.mkdir(parents=True, exist_ok=True)
    for stale in fdir.glob("frozen_cal_*.json"):
        raise SystemExit(f"[abort] {stale} already exists; a freeze happens exactly once")
    plan_files = list(fdir.glob("frozen_*.json"))
    if plan_files:
        raise SystemExit(
            f"[abort] {fdir} already contains {plan_files[0].name}; run this freeze before predict"
        )
    temp_files = list(fdir.glob("frozen_*.json.tmp"))
    if temp_files:
        raise SystemExit(
            f"[abort] {fdir} contains an incomplete frozen artifact {temp_files[0].name}"
        )
    rows_files = list(fdir.glob("*_rows*.jsonl"))
    if rows_files:
        raise SystemExit(
            f"[abort] {fdir} already contains generated rows ({rows_files[0].name}...); "
            "freezing after generation would not be prospective"
        )
    summary_files = list(fdir.glob("*_summary.json"))
    if summary_files:
        raise SystemExit(
            f"[abort] {fdir} already contains a generated summary {summary_files[0].name}"
        )

    root = Path(args.root) if args.root else outputs_dir() / "p0"
    rows = load_residual_rows(root)
    cells: dict[tuple[str, float], list[dict]] = {}
    for r in rows:
        cells.setdefault((r["decoder"], r["sigma_blur"]), []).append(r)
    h1_set, h0_set = CONTRAST

    payloads: dict[str, dict] = {}
    for decoder in DECODERS:
        cell_rows = cells.get((decoder, 0.0))
        if not cell_rows:
            raise SystemExit(f"[abort] store cell {decoder}/sb=0 missing under {root}")
        payload: dict = {
            "schema": "frozen_prospective_cal_v1",
            "decoder": decoder,
            "contrast": "w20_vs_1",
            "alpha": ALPHA,
            "k_list": list(K_LIST),
            "group_rule": {
                "cal_permutation_seed": 0,
                "fresh_h0_seed": 1,
                "fresh_h1_seed": 2,
                "remainder": "dropped",
                "order_statistic_k": "ceil((G+1)*(1-alpha))",
            },
            "env": {
                "CFGID_FRESH_DIR": str(fdir),
                "CFGID_PROMPT_OFFSET": str(PROMPT_OFFSET),
                "CFGID_SEED_BASE": str(SEED_BASE),
            },
            "fresh_plan": {
                "prompt_offset": PROMPT_OFFSET,
                "seed_base": SEED_BASE,
                "n_prompts": D7_N_PROMPTS,
                "n_shards": D7_N_SHARDS,
                "steps": EXPECTED_CHANNEL["steps"],
                "w_list": [W_H0, W_H1],
            },
            "readouts": {},
        }
        for readout in DEPLOYED_READOUTS:
            head = fit_deployed(cell_rows, h1_set, h0_set, readout)
            if head is None:
                raise SystemExit(f"[abort] deployed fit failed: {decoder}/{readout}")
            cal = score_deployed(cell_rows, h0_set, "calibration", "h0", head)
            test0 = score_deployed(cell_rows, h0_set, "test", "h0", head)
            test1 = score_deployed(cell_rows, h1_set, "test", "h1", head)
            if min(cal.size, test0.size, test1.size) < max(K_LIST):
                raise SystemExit(
                    f"[abort] {decoder}/{readout}: CAL/TEST pools are too thin "
                    f"({cal.size}/{test0.size}/{test1.size})"
                )
            entry: dict = {
                "weights": {
                    "m0": [float(x) for x in np.atleast_1d(head.m0)],
                    "s0": [float(x) for x in np.atleast_1d(head.s0)],
                    "wvec": [float(x) for x in np.atleast_1d(head.wvec)],
                },
                "n_cal": int(cal.size),
                "thresholds": {},
                "same_grid_reference": {},
            }
            # Grouping seeds: CAL 0, TEST-H0 1, TEST-H1 2 (same as group_conformal_k.py).
            for k in K_LIST:
                cal_g = _group_means(cal, k, None if k == 1 else np.random.default_rng(0))
                thr = _CSF._conformal_threshold(cal_g, ALPHA)
                if thr is None:
                    raise SystemExit(
                        f"[abort] {decoder}/{readout}: K={k} infeasible at alpha={ALPHA} "
                        f"(G={cal_g.size}); the freeze plan assumed feasibility"
                    )
                g0 = _group_means(test0, k, None if k == 1 else np.random.default_rng(1))
                g1 = _group_means(test1, k, None if k == 1 else np.random.default_rng(2))
                entry["thresholds"][f"k{k}"] = {
                    "threshold": float(thr),
                    "n_cal_groups": int(cal_g.size),
                }
                entry["same_grid_reference"][f"k{k}"] = {
                    "fpr": float(np.mean(g0 > thr)) if g0.size else None,
                    "tpr": float(np.mean(g1 > thr)) if g1.size else None,
                    "n_test_h0_groups": int(g0.size),
                    "n_test_h1_groups": int(g1.size),
                }
            payload["readouts"][readout] = entry

        # Digest over the canonical JSON of the whole payload (the digest key itself excluded).
        payload["freeze_sha256"] = frozen_cal_sha256(payload)
        payloads[decoder] = payload

    # Validate every decoder completely before writing either artifact. A bad second cell
    # must not strand a partial "freeze once" campaign that cannot be retried safely.
    for decoder, payload in payloads.items():
        out = fdir / f"frozen_cal_{decoder}.json"
        write_json_strict(out, payload)
        lda = payload["readouts"]["lda_dep"]
        print(
            f"[freeze] {decoder:13s} n_cal={lda['n_cal']} "
            f"thr_k1={lda['thresholds']['k1']['threshold']:.6f} "
            f"thr_k2={lda['thresholds']['k2']['threshold']:.6f} "
            f"sha256={payload['freeze_sha256'][:16]}...  -> {out}"
        )

    print(
        "[freeze] done. NEXT (same env exported): fresh_batch_validation.py --stage predict "
        "--n-prompts 1024, then --stage run, then eval_prospective_cal.py"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
