"""Fresh-batch tail comparison: Pair-LDA vs Pair-zsum on the INDEPENDENT fresh batch (GPU-free).

Compares the two deployed pair readouts on independent data: both are frozen from the MAIN-GRID
store (exact ``fit_deployed`` heads for the sdxl/pixart_sigma sigma_blur=0, w=20-vs-1 cells) and
evaluated on the first preregistered fresh batch, which shares no image or prompt with the grid
and uses its own seed base.

Safety gates:
* the store-refit lda_dep head must reproduce the frozen head recorded in
  ``outputs/fresh_batch/frozen_<decoder>.json`` (m0/s0/wvec elementwise; tolerance --head-tol) --
  this proves the zsum_dep head (same fit, sign weights) belongs to the same frozen FIT state;
* the frozen artifact's schema/hash/channel/plan are validated, and the exact committed shard and
  image-id set is required; every row is checked against that plan (no partial or mixed batches).

Reports per decoder and readout: fresh AUC_1; TPR at matched-quantile fresh-H0 thresholds
(alpha = 1%/5%); FPR/TPR at the FROZEN main-grid thresholds where recorded (lda_dep only -- the
zsum head has no frozen threshold, reported as null); and the paired prompt-level bootstrap 95%
CI of the LDA-minus-zsum AUC_1 and TPR@1% differences (--n-boot 2000 resamples of prompts, each
carrying its w=1 and w=20 image; seed --seed). Strict JSON, no torch.

Produces: the fresh-null rows of Table VI (matched-quantile thresholds) and the paired
Pair-LDA-minus-Pair-zsum TPR@1% interval of Section IV-D.
Inputs: outputs/fresh_batch/{frozen_<dec>.json, <dec>_rows_shard*.jsonl}, plus
main-grid p0_table.json + the server-side feature store for the heads; CPU.
Outputs: outputs/p0/fresh_tail_compare.json.
Usage:
  uv run --no-sync python scripts/analysis/fresh_tail_compare.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from _fresh_batch_support import load_frozen_path, read_jsonl, validate_run_row
from _readout_common import (
    fit_deployed,
    load_residual_rows,
    write_json_strict,
)

from cfgid.detector.empirical import auc
from cfgid.paths import outputs_dir

CONTRAST = ({20.0}, {1.0})
DECODER_CELLS = {"sdxl": "sdxl", "pixart_sigma": "pixart_sigma"}
READOUTS = ("lda_dep", "zsum_dep")


def _load_fresh_rows(fresh_dir: Path, decoder: str, frozen: dict) -> list[dict]:
    """Load exactly the shard/row set committed by the validated frozen plan."""
    plan = frozen["plan"]
    n_shards = int(plan["n_shards"])
    if n_shards > 1:
        paths_by_shard = [
            fresh_dir / f"{decoder}_rows_shard{shard}of{n_shards}.jsonl"
            for shard in range(n_shards)
        ]
    else:
        paths_by_shard = [fresh_dir / f"{decoder}_rows.jsonl"]
    expected_paths = set(paths_by_shard)
    actual_paths = set(fresh_dir.glob(f"{decoder}_rows*.jsonl"))
    if actual_paths != expected_paths:
        missing = sorted(expected_paths - actual_paths)
        extra = sorted(actual_paths - expected_paths)
        raise SystemExit(
            f"[abort] {decoder}: fresh shard set disagrees with frozen plan; "
            f"missing={missing}, extra={extra}"
        )

    by_id: dict[str, dict] = {}
    for shard, path in enumerate(paths_by_shard):
        for row in read_jsonl(path, recover_trailing_partial=False):
            try:
                image_id = validate_run_row(row, decoder, frozen, shard, n_shards)
            except ValueError as exc:
                raise SystemExit(f"[abort] {path.name}: {exc}") from exc
            if image_id in by_id:
                raise SystemExit(f"[abort] duplicate fresh image_id {image_id}")
            by_id[image_id] = row

    expected_ids = {
        f"{decoder}_fp{int(plan['prompt_offset']) + i}_w{int(w)}"
        for i in range(int(plan["n_prompts"]))
        for w in plan["w_list"]
    }
    if set(by_id) != expected_ids:
        raise SystemExit(
            f"[abort] {decoder}: incomplete fresh row set; "
            f"missing={len(expected_ids - set(by_id))}, extra={len(set(by_id) - expected_ids)}"
        )
    return list(by_id.values())


def _score(rows: list[dict], m0: np.ndarray, s0: np.ndarray, wvec: np.ndarray) -> dict:
    """Score fresh (rho, m) rows with a frozen head: ((y - m0) / s0) @ wvec, split by class."""
    y = np.array([(r["rho"], r["m"]) for r in rows], dtype=np.float64)
    z = (y - m0) / s0
    sc = z @ wvec
    if not np.all(np.isfinite(sc)):
        raise SystemExit("[abort] non-finite fresh scores")
    w = np.array([r["w"] for r in rows])
    prompts = np.array([r["prompt_idx"] for r in rows])
    h0 = w == 1.0
    h1 = w != 1.0
    return {
        "h0": sc[h0],
        "h1": sc[h1],
        "prompt_h0": prompts[h0],
        "prompt_h1": prompts[h1],
    }


def _prompt_lookup_indices(class_prompts: np.ndarray, prompt_ids: np.ndarray) -> np.ndarray:
    """Row indices aligned to canonical prompt ids; fresh data has exactly one row/prompt/side."""
    if class_prompts.size != prompt_ids.size:
        raise ValueError("fresh class does not contain exactly one row per prompt")
    order = np.argsort(class_prompts)
    if not np.array_equal(class_prompts[order], prompt_ids):
        raise ValueError("fresh class prompt ids do not match the paired prompt set")
    return order.astype(np.intp, copy=False)


def _tpr_at_alpha(h0: np.ndarray, h1: np.ndarray, alpha: float) -> tuple[float, float]:
    """Matched-quantile threshold on fresh H0: (threshold, TPR)."""
    thr = float(np.quantile(h0, 1.0 - alpha))
    return thr, float(np.mean(h1 > thr))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--root", default=None, help="main-grid store root (def outputs/p0)")
    ap.add_argument("--fresh-dir", default=None, help="fresh batch dir (def outputs/fresh_batch)")
    ap.add_argument("--head-tol", type=float, default=1e-6)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if not math.isfinite(args.head_tol) or args.head_tol < 0.0:
        raise SystemExit("--head-tol must be finite and non-negative")
    if args.n_boot < 1:
        raise SystemExit("--n-boot must be positive")

    root = Path(args.root) if args.root else outputs_dir() / "p0"
    fresh_dir = Path(args.fresh_dir) if args.fresh_dir else outputs_dir() / "fresh_batch"
    h1_set, h0_set = CONTRAST

    rows = load_residual_rows(root)
    cells: dict[tuple[str, float], list[dict]] = {}
    for r in rows:
        cells.setdefault((r["decoder"], r["sigma_blur"]), []).append(r)

    results: dict[str, dict] = {}
    flags: list[str] = []
    for decoder in sorted(DECODER_CELLS):
        frozen_path = fresh_dir / f"frozen_{decoder}.json"
        frozen = load_frozen_path(frozen_path, decoder)
        cell_rows = cells.get((decoder, 0.0))
        if not cell_rows:
            raise SystemExit(f"[abort] store cell {decoder}/sb=0 missing under {root}")
        heads_raw = {ro: fit_deployed(cell_rows, h1_set, h0_set, ro) for ro in READOUTS}
        heads = {ro: h for ro, h in heads_raw.items() if h is not None}
        if len(heads) != len(READOUTS):
            raise SystemExit(f"[abort] deployed fit failed for {decoder}")
        for name in ("m0", "s0"):
            if not np.array_equal(
                getattr(heads["lda_dep"], name), getattr(heads["zsum_dep"], name)
            ):
                raise SystemExit(
                    f"[abort] {decoder}: zsum and LDA do not share the same FIT {name} state"
                )
        # Gate: store-refit LDA head must reproduce the frozen head used at generation time.
        fw = frozen["weights"]
        for name, got, want in (
            ("m0", heads["lda_dep"].m0, np.array(fw["m0"], float)),
            ("s0", heads["lda_dep"].s0, np.array(fw["s0"], float)),
            ("wvec", heads["lda_dep"].wvec, np.array(fw["wvec"], float)),
        ):
            dev = float(np.max(np.abs(got - want)))
            if dev > args.head_tol:
                raise SystemExit(
                    f"[abort] {decoder}: store-refit lda head '{name}' deviates from frozen "
                    f"by {dev:.3g} (> {args.head_tol:g}); FIT state drifted -- refusing numbers"
                )
        fresh = _load_fresh_rows(fresh_dir, decoder, frozen)
        per_readout: dict[str, dict] = {}
        packs = {}
        for ro in READOUTS:
            head = heads[ro]
            pack = _score(fresh, head.m0, head.s0, head.wvec)
            packs[ro] = pack
            entry: dict = {
                "auc1": float(
                    auc(
                        np.concatenate([pack["h1"], pack["h0"]]),
                        np.concatenate(
                            [np.ones(pack["h1"].size, int), np.zeros(pack["h0"].size, int)]
                        ),
                    )
                )
            }
            for alpha in (0.05, 0.01):
                thr, tpr = _tpr_at_alpha(pack["h0"], pack["h1"], alpha)
                entry[f"matched_tpr@{alpha:g}"] = tpr
                entry[f"matched_thr@{alpha:g}"] = thr
            if ro == "lda_dep":
                for alpha, thr in frozen["thresholds"].items():
                    entry[f"frozen_fpr@{alpha}"] = float(np.mean(pack["h0"] > float(thr)))
                    entry[f"frozen_tpr@{alpha}"] = float(np.mean(pack["h1"] > float(thr)))
            else:
                for alpha in frozen["thresholds"]:
                    entry[f"frozen_fpr@{alpha}"] = None
                    entry[f"frozen_tpr@{alpha}"] = None
            per_readout[ro] = entry
        # Paired prompt-level bootstrap of LDA - zsum differences. Per-class prompt indices follow
        # the exact score split order; repeated draws duplicate the whole prompt block.
        rng = np.random.default_rng(args.seed)
        per_class_prompts = {side: packs["lda_dep"][f"prompt_{side}"] for side in ("h0", "h1")}
        p_ids = np.unique(per_class_prompts["h0"])
        if not np.array_equal(p_ids, np.unique(per_class_prompts["h1"])):
            raise SystemExit(f"[abort] {decoder}: fresh H0/H1 prompt sets are not paired")
        try:
            prompt_lookup = {
                side: _prompt_lookup_indices(per_class_prompts[side], p_ids)
                for side in ("h0", "h1")
            }
        except ValueError as exc:
            raise SystemExit(f"[abort] {decoder}: {exc}") from exc
        diffs_auc, diffs_tpr = [], []
        for _ in range(args.n_boot):
            pick = rng.integers(0, p_ids.size, size=p_ids.size)
            i0 = prompt_lookup["h0"][pick]
            i1 = prompt_lookup["h1"][pick]
            vals = {}
            for ro in READOUTS:
                h0b, h1b = packs[ro]["h0"][i0], packs[ro]["h1"][i1]
                a = float(
                    auc(
                        np.concatenate([h1b, h0b]),
                        np.concatenate([np.ones(h1b.size, int), np.zeros(h0b.size, int)]),
                    )
                )
                _, tpr = _tpr_at_alpha(h0b, h1b, 0.01)
                vals[ro] = (a, tpr)
            diffs_auc.append(vals["lda_dep"][0] - vals["zsum_dep"][0])
            diffs_tpr.append(vals["lda_dep"][1] - vals["zsum_dep"][1])

        def ci(v: list[float]):
            return [float(x) for x in np.percentile(v, [2.5, 97.5])] if v else None

        results[decoder] = {
            "n_h0": int(packs["lda_dep"]["h0"].size),
            "n_h1": int(packs["lda_dep"]["h1"].size),
            "per_readout": per_readout,
            "paired_diff_lda_minus_zsum": {
                "auc1_ci95": ci(diffs_auc),
                "tpr@0.01_ci95": ci(diffs_tpr),
                "n_boot_effective": len(diffs_auc),
            },
        }

    payload = {
        "analysis": "fresh_tail_compare",
        "protocol": "Fresh-batch tail comparison: main-grid frozen deployed heads (lda_dep "
        "verified against frozen_<dec>.json; zsum_dep from the same FIT state) scored on the "
        "independent preregistered fresh batches; matched-quantile fresh-H0 thresholds; paired "
        "prompt bootstrap for readout differences",
        "n_boot": args.n_boot,
        "seed": args.seed,
        "flags": flags,
        "results": results,
    }
    out = Path(args.out) if args.out else outputs_dir() / "p0" / "fresh_tail_compare.json"
    write_json_strict(out, payload)
    print(f"[fresh_tail_compare] wrote {out}")
    for dec, res in results.items():
        for ro, e in res["per_readout"].items():
            print(
                f"  {dec:>12s} {ro:>8s}  auc1={e['auc1']:.4f}  "
                f"tpr@1%={e['matched_tpr@0.01']:.3f}  tpr@5%={e['matched_tpr@0.05']:.3f}"
            )
        print(f"  {dec:>12s} paired LDA-zsum: {res['paired_diff_lda_minus_zsum']}")
    print("[done] fresh_tail_compare")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
