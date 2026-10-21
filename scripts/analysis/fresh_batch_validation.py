"""Preregistered fresh-batch validation of the deployed pair readout (frozen heads, no refitting).

Three stages, each run with the same environment:

  ``predict``    freezes the deployed Pair-LDA head, the FIT-null thresholds and the Gaussian
                 AUC_K predictions from the main-grid FIT split, bound by SHA-256 (``frozen_id``);
                 see _fresh_batch_predict.py.
  ``run``        generates the w=1 and w=20 image of every planned prompt (seed = seed_base + i),
                 PNG round-trips it, runs the dual-branch inversion, and appends (rho, m) rows to
                 strict per-shard JSONL; every row records ``frozen_id``; resumable.
  ``summarize``  requires the complete frozen shard set, scores rows with the frozen head only,
                 and reports AUC_1, realized same-cell AUC_K on non-overlapping groups (mean over
                 N_PART random partitions, seed 0), and FPR (with one-sided CP95) / TPR at the
                 frozen FIT-null thresholds.

Produces: Fig. 4 (frozen predictions vs realized AUC_K), the
FIT-null rows of Table VI and the frozen S_eff values of Section IV-D, on the first fresh batch
(SDXL and PixArt-Sigma, 2,048 images per class per decoder). The ``predict`` and ``run`` stages
also generate the second (prospective-calibration) batch of Table VII when run with the batch-2
environment after freeze_prospective_cal.py.
Inputs: predict -- main-grid p0_table.json + server-only feature store (CPU); run -- GPU and
model weights; summarize -- only the frozen_<dec>.json and row JSONL files (CPU).
Outputs: outputs/fresh_batch/{frozen_<dec>.json, <dec>_rows_shard<i>of<n>.jsonl,
<dec>_summary.json}, or the same names under $CFGID_FRESH_DIR.

Usage (single job: omit --n-shards; sharded campaign: --n-shards 2)::

  uv run --no-sync python scripts/analysis/fresh_batch_validation.py --stage predict \\
      --decoders sdxl,pixart_sigma --n-prompts 2048 --n-shards 2
  uv run --no-sync python scripts/analysis/fresh_batch_validation.py --stage run \\
      --decoder sdxl --n-prompts 2048 --n-shards 2 --shard 0
  uv run --no-sync python scripts/analysis/fresh_batch_validation.py --stage summarize \\
      --decoder sdxl
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from _fresh_batch_predict import stage_predict
from _fresh_batch_support import (
    ALPHAS,
    GROUP_K,
    MASK_Q,
    WINSOR,
    cp_upper,
    finite_vector,
    fresh_dir,
    load_frozen,
    m_masked_winsor,
    png_roundtrip,
    read_jsonl,
    rows_path,
    stored_residual_roundtrip,
    validate_run_row,
    write_json,
)

from cfgid.channel.features import build_inversion_result
from cfgid.channel.inversion.registry import get_backend
from cfgid.channel.rho import image_rho_summary
from cfgid.config import load_models_config
from cfgid.detector.empirical import auc
from cfgid.device import get_device
from cfgid.numerics import finite_float
from cfgid.paths import outputs_dir
from cfgid.sweep.run_p0 import _placeholder_prompts

N_PART = 200  # random non-overlapping partitions averaged per K in summarize()


def _two_sample_auc(h0: np.ndarray, h1: np.ndarray) -> float:
    h0 = np.asarray(h0, dtype=np.float64).reshape(-1)
    h1 = np.asarray(h1, dtype=np.float64).reshape(-1)
    if h0.size == 0 or h1.size == 0 or not np.all(np.isfinite(h0)) or not np.all(np.isfinite(h1)):
        raise ValueError("two-sample AUC requires non-empty finite H0/H1 arrays")
    scores = np.concatenate([h0, h1])
    labels = np.concatenate([np.zeros(len(h0)), np.ones(len(h1))])
    value = finite_float("two-sample AUC", auc(scores, labels))
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"two-sample AUC must be in [0,1], got {value}")
    return value


def stage_run(decoder: str, n_prompts: int, steps: int, shard: int, n_shards: int) -> int:
    frozen = load_frozen(decoder)
    plan = frozen["plan"]
    frozen_n = int(plan["n_prompts"])
    frozen_shards = int(plan["n_shards"])
    frozen_steps = int(plan["steps"])
    if n_prompts != frozen_n:
        raise SystemExit(f"--n-prompts={n_prompts} != frozen plan {frozen_n}")
    if n_shards != frozen_shards:
        raise SystemExit(f"--n-shards={n_shards} != frozen plan {frozen_shards}")
    if steps != frozen_steps:
        raise SystemExit(f"--steps={steps} != frozen plan {frozen_steps}")
    if not 0 <= shard < n_shards:
        raise SystemExit(f"--shard must be in [0, {n_shards})")

    jsonl = rows_path(decoder, shard, n_shards)
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    previous: dict[str, dict] = {}
    for row in read_jsonl(jsonl, recover_trailing_partial=True):
        image_id = validate_run_row(row, decoder, frozen, shard, n_shards)
        if image_id in previous and previous[image_id] != row:
            raise SystemExit(f"{jsonl}: conflicting duplicate image_id {image_id}")
        previous[image_id] = row
    done = set(previous)
    if done:
        print(f"[run] resuming: {len(done)} rows already in {jsonl}")

    offset = int(plan["prompt_offset"])
    expected_ids = {
        f"{decoder}_fp{offset + i}_w{int(w)}"
        for i in range(frozen_n)
        if i % n_shards == shard
        for w in plan["w_list"]
    }
    if done == expected_ids:
        print(f"[run] {decoder} shard {shard}/{n_shards}: already complete; no model load")
        if n_shards > 1:
            return 0
        return summarize(decoder)
    if not done < expected_ids:
        raise SystemExit(f"{jsonl}: resumed image_id set is not a subset of the frozen shard")

    # Fresh prompts continue the deterministic main-grid prompt bank from the frozen offset;
    # shard s takes the local prompt indices i with i % n_shards == s.
    prompts = _placeholder_prompts(offset + frozen_n)[offset:]
    prompt_items = [(i, prompt) for i, prompt in enumerate(prompts) if i % n_shards == shard]
    print(f"[run] {decoder}: shard {shard}/{n_shards} -> {len(prompt_items)} prompts")

    specs = load_models_config()
    if decoder not in specs:
        raise SystemExit(f"unknown decoder {decoder!r}")
    if specs[decoder].branch_mode != "true_cfg_core":
        raise SystemExit(f"{decoder}: fresh validation requires a true_cfg_core backend")
    backend = get_backend(specs[decoder], get_device())
    backend.load()
    channel = frozen["channel"]
    n_new = 0
    try:
        with jsonl.open("a", encoding="utf-8") as fh:
            for i, prompt in prompt_items:
                for w_raw in plan["w_list"]:
                    w = float(w_raw)
                    image_id = f"{decoder}_fp{offset + i}_w{int(w)}"
                    if image_id in done:
                        continue
                    # One seed per prompt, shared by its w=1 and w=20 images.
                    seed = int(plan["seed_base"]) + i
                    image = backend.generate_image(
                        prompt,
                        w=w,
                        seed=seed,
                        num_steps=frozen_steps,
                        height=int(channel["height"]),
                        width=int(channel["width"]),
                    )
                    # Invert the 8-bit PNG representation, as for stored main-grid images.
                    dual = backend.dual_branch_residual(
                        png_roundtrip(image),
                        prompt,
                        num_steps=frozen_steps,
                        seed=seed,
                        capture_step_frac=float(channel["capture_frac"]),
                    )
                    result = build_inversion_result(
                        dual,
                        patch_size=int(channel["patch"]),
                        floor=float(channel["floor"]),
                        log_ratio=bool(channel["log_ratio"]),
                        sigma=0.0,
                        e_tex_method=str(channel["e_tex_method"]),
                    )
                    rho = image_rho_summary(
                        result.rho_patch,
                        result.e_tex_patch,
                        mask_quantile=MASK_Q,
                        winsor_quantiles=WINSOR,
                    )
                    # m uses the same fp16-scaled residual representation as the store.
                    stored_residual = stored_residual_roundtrip(result.residual)
                    m = m_masked_winsor(np.linalg.norm(stored_residual, axis=1), result.e_tex_patch)
                    row = {
                        "frozen_id": frozen["frozen_id"],
                        "image_id": image_id,
                        "prompt_idx": offset + i,
                        "w": w,
                        "seed": seed,
                        "rho": finite_float("rho", rho),
                        "m": finite_float("m", m),
                    }
                    fh.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
                    done.add(image_id)
                    n_new += 1
                if n_new and n_new % 128 == 0:
                    print(f"[run] {decoder} shard {shard}: {n_new} new rows written")
        if done != expected_ids:
            missing_count = len(expected_ids - done)
            extra_count = len(done - expected_ids)
            raise RuntimeError(
                f"shard ended incomplete: missing={missing_count} extra={extra_count}"
            )
    finally:
        backend.unload()
    print(f"[run] {decoder} shard {shard}/{n_shards}: complete ({n_new} new rows)")
    if n_shards > 1:
        print("[run] run --stage summarize once after ALL shards finish")
        return 0
    return summarize(decoder)


def _complete_rows(decoder: str, frozen: dict) -> list[dict]:
    plan = frozen["plan"]
    n_shards = int(plan["n_shards"])
    by_id: dict[str, dict] = {}
    for shard in range(n_shards):
        jsonl = rows_path(decoder, shard, n_shards)
        if not jsonl.is_file():
            raise SystemExit(f"missing shard {jsonl}; summarize only after all shards run")
        for row in read_jsonl(jsonl, recover_trailing_partial=False):
            image_id = validate_run_row(row, decoder, frozen, shard, n_shards)
            if image_id in by_id and by_id[image_id] != row:
                raise SystemExit(f"conflicting duplicate image_id {image_id}")
            by_id[image_id] = row
    expected = {
        f"{decoder}_fp{int(plan['prompt_offset']) + i}_w{int(w)}"
        for i in range(int(plan["n_prompts"]))
        for w in plan["w_list"]
    }
    if set(by_id) != expected:
        raise SystemExit(
            f"incomplete shard set: missing={len(expected - set(by_id))} "
            f"extra={len(set(by_id) - expected)}"
        )
    return list(by_id.values())


def summarize(decoder: str) -> int:
    frozen = load_frozen(decoder)
    plan = frozen["plan"]
    rows = _complete_rows(decoder, frozen)
    m0 = finite_vector("weights.m0", frozen["weights"]["m0"], (2,))
    s0 = finite_vector("weights.s0", frozen["weights"]["s0"], (2,))
    wvec = finite_vector("weights.wvec", frozen["weights"]["wvec"], (2,))

    def score(w: float) -> np.ndarray:
        values = np.asarray([(r["rho"], r["m"]) for r in rows if r["w"] == w], dtype=np.float64)
        if values.shape != (int(plan["n_prompts"]), 2) or not np.all(np.isfinite(values)):
            raise ValueError(f"w={w:g} feature pool is invalid: {values.shape}")
        scores = ((values - m0) / s0) @ wvec
        if not np.all(np.isfinite(scores)):
            raise ValueError(f"w={w:g} frozen scores contain NaN/Inf")
        return scores

    t0 = score(float(frozen["contrast"]["h0_w"]))
    t1 = score(float(frozen["contrast"]["h1_w"]))
    # Realized same-cell AUC_K: within each class, a random permutation is cut into
    # floor(n / K) non-overlapping groups (remainder dropped); group means are compared and the
    # AUC is averaged over N_PART partitions drawn from one generator (seed 0).
    rng = np.random.default_rng(0)
    realized: dict[str, float | None] = {}
    for k in GROUP_K:
        groups = min(len(t0) // k, len(t1) // k)
        if groups < 2:
            realized[str(k)] = None
            continue
        aucs = []
        for _ in range(N_PART):
            p0 = rng.permutation(len(t0))[: groups * k].reshape(groups, k)
            p1 = rng.permutation(len(t1))[: groups * k].reshape(groups, k)
            aucs.append(_two_sample_auc(t0[p0].mean(1), t1[p1].mean(1)))
        realized[str(k)] = finite_float(f"realized_auc_k[{k}]", np.mean(aucs))

    # Operating points at the FIT-null thresholds frozen before generation.
    ops = {}
    for alpha in ALPHAS:
        threshold = finite_float(f"threshold[{alpha}]", frozen["thresholds"][str(alpha)])
        false_positives = int(np.sum(t0 > threshold))
        ops[str(alpha)] = {
            "fpr": false_positives / len(t0),
            "fpr_cp95_upper": cp_upper(false_positives, len(t0)),
            "tpr": finite_float("tpr", np.mean(t1 > threshold)),
            "n0": int(len(t0)),
        }
    summary = {
        "frozen_id": frozen["frozen_id"],
        "decoder": decoder,
        "n_h0": int(len(t0)),
        "n_h1": int(len(t1)),
        "auc1_realized": _two_sample_auc(t0, t1),
        "predicted_auc_k": frozen["predicted_auc_k"],
        "realized_auc_k_nonoverlap": realized,
        "operating_points_frozen_thresholds": ops,
        "n_groups_k16": int(len(t0) // 16),
    }
    out = fresh_dir() / f"{decoder}_summary.json"
    write_json(out, summary)
    print(f"\n=== fresh-batch validation: {decoder} (frozen head, no refitting) ===")
    print(
        f"  n/class: {len(t0)}/{len(t1)}  AUC1={summary['auc1_realized']:.3f}  "
        f"K=16 independent groups/class: {summary['n_groups_k16']}"
    )
    print("  K      predicted   realized(non-overlap)")
    for k in GROUP_K:
        pred = summary["predicted_auc_k"][str(k)]
        real = summary["realized_auc_k_nonoverlap"][str(k)]
        real_s = f"{real:.3f}" if real is not None else "  -  "
        print(f"  {k:>2}    {pred:.3f}       {real_s}")
    for a_key, o in summary["operating_points_frozen_thresholds"].items():
        print(
            f"  alpha={a_key}: FPR {o['fpr']:.4f} (CP95 upper {o['fpr_cp95_upper']:.4f}), "
            f"TPR {o['tpr']:.3f}, n0={o['n0']}"
        )
    print(f"[fresh_batch] wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["predict", "run", "summarize"], required=True)
    parser.add_argument("--decoders", default="sdxl,pixart_sigma")
    parser.add_argument("--decoder", default=None)
    parser.add_argument("--n-prompts", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--root", default=None)
    args = parser.parse_args(argv)
    root = Path(args.root) if args.root else outputs_dir() / "p0"
    if args.stage == "predict":
        decoders = [item.strip() for item in args.decoders.split(",") if item.strip()]
        return stage_predict(decoders, root, args.n_prompts, args.n_shards)
    if args.decoder is None:
        raise SystemExit("--decoder is required for run/summarize")
    if args.stage == "run":
        return stage_run(args.decoder, args.n_prompts, args.steps, args.shard, args.n_shards)
    return summarize(args.decoder)


if __name__ == "__main__":
    raise SystemExit(main())
