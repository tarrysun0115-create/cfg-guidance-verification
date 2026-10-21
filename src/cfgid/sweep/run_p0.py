"""Main-grid (p0) driver: paired generation -> degradation -> dual-branch inversion -> p0 table.

For one decoder this generates H0 (w=1) and H1 (w>1) images from a deterministic prompt bank,
applies the channel (optional JPEG QF -> Gaussian post-decode blur sigma_blur -> optional
post-processing op), stores the observed 8-bit PNG, inverts it with the conditional and the
null-text branch (Section III-B, Eq. (1)) and reduces the captured residuals to the per-image
coordinates rho (Eqs. (2)-(4)) and m (Eqs. (6)-(8)). H0 is generated once per prompt and shares
the generation seed with every H1 of that prompt, so H0/H1 differ only in guidance.

Prompt bank (Section IV-A): prompt i = "<stem[i % 8]>, <context[(i // 8) % 4]>" plus
", variant <i // 32 + 1>" for i >= 32 (``_PROMPT_SEEDS`` x the four context clauses below).
Generation seed: ``prompt_idx * 100003 + seed_idx`` (``_effective_seed``). Splits: prompt ids
"p<i>" over the FULL prompt set are assigned to train / null_adapt / calibration / test
(FIT / NULL / CAL / TEST in the paper) by ``cfgid.sweep.grid.assign_splits`` with the
configs/p0.yaml fractions and seed, so a sharded job reproduces the un-sharded split.

Paper grid: configs/p0.yaml keeps ``grid.prompts_per_cell: 128`` as its default; the main grid of
the paper was run per decoder with command-line overrides of that file, ``--prompts-per-cell 512``
for SD1.5 and SDXL and ``--prompts-per-cell 256`` for SD2.1, PixArt-Sigma and HunyuanDiT, each with
``--sigma-blur 0,8,16,32`` (see README.md). With
``--config configs/robustness.yaml --lean`` the same driver writes the post-processing pool
behind Table XII(a) (Section IV-E6).

Produces: the main-grid per-image tables (Section IV-A, Appendix B) that every main-grid analysis
script reads.
Inputs: frozen model weights (configs/models.yaml under the checkpoints root of
configs/paths.yaml); a GPU for real runs. Weight-gated: without local weights it prints a skip
notice and returns 0. ``--smoke`` runs 1 prompt / 1 step / 1 w / uncompressed (a single
end-to-end pass).
Outputs: <out>/<decoder>/p0_table.json, images/*.png, traces/*.npz (backends with an oracle-trace
hook), and the feature store (metadata.json, compact_*.npy, residual_v1.dat,
residual_v1_scale.npy); ``--lean`` writes p0_table.json only. Default <out> is outputs/p0
(outputs/p0_smoke with --smoke).
Usage:
  scripts/run_p0 --decoder sdxl --smoke
  scripts/run_p0 --decoder sd15 --prompts-per-cell 512 --sigma-blur 0,8,16,32
  scripts/run_p0 --decoder sdxl --config configs/robustness.yaml --lean --prompt-shard 0/2
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np

from cfgid.channel.features import build_inversion_result
from cfgid.channel.generate import jpeg_compress
from cfgid.channel.inversion.base import (
    BackendUnavailableError,
    InversionResult,
    pack_results,
)
from cfgid.channel.inversion.registry import get_backend
from cfgid.channel.postproc import center_crop_zoom, resize_roundtrip
from cfgid.channel.psf_blur import gaussian_psf_blur
from cfgid.channel.rho import image_m_summary, image_rho_summary
from cfgid.channel.sigma_probe import nominal_sigma
from cfgid.config import load_models_config
from cfgid.device import get_device
from cfgid.featurestore import read_compact, read_residual_v1, write_compact, write_residual_v1
from cfgid.numerics import loads_strict_json
from cfgid.p0config import P0Config, PostprocOp, _sigma_blur_levels, load_p0_config
from cfgid.paths import ensure_dir, outputs_dir
from cfgid.sweep.grid import assign_splits

_SMOKE_PROMPTS = ["a photograph of a tabby cat on a sofa"]
_PROMPT_SEEDS = [
    "a photograph of a tabby cat on a sofa",
    "a wooden sailboat on a calm lake at sunrise",
    "a plate of fresh strawberries on a marble table",
    "a city street with neon signs at night",
    "a ceramic vase beside a window with afternoon light",
    "a mountain trail covered with fresh snow",
    "a vintage camera on a cluttered desk",
    "a red bicycle leaning against a brick wall",
]
_REQUIRED_DECODER_OUTPUTS = (
    "metadata.json",
    "compact_rho.npy",
    "compact_e_tex.npy",
    "compact_sigma.npy",
    "compact_qf.npy",
    "residual_v1.dat",
    "residual_v1_scale.npy",
    "p0_table.json",
)
_REQUIRED_TABLE_FIELDS = {
    "decoder",
    "prompt",
    "prompt_text",
    "seed",
    "split",
    "w",
    "qf",
    "sigma_blur",
    "label",
    "rho_summary",
    "image_id",
    "image",
    "gen_seed",
}


def _placeholder_prompts(count: int) -> list[str]:
    """Deterministic English prompt bank: 8 scene stems x 4 context clauses, then numbered variants.

    Prompt i is "<stem[i % 8]>, <context[(i // 8) % 4]>", with ", variant <i // 32 + 1>" appended
    once i >= 32. The bank is nested (the first n prompts are the same for every count), and the
    fresh batches reuse it from a prompt offset.
    """
    if count <= 0:
        raise ValueError(f"prompts_per_cell must be positive, got {count}")
    contexts = (
        "under soft window light",
        "with shallow depth of field",
        "on a neutral background",
        "with detailed natural texture",
    )
    prompts: list[str] = []
    cycle_size = len(_PROMPT_SEEDS) * len(contexts)
    for i in range(count):
        base = _PROMPT_SEEDS[i % len(_PROMPT_SEEDS)]
        context = contexts[(i // len(_PROMPT_SEEDS)) % len(contexts)]
        cycle = i // cycle_size
        suffix = "" if cycle == 0 else f", variant {cycle + 1}"
        prompts.append(f"{base}, {context}{suffix}")
    return prompts


def _expected_result_count(
    cfg: P0Config,
    *,
    prompts_per_cell: int | None = None,
    sigma_blurs: list[float] | None = None,
) -> int:
    grid = cfg.grid
    ppc = grid.prompts_per_cell if prompts_per_cell is None else prompts_per_cell
    sbs = list(grid.sigma_blur) if sigma_blurs is None else sigma_blurs
    h1_w_count = len([w for w in grid.w if w > 1])
    channel_variants = len(grid.qf) * len(sbs) * len(grid.postproc)  # qf x sigma_blur x postproc
    return ppc * grid.seeds_per_prompt * channel_variants * (1 + h1_w_count)


def _require_supported_gen_scheduler(name: str) -> None:
    if name.lower() != "ddim":
        raise ValueError(f"run_p0 supports grid.gen_scheduler=ddim only, got '{name}'")


def _default_output_root(*, smoke: bool, out_arg: str | None) -> str:
    """Default smoke artifacts away from the real main-grid feature root (avoid clobbering runs)."""
    if out_arg is not None:
        return out_arg
    name = "p0_smoke" if smoke else "p0"
    return str(outputs_dir() / name)


def _save_png(image: np.ndarray, path: Path) -> np.ndarray:
    """Persist an 8-bit PNG and return the exact float image represented by those bytes."""
    from PIL import Image

    arr = np.asarray(image, dtype=np.float32)
    # Fail loud on a non-finite pixel: np.clip(np.rint(NaN)) would silently coerce NaN->0 / Inf->255
    # and persist a corrupt image (no silent NaN/Inf). The inversion/generation guards
    # should have raised already; this is the last line of defence before bytes hit disk.
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"_save_png received a non-finite pixel for {path}")
    u8 = np.clip(np.rint(arr * 255.0), 0.0, 255.0).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(u8).save(path, format="PNG")
    return u8.astype(np.float32) / 255.0


def _save_trace(trace: dict[str, np.ndarray | int | float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for key, value in trace.items():
        arr = np.asarray(value)
        try:
            finite = bool(np.all(np.isfinite(arr)))
        except TypeError as exc:
            raise ValueError(f"trace field '{key}' must be numeric and finite") from exc
        if not finite:
            raise ValueError(f"trace field '{key}' contains non-finite values (NaN/Inf)")
        arrays[key] = arr
    with path.open("wb") as fh:
        # cast: numpy's savez stub mis-types dynamic **kwargs against allow_pickle.
        cast(Any, np.savez)(fh, **arrays)


def _apply_postproc(image: np.ndarray, op: PostprocOp) -> np.ndarray:
    """Apply one off-gate post-processing degradation to a decoded image (size-preserving).

    'none' is identity; 'jpeg' re-encodes at the given quality; 'resize' downscales to ``param`` of
    each side then upscales back to (H, W); 'crop' center-crops to ``param`` then zooms back. resize
    and crop are size-preserving, so forensic_observation == inversion_input holds: the analyst
    observes a degraded FULL-resolution image and inverts it as-is, exactly like the JPEG/blur axes.
    """
    if op.kind == "none":
        return image
    if op.kind == "jpeg":
        return jpeg_compress(image, int(op.param))[0]
    if op.kind == "resize":
        return resize_roundtrip(image, op.param)
    if op.kind == "crop":
        return center_crop_zoom(image, op.param)
    raise ValueError(f"unknown postproc kind {op.kind!r}")


def _shard_prompt_indices(n_prompts: int, shard: tuple[int, int] | None) -> set[int]:
    """Contiguous prompt-index slice for one shard ``(k, total)`` of ``n_prompts`` prompts.

    ``None`` -> all indices. The splits + image_ids are keyed by the GLOBAL prompt index, so each
    shard processes only its own indices while the un-sharded analysis pools every shard's rows back
    into one table. Shard ``k`` gets ``[k*ceil(n/total) : (k+1)*ceil(n/total)]`` (last shard short).
    """
    if shard is None:
        return set(range(n_prompts))
    k, total = shard
    if not (isinstance(k, int) and isinstance(total, int)) or total < 1 or not (0 <= k < total):
        raise ValueError(f"prompt_shard must be (k, total) with 0 <= k < total >= 1, got {shard}")
    per = -(-n_prompts // total)  # ceil
    return set(range(k * per, min((k + 1) * per, n_prompts)))


def _decoder_output_complete(out: Path, *, expected_n_images: int | None = None) -> bool:
    if not all((out / name).is_file() for name in _REQUIRED_DECODER_OUTPUTS):
        return False
    try:
        batch, meta = read_compact(out)
        residual, residual_meta = read_residual_v1(out)
        table_raw = loads_strict_json((out / "p0_table.json").read_text(encoding="utf-8"))
    except (
        EOFError,
        FileNotFoundError,
        json.JSONDecodeError,
        KeyError,
        OSError,
        TypeError,
        UnicodeDecodeError,
        ValueError,
    ):
        return False
    if not isinstance(table_raw, list) or not table_raw:
        return False
    if any(
        not isinstance(row, dict) or not _REQUIRED_TABLE_FIELDS <= set(row) for row in table_raw
    ):
        return False
    table_rows = cast(list[dict[str, object]], table_raw)
    store_ids = list(getattr(meta, "image_ids", []) or [])
    if store_ids:
        table_ids = [str(row.get("image_id", "")) for row in table_rows]
        if table_ids != store_ids:
            return False
    # Persistence check: every row's stored inversion-input image must exist on disk.
    if any(not (out / str(row.get("image", ""))).is_file() for row in table_rows):
        return False
    # Oracle traces: a decoder that captured traces (e.g. pixart writes one per generation) is only
    # complete if every referenced .npz is still on disk -- a partial/deleted trace must trigger a
    # re-run, not a [skip], so the downstream oracle-trace cross-check never reads a missing file.
    if any(row.get("trace") and not (out / str(row["trace"])).is_file() for row in table_rows):
        return False
    n_images = meta.n_images
    if expected_n_images is not None and n_images != expected_n_images:
        return False
    return (
        residual_meta.n_images == n_images
        and batch.rho.shape[0] == n_images
        and residual.shape[0] == n_images
        and len(table_rows) == n_images
    )


def _effective_seed(prompt_idx: int, seed_idx: int) -> int:
    """Per-(prompt, seed-index) generation seed.

    A constant seed across prompts would give EVERY prompt the same initial latent noise,
    correlating all test-split samples and silently shrinking the effective sample size. The
    prime stride decorrelates prompts while keeping the H0/H1 pairing exact (the pair shares
    prompt_idx AND seed_idx, hence the same effective seed).
    """
    return prompt_idx * 100_003 + seed_idx


def run(
    decoder: str,
    *,
    smoke: bool,
    out_dir: str,
    prompts_per_cell: int | None = None,
    sigma_blur: list[float] | None = None,
    config_path: Path | None = None,
    lean: bool = False,
    prompt_shard: tuple[int, int] | None = None,
) -> int:
    # config_path None -> default p0.yaml (keeps the no-arg call so existing test stubs hold);
    # an off-gate robustness config is passed explicitly via --config.
    cfg = load_p0_config(config_path) if config_path is not None else load_p0_config()
    # Per-job grid-dimension overrides (for sbatch: a different N / sigma set per decoder). These
    # change the swept DIMENSIONS only -- the gate: block in p0.yaml is untouched. None -> config.
    if prompts_per_cell is not None and prompts_per_cell < 1:
        print(f"[error] --prompts-per-cell must be >= 1, got {prompts_per_cell}")
        return 2
    eff_ppc = cfg.grid.prompts_per_cell if prompts_per_cell is None else prompts_per_cell
    if sigma_blur is None:
        eff_sigma = list(cfg.grid.sigma_blur)
    else:
        # A CLI sigma override must get the SAME normalization the config path applies via
        # _sigma_blur_levels: dedup + sort + force-inject the 0.0 baseline + reject %g-unstable
        # levels. Skipping it would let a duplicate value silently overwrite a same-image_id row,
        # let a set omitting 0.0 disable the wrong-prompt baseline cell, and let a non-%g-stable
        # level diverge between the image_id %g suffix and the stored full-precision float.
        try:
            eff_sigma = list(_sigma_blur_levels(sigma_blur))
        except ValueError as exc:
            print(f"[error] --sigma-blur {exc}")
            return 2
        if eff_sigma != list(cfg.grid.sigma_blur):
            print(
                f"[notice] --sigma-blur normalized to {eff_sigma} differs from the preregistered "
                f"grid.sigma_blur {list(cfg.grid.sigma_blur)}; this run is OFF the frozen "
                "preregistered grid "
                "(its (decoder, sigma_blur) cells will not align with the analysis grid)."
            )
    try:
        _require_supported_gen_scheduler(cfg.grid.gen_scheduler)
    except ValueError as exc:
        print(f"[error] {exc}")
        return 2
    models = load_models_config()
    if decoder not in models:
        print(f"[error] unknown decoder '{decoder}' (have {sorted(models)})")
        return 2
    spec = models[decoder]
    # Hard rule: only true-CFG decoders (branch_mode true_cfg_core) enter the main grid. A
    # guidance-distilled decoder's rho is NOT homogeneous with the SD-family channel and must never
    # enter the sigma-w* core regression; FLUX is not used in the paper (Appendix B).
    if spec.branch_mode != "true_cfg_core":
        print(
            f"[error] decoder '{decoder}' has branch_mode '{spec.branch_mode}'; the main-grid "
            "driver accepts true_cfg_core decoders only (guidance-distilled decoders such as FLUX "
            "are not supported)."
        )
        return 2
    if not spec.is_present():
        print(
            f"[skip] weights for '{decoder}' not present at {spec.resolved_local_path()}; "
            "download per README.md, then re-run."
        )
        return 0

    out = ensure_dir(Path(out_dir) / decoder)
    # Decoder-level resumability: skip only a complete feature directory. A crash after compact
    # metadata but before residual/table write must not be mistaken for a finished decoder.
    # lean runs write only p0_table.json (no residual/compact store), so the residual-backed
    # completeness check cannot certify them -- they are single-shot (clear the dir to re-run).
    if (
        not smoke
        and not lean
        and _decoder_output_complete(
            out,
            expected_n_images=_expected_result_count(
                cfg, prompts_per_cell=eff_ppc, sigma_blurs=eff_sigma
            ),
        )
    ):
        print(f"[skip] {decoder}: features already present at {out}; delete the dir to re-run.")
        return 0

    backend = get_backend(spec, get_device())
    try:
        backend.load()
    except BackendUnavailableError as exc:
        print(f"[skip] {exc}")
        return 0

    steps = 1 if smoke else cfg.patch.inversion_steps  # T = 50 DDIM steps on the main grid
    ws = [w for w in cfg.grid.w if w > 1][: 1 if smoke else None]  # H1 levels; w=1 is H0
    qfs: list[int | None] = [None] if smoke else list(cfg.grid.qf)
    sigma_blurs: list[float] = [0.0] if smoke else eff_sigma
    # One-at-a-time off-gate degradation axis (robustness config); 'none' is the main-grid channel.
    postproc_ops: list[PostprocOp] = (
        [PostprocOp("none", "none", 0.0)] if smoke else list(cfg.grid.postproc)
    )
    prompts = _SMOKE_PROMPTS if smoke else _placeholder_prompts(eff_ppc)
    seeds = [0] if smoke else list(range(cfg.grid.seeds_per_prompt))
    sigma = nominal_sigma(decoder)
    # Leakage-proof split grouped by prompt (recorded per image; consumed by the analysis fits).
    # Built from the FULL prompt set so a sharded job's splits + image_ids match the un-sharded
    # pool.
    # Prompt ids are the strings "p0", "p1", ...; assign_splits sorts them as strings before its
    # seeded permutation, so the assignment depends on the prompt count of the run.
    prompt_ids = [f"p{i}" for i in range(len(prompts))]
    splits = assign_splits(prompt_ids, cfg.split.fractions, seed=cfg.split.seed)
    shard_indices = _shard_prompt_indices(len(prompts), prompt_shard)

    results: list[InversionResult] = []
    table: list[dict[str, object]] = []

    def _rho_summary_for(comp: np.ndarray, text: str, seed: int) -> float:
        dual = backend.dual_branch_residual(
            comp, text, num_steps=steps, seed=seed, capture_step_frac=cfg.patch.capture_step_frac
        )
        res = build_inversion_result(
            dual,
            patch_size=cfg.patch.patch_size,
            floor=cfg.rho.denom_floor,
            log_ratio=cfg.rho.log_ratio,
            sigma=sigma,
            e_tex_method=cfg.e_tex.method,
        )
        return image_rho_summary(
            res.rho_patch,
            res.e_tex_patch,
            mask_quantile=cfg.rho.low_etex_mask_quantile,
            winsor_quantiles=cfg.rho.winsor_quantiles,
        )

    def _process(
        image: np.ndarray,
        prompt: str,
        w_label: int,
        label: str,
        pi: int,
        seed: int,
        gen_seed: int,
        trace_rel: str | None,
    ) -> None:
        # Channel = decode -> [JPEG qf] -> [Gaussian blur sigma_blur] -> [postproc op] -> observe
        # -> invert. The main grid sweeps sigma_blur (qf=none, postproc=none). The postproc axis
        # adds a ONE-AT-A-TIME robustness degradation (jpeg/resize/crop); it is the single 'none'
        # op on the preregistered grid, so this collapses exactly to the main-grid channel there.
        for qf in qfs:
            comp = image if qf is None else jpeg_compress(image, qf)[0]
            for sigma_blur in sigma_blurs:
                blurred = gaussian_psf_blur(comp, sigma_blur)
                for op in postproc_ops:
                    processed = _apply_postproc(blurred, op)
                    pp_tag = "" if op.kind == "none" else f"_pp{op.name}"
                    image_id = (
                        f"{decoder}_p{pi}_s{seed}_w{w_label}_{label}"
                        f"_qf{qf}_sb{sigma_blur:g}{pp_tag}"
                    )
                    image_rel = f"images/{image_id}.png"
                    # The stored PNG is the forensic observation. Feed the same processed image into
                    # inversion so rho and the persisted image share one input (lean: deleted).
                    inversion_input = _save_png(processed, out / image_rel)
                    dual = backend.dual_branch_residual(
                        inversion_input,
                        prompt,
                        num_steps=steps,
                        seed=gen_seed,
                        capture_step_frac=cfg.patch.capture_step_frac,
                    )
                    res = build_inversion_result(
                        dual,
                        patch_size=cfg.patch.patch_size,
                        floor=cfg.rho.denom_floor,
                        log_ratio=cfg.rho.log_ratio,
                        # Metadata only: the gate uses sigma_calib.sigma_eff(sigma_blur), not this
                        # nominal composition. Keep compact_sigma aligned with the observed channel.
                        sigma=math.sqrt(sigma**2 + float(sigma_blur) ** 2),
                        e_tex_method=cfg.e_tex.method,
                        qf=float(qf) if qf is not None else float("nan"),
                        image_id=image_id,
                    )
                    # lean: only p0_table.json is kept, so the full-dim residual is not accumulated.
                    if not lean:
                        results.append(res)
                    row: dict[str, object] = {
                        "decoder": decoder,
                        "branch_mode": spec.branch_mode,
                        "prompt": pi,
                        "prompt_text": prompt,
                        "seed": seed,
                        "gen_seed": gen_seed,
                        "split": splits[f"p{pi}"],
                        "w": w_label,
                        "qf": qf if qf is not None else -1,
                        "sigma_blur": float(sigma_blur),
                        "postproc": op.name,
                        "postproc_kind": op.kind,
                        "postproc_param": float(op.param),
                        "label": label,
                        # rho(x), Eq. (4): mean of the winsorized per-patch log-ratio rho_p over
                        # patches at or above the low_etex_mask_quantile of E_tex (mask M(x)).
                        "rho_summary": image_rho_summary(
                            res.rho_patch,
                            res.e_tex_patch,
                            mask_quantile=cfg.rho.low_etex_mask_quantile,
                            winsor_quantiles=cfg.rho.winsor_quantiles,
                        ),
                        # Pair coordinate m(x), Eq. (8) (masked winsorized branch-difference
                        # magnitude) from the in-memory residual: lean runs never persist the
                        # residual store, so the scalar must be captured at inversion time or it is
                        # unrecoverable.
                        "m_summary": image_m_summary(
                            res.residual,
                            res.e_tex_patch,
                            mask_quantile=cfg.rho.low_etex_mask_quantile,
                            winsor_quantiles=cfg.rho.winsor_quantiles,
                        ),
                        "image_id": res.image_id,
                        "image": image_rel,
                        "trace": trace_rel,
                    }
                    # Wrong-prompt null (report-only, ONE tier = the undegraded baseline cell):
                    # re-invert with a PERMUTED prompt. If rho still separates without the true
                    # prompt, the signal is the channel, not prompt semantics.
                    is_baseline_cell = (
                        op.kind == "none"
                        and qf == cfg.evidence.wrong_prompt_subset_qf
                        and sigma_blur == 0.0
                    )
                    if not smoke and is_baseline_cell and len(prompts) > 1:
                        wrong = prompts[(pi + 1) % len(prompts)]
                        row["rho_summary_wrong_prompt"] = _rho_summary_for(
                            inversion_input, wrong, gen_seed
                        )
                    table.append(row)
                    # lean: the inversion already consumed the in-memory array; drop the PNG so the
                    # run stores ONLY results (no images/ accumulation on a space-tight host).
                    if lean:
                        (out / image_rel).unlink(missing_ok=True)

    # Oracle traces are captured at the generation step that mirrors the inversion capture step
    # (generation runs T->0, inversion 0->T, so index num_steps-1-capture_idx).
    capture_idx = max(0, min(steps - 1, int(cfg.patch.capture_step_frac * steps)))
    gen_trace_step = steps - 1 - capture_idx
    # lean (space-tight host) writes ONLY p0_table.json -- never oracle traces. The real sd_ddim
    # backends (sdxl/sd15/sd21) expose generate_image_with_trace, so without this guard _generate
    # would _save_trace() a .npz per (prompt, w) into traces/ -- which the lean PNG-unlink never
    # removes -- breaking the --lean "no traces" contract. Disabling also skips the hook compute.
    with_trace = None if lean else getattr(backend, "generate_image_with_trace", None)

    def _generate(prompt: str, w: float, pi: int, **gen_kwargs: int) -> tuple:
        if with_trace is None:
            return backend.generate_image(prompt, w=w, **gen_kwargs), None
        img, trace = with_trace(prompt, w=w, trace_step=gen_trace_step, **gen_kwargs)
        trace_rel = f"traces/p{pi}_s{gen_kwargs['seed']}_w{int(w)}.npz"
        _save_trace(trace, out / trace_rel)
        return img, trace_rel

    res_side = cfg.grid.resolution  # PREREGISTERED resolution (p0.yaml grid.resolution)
    for pi, prompt in enumerate(prompts):
        if pi not in shard_indices:  # this job owns only its shard's prompt indices
            continue
        for seed in seeds:
            gen_seed = _effective_seed(pi, seed)
            gen_kwargs = {
                "seed": gen_seed,
                "num_steps": steps,
                "height": res_side,
                "width": res_side,
            }
            h0, h0_trace = _generate(prompt, 1.0, pi, **gen_kwargs)
            _process(h0, prompt, 1, "h0", pi, seed, gen_seed, h0_trace)
            for w in ws:
                h1, h1_trace = _generate(prompt, float(w), pi, **gen_kwargs)
                _process(h1, prompt, w, "h1", pi, seed, gen_seed, h1_trace)

    # lean: skip the full-dim residual + compact store (256 MB residual_v1.dat is the space hog);
    # the scalar rho_summary/m_summary rows suffice for the pair robustness analysis.
    if not lean:
        batch, residual, meta = pack_results(
            results, model_name=decoder, branch_mode=spec.branch_mode
        )
        write_compact(out, batch, meta)
        write_residual_v1(out, residual, meta)
    # allow_nan=False: a non-finite summary must fail loud at the WRITE site, not serialize a
    # non-standard NaN token that downstream validation would only catch on the next read.
    (out / "p0_table.json").write_text(
        json.dumps(table, indent=2, allow_nan=False), encoding="utf-8"
    )
    store = "p0_table.json only (lean)" if lean else "feature store + p0_table.json"
    print(f"[ok] {decoder}: wrote {len(table)} rows -> {store} to {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cfgid.sweep.run_p0",
        description="Main-grid (p0) generation and dual-branch inversion driver",
    )
    parser.add_argument("--decoder", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--prompts-per-cell",
        type=int,
        default=None,
        help="override grid.prompts_per_cell for THIS job (per-decoder N via sbatch); else config",
    )
    parser.add_argument(
        "--sigma-blur",
        default=None,
        help="comma sigma_blur px overriding grid.sigma_blur (e.g. 0,1,2,4); else config",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="config path (default configs/p0.yaml); use an OFF-gate robustness config to add a "
        "postproc degradation axis without touching the preregistered gate",
    )
    parser.add_argument(
        "--lean",
        action="store_true",
        help="results-only: write ONLY p0_table.json (no residual/compact store, no traces, PNGs "
        "deleted after each inversion) -- for space-tight robustness runs",
    )
    parser.add_argument(
        "--prompt-shard",
        default=None,
        help="'k/total' contiguous prompt-index slice for sbatch fan-out (e.g. 0/2, 1/2); the "
        "un-sharded analysis pools every shard's p0_table.json back together",
    )
    args = parser.parse_args(argv)
    sigma_blur = None
    if args.sigma_blur is not None:
        sigma_blur = [float(x) for x in args.sigma_blur.split(",") if x.strip()]
    prompt_shard: tuple[int, int] | None = None
    if args.prompt_shard is not None:
        parts = args.prompt_shard.split("/")
        if len(parts) != 2:
            print(f"[error] --prompt-shard must be 'k/total', got {args.prompt_shard!r}")
            return 2
        try:
            prompt_shard = (int(parts[0]), int(parts[1]))
        except ValueError:
            print(f"[error] --prompt-shard k and total must be ints, got {args.prompt_shard!r}")
            return 2
    config_path = Path(args.config) if args.config is not None else None
    out = _default_output_root(smoke=bool(args.smoke), out_arg=args.out)
    return run(
        args.decoder,
        smoke=args.smoke,
        out_dir=out,
        prompts_per_cell=args.prompts_per_cell,
        sigma_blur=sigma_blur,
        config_path=config_path,
        lean=bool(args.lean),
        prompt_shard=prompt_shard,
    )


if __name__ == "__main__":
    sys.exit(main())
