"""Retargeted external-detector baselines: per-image feature collection (GPU).

Four published real-vs-generated detector families are re-scored on our estimand (CFG w=20 vs
w=1) on the main-grid images and splits. This script only COLLECTS per-image raw features into
per-(stage, decoder) JSONL checkpoints; orientation, LDA heads and all metrics are computed
afterwards by retarget_analyze.py from FIT rows only, so collection never uses evaluation labels.

Stages (``--stages``, comma list) and the per-image features they record:
* ``aeroblade``  AEROBLADE-style VAE round trip (Ricker et al., CVPR'24): the decoder's own VAE
    encode -> decode; pixel-space mean |x - x_hat| (``ae_l1``, used in Table IV), RMS error
    (``ae_l2``) and LPIPS-VGG (``ae_lpips``, only if the optional ``lpips`` package is
    installed), pixels in [-1, 1]. One scalar per feature.
* ``dire``       DIRE-style invert-regenerate (Wang et al., ICCV'23): deterministic DDIM
    inversion with the empty prompt and guidance 1 over 20 steps (DIRE_STEPS, as the original
    release's ddim20 setting), then 20-step DDIM regeneration and VAE decode; pixel-space
    mean |x - x_hat| (``dire_l1``, used in Table IV) and RMS error (``dire_l2``). Differences from
    the pixel-space ADM original: latent-space backbone at the grid's 512 resolution and no
    ``clip_denoised`` clamp (the schedulers run with clip_sample=False).
* ``wangk``      re-noising features (Wang & Kalogeiton, ECCV-W'24, ``concat2uncond`` path):
    for each of T=10 scheduler timesteps (WANGK_T, ascending) draw N=20 fresh Gaussian noises
    (WANGK_N; CPU torch generator seeded with ``--seed`` + row index), form
    x_t = add_noise(z0, eps, t) and run one batched empty-prompt denoiser forward; with
    e_opt = the mean of the 20 noises and e_uncond = the mean of the 20 predictions, keep the
    means over latent channels and positions of e_opt^2, (e_opt - e_uncond)^2 and e_uncond^2
    -> a 30-dimensional vector (``wangk_feats``, 3 values per timestep, timesteps ascending).
    Their 3D-conv + ResNet head is replaced by our
    FIT-only per-cell ridge-LDA head (retarget_analyze.py); the features use no guidance and no
    conditional branch, so they are prompt-free. Table IV's "Re-noising + LDA" row.
* ``xu``         attributor logits (Xu et al., WACV'25): their released frozen EfficientFormer
    torchscript attributor (12 generators + real); the 13 logits per image form the feature
    vector (``xu_logits``). Preprocessing mirrors their demo.py exactly, including its cv2 BGR
    channel order (224 px shorter-side bicubic resize, center crop, per-channel mean/std on the
    BGR array). Needs ``--xu-checkpoint`` (download per the authors' README, e.g. to
    ``checkpoints/xu_imageattribution/``); runs on CPU. Table IV's "Attributor logits + LDA" row.

For SD2.1 and HunyuanDiT the empty-prompt denoiser returns v rather than epsilon, which is what
their v-prediction DDIM schedulers consume (see ``_make_eps_fn``).

Row selection per decoder: ``p0_table.json`` rows with (label h0, w=1) or (label h1, w=W_HIGH),
all splits (FIT orientation needs the train rows); ``--cells sb0`` keeps sigma_blur=0 only,
``--cells all`` keeps every blur level. Every stage is checkpointed per (stage, decoder) and
resumable; failures raise (no silent NaN).

Produces (through retarget_analyze.py, unified_baselines.py, baseline_incremental.py and
renoising_group_compare.py): Table IV and the re-noising comparisons of Sections IV-B and IV-D.

Inputs: ``outputs/p0/<decoder>/p0_table.json``, the stored PNG images, the frozen backbone
weights (``configs/models.yaml``), the Xu et al. checkpoint (``xu`` stage) and a GPU (CPU/MPS
work for the smoke test).

Outputs: ``outputs/p0/retarget/retarget_{aeroblade,dire,wangk,xu}_<decoder>.jsonl`` (or
``--out``).

Usage (the paper's runs, one decoder per job):
    uv run --no-sync python scripts/analysis/retarget_baselines.py --stages aeroblade,xu \\
        --cells all --decoders sd15 --xu-checkpoint checkpoints/xu_imageattribution/end2end.pt
    uv run --no-sync python scripts/analysis/retarget_baselines.py --stages dire,wangk \\
        --cells sb0 --decoders sd15
Local smoke test (real backend, synthetic images; --smoke-decoder picks the backend, def sd15):
    uv run --no-sync python scripts/analysis/retarget_baselines.py --smoke
    uv run --no-sync python scripts/analysis/retarget_baselines.py --smoke --smoke-decoder sdxl
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from cfgid.channel.inversion.registry import get_backend
from cfgid.config import load_models_config
from cfgid.device import get_device
from cfgid.p0config import load_p0_config
from cfgid.paths import outputs_dir
from cfgid.sweep.neff_scaling import _load_p0_rows

W_HIGH = 20
DIRE_STEPS = 20  # DIRE release, compute_dire.sh: ddim20
WANGK_T = 10  # Wang & Kalogeiton release, compute_latent.sh: T=10
WANGK_N = 20  # Wang & Kalogeiton release, compute_latent.sh: num_samples=20
STAGES = ("aeroblade", "dire", "wangk", "xu")
DECODERS = ("sd15", "sd21", "sdxl", "pixart_sigma", "hunyuandit")


# --- shared plumbing ------------------------------------------------------------------------


def _select_rows(dec_dir: Path, cells: str) -> list[dict]:
    rows = _load_p0_rows(dec_dir / "p0_table.json")
    keep = []
    for r in rows:
        lab, w = str(r.get("label")), int(r["w"])
        if not ((lab == "h0" and w == 1) or (lab == "h1" and w == W_HIGH)):
            continue
        if cells == "sb0" and float(r["sigma_blur"]) != 0.0:
            continue
        keep.append(r)
    return keep


def _load_png(dec_dir: Path, row: dict) -> np.ndarray:
    png = dec_dir / str(row["image"])
    img = np.asarray(Image.open(png).convert("RGB"), dtype=np.float32) / 255.0
    if not np.all(np.isfinite(img)):
        raise ValueError(f"non-finite pixels in {png}")
    return img


def _ckpt_path(out_dir: Path, stage: str, decoder: str) -> Path:
    return out_dir / f"retarget_{stage}_{decoder}.jsonl"


def _load_done(path: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if path.exists():
        with path.open() as fh:
            for line in fh:
                rec = json.loads(line)
                done[rec["image_id"]] = rec
    return done


def _append(path: Path, rec: dict) -> None:
    with path.open("a") as fh:
        fh.write(json.dumps(rec, allow_nan=False) + "\n")


def _row_meta(row: dict) -> dict:
    return {
        "image_id": str(row["image_id"]),
        "split": str(row["split"]),
        "label": str(row["label"]),
        "w": int(row["w"]),
        "sigma_blur": float(row["sigma_blur"]),
    }


def _load_backend(decoder: str):
    models = load_models_config()
    spec = models[decoder]
    if not spec.is_present():
        raise FileNotFoundError(f"weights for {decoder!r} not present under checkpoints/")
    backend = get_backend(spec, get_device())
    backend.load()
    return backend


def _latent_tensor(backend, latent_obj) -> torch.Tensor:
    """_encode_image_to_latent may return a torch tensor (on device) or a numpy array."""
    if isinstance(latent_obj, torch.Tensor):
        t = latent_obj.detach()
    else:
        t = torch.from_numpy(np.asarray(latent_obj, dtype=np.float32))
    if t.dim() == 3:
        t = t[None]
    return t.to(backend.device)


def _to_pixel_tensor(img: np.ndarray, device, dtype) -> torch.Tensor:
    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return (x * 2.0 - 1.0).to(device=device, dtype=dtype)


def _vae_decode_pixels(backend, latent: torch.Tensor) -> torch.Tensor:
    """latent (1,C,h,w) scaled by scaling_factor -> pixels (1,3,H,W) in [-1,1], fp32."""
    vae = backend._pipe.vae
    needs_upcast = vae.dtype == torch.float16 and bool(getattr(vae.config, "force_upcast", False))
    z = latent / vae.config.scaling_factor
    if needs_upcast:
        vae.to(torch.float32)
        z = z.to(torch.float32)
    else:
        z = z.to(vae.dtype)
    with torch.no_grad():
        px = vae.decode(z).sample
    if needs_upcast:
        vae.to(torch.float16)
    px = px.to(torch.float32).clamp(-1.0, 1.0)
    if not torch.all(torch.isfinite(px)):
        raise ValueError("non-finite VAE decode output")
    return px


# --- stage: aeroblade -----------------------------------------------------------------------


def _lpips_or_none(device):
    try:
        import lpips  # type: ignore[import-not-found]  # optional `attack` extra

        return lpips.LPIPS(net="vgg").to(device).eval()
    except Exception:
        return None


def _stage_aeroblade(backend, rows: list[dict], dec_dir: Path, ckpt: Path) -> None:
    """VAE encode -> decode round trip; L1 / RMS (/ LPIPS) error in [-1, 1] pixel space."""
    done = _load_done(ckpt)
    lp = _lpips_or_none(backend.device)
    if lp is None:
        print("[aeroblade] lpips unavailable (optional extra not installed); L1/L2 only")
    for i, row in enumerate(rows):
        if str(row["image_id"]) in done:
            continue
        img = _load_png(dec_dir, row)
        latent = _latent_tensor(backend, backend._encode_image_to_latent(img))
        recon = _vae_decode_pixels(backend, latent.to(torch.float32))
        x = _to_pixel_tensor(img, backend.device, torch.float32)
        rec = _row_meta(row)
        diff = (x - recon).abs()
        rec["ae_l1"] = float(diff.mean())
        rec["ae_l2"] = float(diff.pow(2).mean().sqrt())
        if lp is not None:
            with torch.no_grad():
                rec["ae_lpips"] = float(lp(x, recon).flatten()[0])
        _append(ckpt, rec)
        if (i + 1) % 50 == 0:
            print(f"[aeroblade] {i + 1}/{len(rows)}")


# --- per-architecture empty-prompt denoiser closure ------------------------------------------

# eps_fn(z (B,C,h,w), t scalar timestep tensor) -> model prediction (B,C,h,w) fp32.
EpsFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _expand_batch(t: torch.Tensor, b: int) -> torch.Tensor:
    """Expand a batch-1 conditioning tensor along dim 0 (a view, as the diffusers pipelines do)."""
    return t.expand(b, *(-1 for _ in range(t.ndim - 1)))


def _make_eps_fn(backend) -> tuple[EpsFn, torch.dtype]:
    """Build the single-forward, guidance-free, EMPTY-prompt denoiser closure for one backend.

    Returns ``(eps_fn, model_dtype)``. ``eps_fn(z, t)`` runs ONE denoiser forward conditioned on
    the empty prompt and returns the scheduler-native prediction in fp32 (learned-sigma/-variance
    outputs are chunked to the first half). It strictly mirrors each backend's own
    ``_ddim_invert_capture`` call pattern rather than guessing pipeline signatures:

    * ``sd_ddim`` / ``sd_vpred_ddim`` / ``sdxl_ddim``: ``_encode_branch(prompt, latent)`` ->
      ``unet(z, t, encoder_hidden_states=embeds, added_cond_kwargs=added)``; SDXL's override
      supplies pooled text embeds + add_time_ids in ``added``, so the same call works for all 3.
    * ``pixart_ddim``: ``_encode_branch("")`` (T5 embeds + attention mask) -> ``_denoise`` (the
      backend broadcasts the timestep, passes resolution/aspect None, chunks learned sigma).
    * ``hunyuan_dit_ddim``: ``_encode_branch("")`` (CLIP + mT5 embeds/masks) +
      ``_build_image_conditioning`` (rotary/meta-size/style, cached per latent size) ->
      ``_v_output``. As in the backend's own inversion loop the prediction is the raw **v**,
      which is exactly what this pipe's v_prediction-configured DDIM(Inverse) schedulers consume
      (same is true for sd21's UNet output; rho's v->epsilon mapping is not needed here).

    The empty prompt is encoded once per decoder (lazily where the encoding needs the latent
    shape) and cached; per-call work is a batch expand (B>1 for wangk) + the denoiser forward.
    Raises NotImplementedError for backends without a supported path (e.g. ``flux_flow``).
    """
    pipe = backend._pipe
    key = str(backend.backend_key)

    if key in ("sd_ddim", "sd_vpred_ddim", "sdxl_ddim"):
        unet = pipe.unet
        model_dtype = next(iter(unet.parameters())).dtype
        enc_cache: dict[tuple[int, int], tuple[torch.Tensor, dict[str, torch.Tensor] | None]] = {}

        def eps_fn_sd(z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            shape_key = (int(z.shape[-2]), int(z.shape[-1]))
            if shape_key not in enc_cache:
                # sd_ddim._encode_branch ignores the latent; sdxl_ddim uses only its shape
                # (add_time_ids = [h, w, 0, 0, h, w] with h/w = latent*8).
                enc_cache[shape_key] = backend._encode_branch("", z[:1])
            embeds, added = enc_cache[shape_key]
            b = int(z.shape[0])
            embeds_b = _expand_batch(embeds, b)
            added_b = None if added is None else {k: _expand_batch(v, b) for k, v in added.items()}
            with torch.no_grad():
                out = unet(
                    z.to(model_dtype),
                    t,
                    encoder_hidden_states=embeds_b,
                    added_cond_kwargs=added_b,
                ).sample
            if out.shape[1] == 2 * z.shape[1]:  # learned-sigma style split (safety; not hit here)
                out = out[:, : z.shape[1]]
            return out.to(torch.float32)

        return eps_fn_sd, model_dtype

    if key == "pixart_ddim":
        model_dtype = next(iter(pipe.transformer.parameters())).dtype
        embeds, mask = backend._encode_branch("")  # T5 embeds + attention mask, batch 1

        def eps_fn_pixart(z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            b = int(z.shape[0])
            # _denoise broadcasts the timestep to z.shape[0] and chunks learned sigma itself.
            eps = backend._denoise(
                z.to(model_dtype), t, _expand_batch(embeds, b), _expand_batch(mask, b)
            )
            return eps.to(torch.float32)

        return eps_fn_pixart, model_dtype

    if key == "hunyuan_dit_ddim":
        model_dtype = next(iter(pipe.transformer.parameters())).dtype
        branch = backend._encode_branch("")  # CLIP + mT5 embeds/masks, batch 1
        img_cond_cache: dict[tuple[int, int], dict] = {}

        def eps_fn_hunyuan(z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            shape_key = (int(z.shape[-2]), int(z.shape[-1]))
            if shape_key not in img_cond_cache:
                # Prompt-independent micro-conditioning; latent is pixels/8 (resolution-matched).
                img_cond_cache[shape_key] = backend._build_image_conditioning(
                    shape_key[0] * 8, shape_key[1] * 8, model_dtype
                )
            img_cond = img_cond_cache[shape_key]
            b = int(z.shape[0])
            # repeat (contiguous), NOT expand: HunyuanDiT2DModel.forward view(-1)s the mT5 embeds
            # and image_meta_size, which rejects expand's stride-0 views. Mirrors the pipeline's
            # own `.repeat(batch_size * num_images_per_prompt, 1)` batching.
            branch_b = {k: v.repeat(b, *(1 for _ in range(v.ndim - 1))) for k, v in branch.items()}
            img_cond_b = {
                "image_rotary_emb": img_cond["image_rotary_emb"],  # positional; batch-free
                "image_meta_size": img_cond["image_meta_size"].repeat(b, 1),
                "style": img_cond["style"].repeat(b),
            }
            # _v_output mirrors HunyuanDiTPipeline.__call__ exactly (timestep expand + float
            # cast, learned-variance chunk); the raw v is the scheduler-native prediction.
            v = backend._v_output(z.to(model_dtype), t, branch_b, img_cond_b)
            return v.to(torch.float32)

        return eps_fn_hunyuan, model_dtype

    raise NotImplementedError(
        f"backend '{key}' has no single-forward empty-prompt denoiser path in this script"
    )


# --- stage: dire ----------------------------------------------------------------------------


def _dire_score(backend, eps_fn: EpsFn, model_dtype: torch.dtype, img: np.ndarray) -> dict:
    """DDIM invert (empty prompt, guidance 1, DIRE_STEPS) -> regenerate -> pixel error."""
    pipe = backend._pipe
    z = _latent_tensor(backend, backend._encode_image_to_latent(img)).to(model_dtype)
    inv, fwd = pipe.inverse_scheduler, pipe.scheduler
    inv.set_timesteps(DIRE_STEPS, device=backend.device)
    fwd.set_timesteps(DIRE_STEPS, device=backend.device)
    with torch.no_grad():
        for t in inv.timesteps:  # x0 -> x_T
            # eps_fn already chunks learned-sigma outputs; step in model dtype as the backends do.
            z = inv.step(eps_fn(z, t).to(z.dtype), t, z).prev_sample
        for t in fwd.timesteps:  # x_T -> x0
            z = fwd.step(eps_fn(z, t).to(z.dtype), t, z).prev_sample
    recon = _vae_decode_pixels(backend, z.to(torch.float32))
    x = _to_pixel_tensor(img, backend.device, torch.float32)
    diff = (x - recon).abs()
    return {"dire_l1": float(diff.mean()), "dire_l2": float(diff.pow(2).mean().sqrt())}


def _stage_dire(
    backend, eps_fn: EpsFn, model_dtype: torch.dtype, rows: list[dict], dec_dir: Path, ckpt: Path
) -> None:
    done = _load_done(ckpt)
    for i, row in enumerate(rows):
        if str(row["image_id"]) in done:
            continue
        rec = _row_meta(row)
        rec.update(_dire_score(backend, eps_fn, model_dtype, _load_png(dec_dir, row)))
        _append(ckpt, rec)
        if (i + 1) % 20 == 0:
            print(f"[dire] {i + 1}/{len(rows)}")


# --- stage: wangk ---------------------------------------------------------------------------


def _wangk_features(backend, eps_fn: EpsFn, img: np.ndarray, seed: int) -> list[float]:
    """concat2uncond re-noising features, spatially pooled: 3 scalars x T timesteps."""
    pipe = backend._pipe
    z0 = _latent_tensor(backend, backend._encode_image_to_latent(img)).to(torch.float32)
    sched = pipe.scheduler
    sched.set_timesteps(WANGK_T, device=backend.device)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    feats: list[float] = []
    with torch.no_grad():
        for t in reversed(sched.timesteps):  # ascending t, as in their released inversion.py
            eps = torch.randn((WANGK_N, *z0.shape[1:]), generator=gen, dtype=torch.float32).to(
                backend.device
            )
            x_t = sched.add_noise(z0.expand(WANGK_N, -1, -1, -1), eps, t.expand(WANGK_N))
            # One batched (B=WANGK_N) empty-prompt denoiser forward; eps_fn handles the model
            # dtype cast, the per-architecture conditioning, and the learned-sigma chunk.
            e_u = eps_fn(x_t, t)
            # average the N draws first, then keep three squared maps averaged over (C, h, w)
            e_u_m = e_u.mean(dim=0)
            e_o_m = eps.mean(dim=0)
            feats.extend(
                [
                    float((e_o_m**2).mean()),
                    float(((e_o_m - e_u_m) ** 2).mean()),
                    float((e_u_m**2).mean()),
                ]
            )
    if not all(np.isfinite(feats)):
        raise ValueError("non-finite wangk features")
    return feats


def _stage_wangk(
    backend, eps_fn: EpsFn, rows: list[dict], dec_dir: Path, ckpt: Path, seed: int
) -> None:
    done = _load_done(ckpt)
    for i, row in enumerate(rows):
        if str(row["image_id"]) in done:
            continue
        rec = _row_meta(row)
        rec["wangk_feats"] = _wangk_features(
            backend, eps_fn, _load_png(dec_dir, row), seed=seed + i
        )
        _append(ckpt, rec)
        if (i + 1) % 20 == 0:
            print(f"[wangk] {i + 1}/{len(rows)}")


# --- stage: xu ------------------------------------------------------------------------------

_XU_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_XU_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def _xu_preprocess(img01: np.ndarray, size: int = 224) -> torch.Tensor:
    """Mirror k8xu demo.py exactly: cv2 BGR order + shorter-side resize + center crop +
    per-channel (mean,std) applied to the BGR array. We keep the BGR quirk deliberately —
    the released torchscript was deployed with it."""
    bgr = (img01[:, :, ::-1] * 255.0).astype(np.float32)
    h, w = bgr.shape[:2]
    scale = size / min(h, w)
    nh, nw = int(h * scale), int(w * scale)
    pil = Image.fromarray(np.clip(bgr, 0, 255).astype(np.uint8))
    bgr = np.asarray(pil.resize((nw, nh), Image.Resampling.BICUBIC), dtype=np.float32)
    sh, sw = (nh - size) // 2, (nw - size) // 2
    bgr = bgr[sh : sh + size, sw : sw + size]
    bgr = (bgr - _XU_MEAN[None, None, :]) / _XU_STD[None, None, :]
    return torch.from_numpy(bgr).permute(2, 0, 1).unsqueeze(0).float()


def _stage_xu(rows: list[dict], dec_dir: Path, ckpt: Path, checkpoint: Path, device) -> None:
    model = torch.jit.load(str(checkpoint), map_location=device)
    done = _load_done(ckpt)
    for i, row in enumerate(rows):
        if str(row["image_id"]) in done:
            continue
        x = _xu_preprocess(_load_png(dec_dir, row)).to(device)
        with torch.no_grad():
            logits = model(x).flatten().to(torch.float32).cpu().numpy()
        if not np.all(np.isfinite(logits)):
            raise ValueError(f"non-finite xu logits for {row['image_id']}")
        rec = _row_meta(row)
        rec["xu_logits"] = [float(v) for v in logits]
        _append(ckpt, rec)
        if (i + 1) % 200 == 0:
            print(f"[xu] {i + 1}/{len(rows)}")


# --- orchestration --------------------------------------------------------------------------


def run(
    features_root: Path,
    out_dir: Path,
    *,
    stages: list[str],
    decoders: list[str],
    cells: str,
    limit: int | None,
    seed: int,
    xu_checkpoint: Path | None,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    _ = load_p0_config()  # fail-loud on config drift before any GPU work
    for decoder in decoders:
        dec_dir = features_root / decoder
        if not (dec_dir / "p0_table.json").exists():
            print(f"[skip] {decoder}: no p0_table.json under {dec_dir}")
            continue
        rows = _select_rows(dec_dir, cells)
        if limit is not None:
            rows = rows[:limit]
        print(f"[{decoder}] {len(rows)} rows (cells={cells})")
        need_backend = any(s in stages for s in ("aeroblade", "dire", "wangk"))
        backend = _load_backend(decoder) if need_backend else None
        t0 = time.time()
        if "aeroblade" in stages:
            _stage_aeroblade(backend, rows, dec_dir, _ckpt_path(out_dir, "aeroblade", decoder))
        if "dire" in stages or "wangk" in stages:
            try:
                eps_fn, model_dtype = _make_eps_fn(backend)
            except NotImplementedError as exc:
                print(f"[skip] {decoder}: dire/wangk unsupported ({exc})")
            else:
                if "dire" in stages:
                    _stage_dire(
                        backend,
                        eps_fn,
                        model_dtype,
                        rows,
                        dec_dir,
                        _ckpt_path(out_dir, "dire", decoder),
                    )
                if "wangk" in stages:
                    _stage_wangk(
                        backend, eps_fn, rows, dec_dir, _ckpt_path(out_dir, "wangk", decoder), seed
                    )
        if "xu" in stages:
            if xu_checkpoint is None or not xu_checkpoint.exists():
                print("[skip] xu: --xu-checkpoint missing (download per k8xu README)")
            else:
                _stage_xu(rows, dec_dir, _ckpt_path(out_dir, "xu", decoder), xu_checkpoint, "cpu")
        print(f"[{decoder}] done in {time.time() - t0:.0f}s")
    print(f"-> JSONL checkpoints under {out_dir}")
    return 0


def _smoke(decoder: str = "sd15") -> int:
    """Real ``decoder`` backend on this machine (MPS), synthetic 512x512 images, 2 rows/stage."""
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="retarget_smoke_"))
    dec_dir = tmp / decoder
    (dec_dir / "images").mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows = []
    for k, (lab, w) in enumerate([("h0", 1), ("h1", W_HIGH)]):
        img = (rng.random((512, 512, 3)) * 255).astype(np.uint8)
        image_id = f"{decoder}_p{k}_s0_w{w}_{lab}_qfnone_sb0"
        Image.fromarray(img).save(dec_dir / "images" / f"{image_id}.png")
        rows.append(
            {
                "image_id": image_id,
                "image": f"images/{image_id}.png",
                "split": "train" if k == 0 else "test",
                "label": lab,
                "w": w,
                "sigma_blur": 0.0,
                "decoder": decoder,
            }
        )
    (dec_dir / "p0_table.json").write_text(json.dumps(rows))
    return run(
        tmp,
        tmp / "out",
        stages=["aeroblade", "dire", "wangk"],
        decoders=[decoder],
        cells="sb0",
        limit=2,
        seed=0,
        xu_checkpoint=None,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="retarget_baselines")
    ap.add_argument("--features-root", default=None)
    ap.add_argument("--out", default=None, help="def outputs/p0/retarget/")
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--decoders", default=",".join(DECODERS))
    ap.add_argument("--cells", choices=("sb0", "all"), default="sb0")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--xu-checkpoint", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-decoder", default="sd15", choices=DECODERS)
    args = ap.parse_args(argv)
    if args.smoke:
        return _smoke(args.smoke_decoder)
    features_root = Path(args.features_root) if args.features_root else (outputs_dir() / "p0")
    out_dir = Path(args.out) if args.out else (outputs_dir() / "p0" / "retarget")
    return run(
        features_root,
        out_dir,
        stages=[s.strip() for s in args.stages.split(",") if s.strip()],
        decoders=[d.strip() for d in args.decoders.split(",") if d.strip()],
        cells=args.cells,
        limit=args.limit,
        seed=args.seed,
        xu_checkpoint=Path(args.xu_checkpoint) if args.xu_checkpoint else None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
