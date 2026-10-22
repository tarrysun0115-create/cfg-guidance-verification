# Verification of classifier-free guidance strength in diffusion-generated images (code for peer review)

This repository holds the core code behind the manuscript under review: the detection library
and the scripts that produce the main comparisons. Section, table, figure and equation numbers
refer to the submitted PDF and its supplementary material (numbers prefixed with S). In module docstrings,
Appendix B/C, Table XV and Lemma C.2 correspond to supplementary Sections S2/S3, Table S2 and Lemma S2. Result files, the test suite, figure scripts and auxiliary diagnostic
scripts are not included; module docstrings occasionally mention such auxiliary scripts by name.

## Overview

Given an image and its prompt, the code inverts the image twice with a frozen latent-diffusion
decoder, once conditioned on the prompt and once on the empty prompt (deterministic DDIM
inversion; Section III-B, Eq. (1)). It captures both noise predictions at the inversion midpoint
and reduces them over 4x4 latent patches to two image-level coordinates: ρ(x), the texture-masked,
winsorized patch mean of the guarded log ratio of the two branch norms (Section III-C,
Eqs. (2)-(4)), and m(x), the same mean of the branch-difference magnitude (Eqs. (6)-(8)). The
deployed readout T_pair-LDA is a plug-in two-dimensional Fisher discriminant on (ρ, m), fitted per
(decoder, σ_blur) cell on the FIT split (Section III-D, Eq. (11)); ρ alone and T_pair-zsum are
comparison readouts. A Gaussian-location model gives the single-image AUC, the channel total
variation, the √K aggregation law and the δ-deniability threshold (Section III-E). Conformal
thresholds on the held-out CAL split control the false-positive rate for single images and for
means of K images (Section IV-D). No network is trained or fine-tuned: all models are used frozen,
and the readouts and baseline heads are linear fits on stored per-image values (numpy/scipy).

## License

Provided solely for peer review of the manuscript; all rights reserved (see [LICENSE](LICENSE)).
Third-party models, weights and datasets keep their own licenses.

## Repository layout

```
configs/           p0.yaml (main-grid protocol), models.yaml (model snapshots),
                   paths.yaml (data roots), robustness.yaml (post-processing pool)
src/cfgid/         library
  channel/           inversion backends, rho / E_tex extraction, decoder PSF (sigma) calibration
  featurestore/      on-disk per-image feature store written by the main-grid run
  sweep/             main-grid driver (run_p0), splits, gate analysis, favorable-subset selection
  bounds/            TV, w*, Monte-Carlo TV check (Section III-E)
  detector/          AUC/ROC primitives, white-box likelihood-ratio detector
  align/ baselines/ attack/   knee-vs-w* gate, negative controls, surrogate attack
scripts/           run_p0 (main-grid entry point), _env_paths.sh
scripts/analysis/  analysis scripts (map below) and their shared helpers
scripts/paper/     sqrtk_pool_curves.py (Table III rho rows; also used by pair_readout_battery.py)
```

"p0" names the preregistered main-grid protocol of `configs/p0.yaml`.
`src/cfgid/channel/inversion/flux_flow.py` and the FLUX.1-dev and Playground v2.5 entries of
`configs/models.yaml` are not used in the paper.

## Installation

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12 (pinned in `.python-version`).

```bash
source scripts/_env_paths.sh   # keeps HF_HOME, TORCH_HOME, UV_CACHE_DIR, ... under .cache/
uv sync                        # locked dependencies (uv.lock)
uv run --no-sync python -m cfgid.doctor   # read-only preflight: device, configs, model snapshots
```

PyTorch is pinned to 2.6; on Linux uv takes the CUDA 12.4 build, elsewhere the PyPI build. The
device is chosen automatically (CUDA > MPS > CPU, `src/cfgid/device.py`). Data roots are set in
`configs/paths.yaml` or via `CFGID_CHECKPOINTS_DIR`, `CFGID_DATA_DIR`, `CFGID_OUTPUTS_DIR`.

## Models

The five decoders (Table S2 of the supplementary material), from `configs/models.yaml` (fp16, DDIM scheduler). Weights are
loaded only from `checkpoints/<key>/`; nothing is downloaded automatically.

| Key | Hugging Face repo | Backend |
|---|---|---|
| `sd15` | `stable-diffusion-v1-5/stable-diffusion-v1-5` | `sd_ddim` |
| `sd21` | `stabilityai/stable-diffusion-2-1` | `sd_vpred_ddim` |
| `sdxl` | `stabilityai/stable-diffusion-xl-base-1.0` | `sdxl_ddim` |
| `pixart_sigma` | `PixArt-alpha/PixArt-Sigma-XL-2-512-MS` (transformer) over `PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers` (other components) | `pixart_ddim` |
| `hunyuandit` | `Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers` | `hunyuan_dit_ddim` |

```bash
uv run --no-sync huggingface-cli download stable-diffusion-v1-5/stable-diffusion-v1-5 --local-dir checkpoints/sd15
uv run --no-sync huggingface-cli download stabilityai/stable-diffusion-2-1 --local-dir checkpoints/sd21
uv run --no-sync huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 --local-dir checkpoints/sdxl
uv run --no-sync huggingface-cli download PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers --local-dir checkpoints/pixart_sigma \
    --include "model_index.json" "scheduler/*" "text_encoder/*" "tokenizer/*" "vae/*"
uv run --no-sync huggingface-cli download PixArt-alpha/PixArt-Sigma-XL-2-512-MS --local-dir checkpoints/pixart_sigma \
    --include "transformer/*"
uv run --no-sync huggingface-cli download Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers --local-dir checkpoints/hunyuandit
```

PixArt-Σ is an assembled snapshot: the 512-MS `transformer/` is placed over the scheduler, T5
encoder, tokenizer and SDXL VAE of `pixart_sigma_sdxlvae_T5_diffusers`. The CLIP negative control
uses `openai/clip-vit-base-patch32` in `checkpoints/clip-vit-b32/`; the attributor baseline of
Table IV uses the released TorchScript checkpoint of its authors
(`checkpoints/xu_imageattribution/end2end.pt`).

## Pipeline

Run every command from the repository root after `source scripts/_env_paths.sh`. The GPU stages
of the paper ran on NVIDIA A100 GPUs, one job per decoder (and per prompt shard where noted).

### 1. Main grid: generation and inversion [GPU] (Section IV-A, supplementary Section S2)

`cfgid.sweep.run_p0` generates the H0 (w=1) and H1 (w in {2,3,5,7,10,14,20}) images of every
prompt (shared seed per prompt), stores each as an 8-bit PNG after the post-decode blur
σ_blur in {0,8,16,32}, runs the dual-branch DDIM inversion (50 steps, capture at step 25) and
writes `outputs/p0/<decoder>/p0_table.json` plus the per-image feature store, with prompts split
into FIT/NULL/CAL/TEST. The paper ran one job per decoder, overriding the `configs/p0.yaml`
default of 128 prompts per cell:

```bash
scripts/run_p0 --decoder sd15         --prompts-per-cell 512 --sigma-blur 0,8,16,32
scripts/run_p0 --decoder sdxl         --prompts-per-cell 512 --sigma-blur 0,8,16,32
scripts/run_p0 --decoder sd21         --prompts-per-cell 256 --sigma-blur 0,8,16,32
scripts/run_p0 --decoder pixart_sigma --prompts-per-cell 256 --sigma-blur 0,8,16,32
scripts/run_p0 --decoder hunyuandit   --prompts-per-cell 256 --sigma-blur 0,8,16,32
# one-step smoke test:
scripts/run_p0 --decoder sdxl --smoke
```

### 2. Decoder PSF calibration and gate analysis

```bash
for d in sd15 sd21 sdxl pixart_sigma; do uv run --no-sync python -m cfgid.channel.sigma_calib --decoder $d; done
uv run --no-sync python -m cfgid.sweep.analyze          # Table VIII rows 1-2, w*/knee verdict
uv run --no-sync python -m cfgid.sweep.pooled_neff_scaling
uv run --no-sync python scripts/analysis/master_table.py
uv run --no-sync python -m cfgid.bounds.montecarlo      # Monte-Carlo TV check (Section III-E, supplementary Section S2)
```

### 3. Main comparisons (CPU, on the main-grid outputs)

All scripts below run on CPU from `p0_table.json`; those marked * also need the feature store
written in stage 1. Each module docstring names the paper items it produces, its inputs and
outputs. (`python` stands for `uv run --no-sync python`.)

| Paper item | Command |
|---|---|
| Table I ρ rows, ρ permutation p, Fig. 2(D) data | `python scripts/analysis/fixed_h1_allcell.py` |
| Table I pair rows, Table III pair rows* | `python scripts/analysis/pair_readout_battery.py` (run first: later scripts gate against its output) |
| Table I weaker-contrast rows (pooled/macro AUC_1, exponent CI, TPR)* | `python scripts/analysis/weak_contrast_rows.py`, `python scripts/analysis/pair_variant_ci_matched.py` |
| Table I ρ same-cell AUC_16 for {10,14,20} vs 1 and w=20 vs 1* | `python scripts/analysis/samecell_nonoverlap.py` |
| Table I other same-cell AUC_16 entries (deployed head), Table V, Section IV-C* | `python scripts/analysis/deployed_within_contrasts.py` |
| Table II (block inference)* | `python scripts/analysis/block_inference_deployed.py` |
| Table III ρ rows | `python scripts/paper/sqrtk_pool_curves.py --features-root outputs/p0 --out-dir outputs/p0` |
| Fig. 3, Section IV-C exponents* | `python scripts/analysis/within_config_curves.py` |
| Section IV-D main-grid conformal FPR (K=1, K>1)* | `python scripts/analysis/cal_split_fpr.py`, then `python scripts/analysis/group_conformal_k.py` |
| Section IV-E forensic toolbox comparison, Fig. 1 bars | `python scripts/analysis/forensic_baselines.py` (needs the stored PNGs) |
| Table IX (prompt mismatch)* | `python scripts/analysis/prompt_mismatch_pair.py` (wrong-prompt re-inversion on GPU; `--audit-only` on CPU) |
| Table X (channel ablation)* | `python scripts/analysis/feature_comparison.py` (after stage 2) |

The remaining files in `scripts/analysis/` are shared helpers imported by these scripts.

### 4. Fresh batches (Section IV-D, Fig. 4, Tables VI-VII) [GPU]

Both batches use SDXL and PixArt-Σ at w=20 vs 1, with prompts disjoint from the main grid and from
each other. Heads, thresholds and predictions are frozen and SHA-256-bound before generation.

First batch (2,048 prompts per decoder; Fig. 4, Table VI):

```bash
uv run --no-sync python scripts/analysis/fresh_batch_validation.py --stage predict --decoders sdxl,pixart_sigma --n-prompts 2048 --n-shards 2
for d in sdxl pixart_sigma; do for s in 0 1; do      # 2 decoders x 2 prompt shards [GPU]
  uv run --no-sync python scripts/analysis/fresh_batch_validation.py --stage run --decoder $d --n-prompts 2048 --n-shards 2 --shard $s
done; done
uv run --no-sync python scripts/analysis/fresh_batch_validation.py --stage summarize --decoder sdxl
uv run --no-sync python scripts/analysis/fresh_batch_validation.py --stage summarize --decoder pixart_sigma
uv run --no-sync python scripts/analysis/fresh_tail_compare.py
```

Second batch (prospective calibration, 1,024 prompts per decoder; Table VII). Run every step in a
shell that exports the same three variables; the order cannot be changed:

```bash
export CFGID_FRESH_DIR=outputs/fresh_cal_batch   # separate directory; the first batch is untouched
export CFGID_PROMPT_OFFSET=110000                # disjoint from the main grid and the first batch
export CFGID_SEED_BASE=3000000                   # disjoint from the first batch (1000000)
uv run --no-sync python scripts/analysis/freeze_prospective_cal.py          # freeze heads and CAL thresholds
uv run --no-sync python scripts/analysis/fresh_batch_validation.py --stage predict --decoders sdxl,pixart_sigma --n-prompts 1024 --n-shards 2
for d in sdxl pixart_sigma; do for s in 0 1; do      # [GPU]
  uv run --no-sync python scripts/analysis/fresh_batch_validation.py --stage run --decoder $d --n-prompts 1024 --n-shards 2 --shard $s
done; done
uv run --no-sync python scripts/analysis/eval_prospective_cal.py            # frozen thresholds only; nothing refit
```

### 5. Retargeted baselines (Table IV) [GPU, then CPU]

```bash
for d in sd15 sd21 sdxl pixart_sigma hunyuandit; do    # feature collection [GPU, stored PNGs]
  uv run --no-sync python scripts/analysis/retarget_baselines.py --stages aeroblade,xu --cells all --decoders $d \
      --xu-checkpoint checkpoints/xu_imageattribution/end2end.pt
  uv run --no-sync python scripts/analysis/retarget_baselines.py --stages dire,wangk --cells sb0 --decoders $d
done
uv run --no-sync python scripts/analysis/retarget_analyze.py
uv run --no-sync python scripts/analysis/retarget_analyze.py --pool sb0 --probe-wangk --out outputs/p0/retarget_analysis_sb0.json
uv run --no-sync python scripts/analysis/retarget_analyze.py --pool sb0 --resid-vs-physical --out outputs/p0/retarget_analysis_resid.json
uv run --no-sync python scripts/analysis/baseline_incremental.py --retarget-dir outputs/p0/retarget
uv run --no-sync python scripts/analysis/unified_baselines.py
```

## Determinism

Seeds are fixed in code and command-line defaults (split seed 0, generation seed per prompt,
fresh-batch seeds `seed base + prompt index`, and per-analysis seeds for every permutation,
bootstrap and Monte-Carlo draw). Cross-cell AUC_K values and partition-averaged same-cell values
are Monte-Carlo estimates (Section IV-A, supplementary Section S2); a different number of draws, or a different
order of contrasts sharing one generator, gives a different estimate in the third decimal. GPU
generation and inversion can differ in low-order digits across GPU models and library versions;
the analyses consume the stored per-image values.
