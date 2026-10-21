"""Negative-control baselines for the main-grid (p0) evidence and gate.

Changing CFG also changes texture / quality / prompt adherence (with known oversaturation), so a
"pass" needs rho to be NOT weaker than cheap nuisance detectors after matching texture/semantics.
``nuisance`` ships the image-only baselines (FFT high-band, low-frequency, saturation, color
dispersion, SRM), whose AUCs give the "rho vs strongest nuisance" rows 1-2 of Table VIII (via
``cfgid.sweep.evidence`` / ``analyze`` and ``outputs/p0/decision_memo.md``); ``clip_align`` ships
the CLIP-text-alignment baseline (needs checkpoints/clip-vit-b32; reported N/A when unavailable).
"""

from cfgid.baselines.clip_align import (
    ClipUnavailableError,
    clip_alignment_scores,
    clip_available,
)
from cfgid.baselines.nuisance import best_nuisance_auc, nuisance_features

__all__ = [
    "nuisance_features",
    "best_nuisance_auc",
    "ClipUnavailableError",
    "clip_alignment_scores",
    "clip_available",
]
