"""Main-grid (p0) cell enumeration + leakage-proof split assignment (grouped by prompt).

A prompt is assigned to exactly one split, so every (w, QF, sigma_blur, seed) cell of
that prompt stays together -- fitting and evaluation never share a prompt (no optimistic
leakage). H0/H1 are paired: each prompt/seed yields the full w-row at fixed QF/decoder, so w=1 (H0)
and w>1 (H1) differ ONLY in guidance.

Splits (Section IV-A, "Data splits"): the configs/p0.yaml fractions train 0.40 / null_adapt 0.20 /
calibration 0.20 / test 0.20 with seed 0 are the paper's FIT / NULL / CAL / TEST splits.
``assign_splits`` is called by ``cfgid.sweep.run_p0`` once per decoder run, over that run's prompt
ids "p0", "p1", ...; the resulting split is stored in every p0_table.json row.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cfgid.p0config import P0Config


@dataclass(frozen=True)
class Cell:
    decoder: str
    w: int
    qf: int | None
    sigma_blur: float
    prompt_id: str
    seed: int
    split: str


def assign_splits(
    prompt_ids: list[str], fractions: dict[str, float], *, seed: int
) -> dict[str, str]:
    """Assign each unique prompt to one split (grouped; deterministic; no prompt spans splits).

    The unique ids are sorted as STRINGS (so "p10" precedes "p2") and then permuted with
    ``np.random.default_rng(seed)``; consecutive blocks of the permutation go to the splits in the
    order of ``fractions``. Each positive-fraction split first receives one prompt (when there are
    enough prompts), and the remainder is apportioned by largest remainders. The assignment
    therefore depends on the number of prompts in the run (512 or 256 on the paper's main grid).
    """
    if not fractions or abs(sum(fractions.values()) - 1.0) > 1e-6:
        raise ValueError(f"split fractions must sum to 1.0, got {fractions}")
    if any(v < 0.0 for v in fractions.values()):
        raise ValueError(f"split fractions must be non-negative, got {fractions}")
    uniq = sorted(set(prompt_ids))
    if not uniq:
        return {}
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    names = list(fractions)
    positive = [i for i, name in enumerate(names) if fractions[name] > 0.0]
    counts = np.zeros(len(names), dtype=np.int64)
    remaining = len(uniq)
    if len(uniq) >= len(positive):
        counts[positive] = 1
        remaining -= len(positive)
    if remaining > 0:
        weights = np.asarray([fractions[name] for name in names], dtype=np.float64)
        if weights.sum() <= 0.0:
            raise ValueError(f"at least one split fraction must be positive, got {fractions}")
        exact = weights / weights.sum() * remaining
        extra = np.floor(exact).astype(np.int64)
        counts += extra
        leftover = remaining - int(extra.sum())
        if leftover:
            frac_order = np.argsort(-(exact - extra), kind="mergesort")
            counts[frac_order[:leftover]] += 1
    # Cumulative integer boundaries so the partition is exact and stable. When the number of prompts
    # is at least the number of positive-fraction splits, no split is empty.
    bounds = np.cumsum(counts)
    bounds[-1] = len(uniq)  # absorb rounding into the last split
    out: dict[str, str] = {}
    b = 0
    for k, name in enumerate(names):
        for idx in order[b : bounds[k]]:
            out[uniq[idx]] = name
        b = bounds[k]
    return out


def enumerate_cells(cfg: P0Config, prompt_ids: list[str]) -> list[Cell]:
    """All main-grid cells across (decoder, QF, sigma_blur, prompt, seed, w)."""
    splits = assign_splits(prompt_ids, cfg.split.fractions, seed=cfg.split.seed)
    cells: list[Cell] = []
    for decoder in cfg.grid.decoders:
        for qf in cfg.grid.qf:
            for sigma_blur in cfg.grid.sigma_blur:
                for prompt_id in prompt_ids:
                    for s in range(cfg.grid.seeds_per_prompt):
                        for w in cfg.grid.w:
                            cells.append(
                                Cell(
                                    decoder=decoder,
                                    w=w,
                                    qf=qf,
                                    sigma_blur=float(sigma_blur),
                                    prompt_id=prompt_id,
                                    seed=s,
                                    split=splits[prompt_id],
                                )
                            )
    return cells
