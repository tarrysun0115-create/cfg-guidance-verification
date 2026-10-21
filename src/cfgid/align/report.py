"""Main-grid (p0) decision memo (dependency-free text) + alignment scatter figure.

``build_decision_memo`` is pure text (testable). ``save_alignment_scatter`` writes the PNG through
matplotlib's Agg backend; it still returns False if matplotlib is unavailable in a partial
environment, and the caller records that the figure was skipped. ``cfgid.sweep.analyze`` writes
the memo to ``outputs/p0/decision_memo.md``; its negative-control block holds the "rho vs
strongest nuisance" values of Table VIII (rows 1-2) and its decision the alignment verdict of
Section IV-E7.
"""

from __future__ import annotations

import numpy as np

from cfgid.align.gate import GateResult
from cfgid.align.regression import RegressionResult


def build_decision_memo(
    *,
    decision: GateResult,
    regression: RegressionResult,
    provenance: dict[str, str],
    notes: list[str] | None = None,
    sections: list[tuple[str, list[str]]] | None = None,
) -> str:
    """Render the main-grid alignment diagnostic memo (markdown). Pure text.

    ``sections`` appends extra evidence tables as (title, pre-formatted lines) blocks.
    """
    lines: list[str] = [
        "# Main-grid alignment diagnostic",
        "",
        f"## Decision: {decision.decision.upper()}",
        "",
        "### Reasons",
    ]
    lines += [f"- {r}" for r in decision.reasons] or ["- (none)"]
    lines += [
        "",
        "### Knee-vs-w* alignment (predictive track)",
        f"- slope: {regression.slope:.4f}  "
        f"CI[{regression.slope_ci_lo:.4f}, {regression.slope_ci_hi:.4f}]",
        f"- correlation: {regression.corr:.4f}  (n_cells={regression.n})",
        "",
        "### Parameter provenance",
    ]
    lines += [f"- {k}: {v}" for k, v in provenance.items()]
    for title, body in sections or []:
        lines += ["", f"### {title}"] + (body or ["- (none)"])
    if notes:
        lines += ["", "### Notes"] + [f"- {n}" for n in notes]
    return "\n".join(lines) + "\n"


def save_alignment_scatter(wstars: np.ndarray, knees: np.ndarray, out_path: str) -> bool:
    """Scatter empirical knee vs analytic w*. Returns False (no-op) if matplotlib is unavailable."""
    try:
        import matplotlib  # type: ignore[import-not-found]

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except ImportError:
        return False
    x = np.asarray(wstars, dtype=np.float64).reshape(-1)
    y = np.asarray(knees, dtype=np.float64).reshape(-1)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(x, y, s=24)
    lim = [float(min(x.min(), y.min())), float(max(x.max(), y.max()))]
    ax.plot(lim, lim, "--", color="gray", linewidth=1)
    ax.set_xlabel("analytic w*")
    ax.set_ylabel("empirical separability knee (w)")
    ax.set_title("p0 alignment: empirical knee vs analytic w*")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True
