"""Shared matplotlib figure conventions for cfgid reanalysis / paper figures.

Mirrors ``cfgid.align.report.save_alignment_scatter``: matplotlib is imported lazily
inside ``pyplot()`` with the headless Agg backend forced, and every entry point
degrades gracefully (returns ``None``) when matplotlib is unavailable in a partial
install. No data computation lives here -- callers build a ``Figure`` and hand it to
``save_figure()``, which writes a deterministic PNG under the in-repo ``outputs/figures/``
directory. That directory may hold both paper figures (e.g. Fig. 2(D)) and new diagnostic
plots, so it is not a disposable cache: only the diagnostic files these scripts generate can
be deleted and regenerated. Used by the analysis scripts that
emit diagnostic figures next to their JSON outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cfgid.paths import REPO_ROOT, outputs_dir

# Stable per-decoder colors so the same decoder reads identically across every figure.
DECODER_COLORS: dict[str, str] = {
    "sd15": "#1f77b4",
    "sd21": "#2ca02c",
    "sdxl": "#d62728",
    "flux": "#9467bd",
    "pixart_sigma": "#ff7f0e",
    "hunyuandit": "#8c564b",
}
_DEFAULT_COLOR = "#7f7f7f"


def decoder_color(decoder: str) -> str:
    """Stable color for a decoder; a deterministic gray for any unmapped name."""
    return DECODER_COLORS.get(decoder, _DEFAULT_COLOR)


def figures_dir() -> Path:
    """In-repo figures directory (``outputs/figures/``), created on demand.

    It may hold paper figures as well as new diagnostic plots; only the generated diagnostic
    files are disposable.
    """
    out = outputs_dir() / "figures"
    out.mkdir(parents=True, exist_ok=True)
    return out


def pyplot() -> Any | None:
    """Return ``matplotlib.pyplot`` with the Agg backend forced, or ``None``.

    ``None`` signals matplotlib is not installed (partial server install). Callers
    must guard every figure path on ``plt is not None`` so analysis still emits its
    JSON data product when plotting is unavailable -- the figure is a convenience,
    never a hard dependency of the numeric result.
    """
    try:
        import matplotlib  # type: ignore[import-not-found]

        matplotlib.use("Agg")
        # Embed text as TrueType (fonttype 42), never Type 3: IEEE PDF eXpress rejects
        # Type 3 fonts, which are matplotlib's PDF/PS default.
        matplotlib.rcParams["pdf.fonttype"] = 42
        matplotlib.rcParams["ps.fonttype"] = 42
        # Match the IEEEtran body font: Times-like serif with STIX math (STIXGeneral
        # ships with matplotlib, so renders identically on macOS and Linux).
        matplotlib.rcParams["font.family"] = "serif"
        matplotlib.rcParams["font.serif"] = [
            "STIXGeneral", "Times New Roman", "Nimbus Roman", "Times",
        ]
        matplotlib.rcParams["mathtext.fontset"] = "stix"
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except ImportError:
        return None
    return plt


def save_figure(
    fig: Any, name: str, *, dpi: int = 600, paper_pdf: bool = False
) -> Path | None:
    """Save ``fig`` to ``outputs/figures/<name>.png`` and close it.

    Writes a high-resolution raster (``<name>.png`` at ``dpi``) for preview and data
    inspection. By default this sink emits no PDF, so a script re-run cannot clobber a
    hand-rendered paper figure. Pass ``paper_pdf=True`` for figures whose camera-ready PDF
    IS script-generated: the same figure is then also written to ``figures/<name>.pdf`` at the
    repo root (only if that directory exists; no script here passes ``paper_pdf=True``).
    Returns the PNG path, or ``None`` if ``fig`` is ``None`` (so a caller that skipped plotting
    because matplotlib was absent can forward that ``None`` through). ``name`` must be a bare
    stem (no path separators / no extension) so figures cannot escape the sink.
    """
    if fig is None:
        return None
    if not name or "/" in name or "\\" in name or "." in name:
        raise ValueError(
            f"figure name must be a bare stem (no path separators, no extension/dot), got {name!r}"
        )
    path = figures_dir() / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.01)
    if paper_pdf:
        paper_dir = REPO_ROOT / "figures"
        if paper_dir.is_dir():
            fig.savefig(paper_dir / f"{name}.pdf", bbox_inches="tight", pad_inches=0.01)
    # PNG only (raster, so no font-embedding / Type-3 concern); the pyplot() fonttype=42
    # setting still applies to any PDF a caller renders explicitly.
    plt = pyplot()
    if plt is not None:
        plt.close(fig)
    return path
