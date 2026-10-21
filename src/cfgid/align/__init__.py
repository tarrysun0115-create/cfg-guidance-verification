"""Knee-vs-w* alignment: empirical ROC -> knee -> alignment regression -> preregistered gate.

Tests whether the empirical separability knee of each cell's AUC-vs-w curve moves WITH the
analytic delta-deniability threshold w* (Eqs. (15), (38)) across (decoder, sigma_blur) cells, on
the predictive track (g_sigma estimated without H1 labels), with rho not weaker than the nuisance
baselines and close to the full residual. Three-state: pass / fail / inconclusive.

Driven by ``cfgid.sweep.analyze``, which writes ``outputs/p0/decision_memo.md`` and
``outputs/p0/alignment_cells.json``. The main-grid verdict is INCONCLUSIVE; the paper
reports the knee alignment as inconclusive (Section IV-E7) and uses w* only as a sensitivity
quantity (Section III-E).
"""

from cfgid.align.gate import GateInputs, GateResult, evaluate_gate
from cfgid.align.knee import extract_knee
from cfgid.align.regression import RegressionResult, fit_alignment
from cfgid.align.roc import auc_curve, cell_auc

__all__ = [
    "cell_auc",
    "auc_curve",
    "extract_knee",
    "fit_alignment",
    "RegressionResult",
    "GateInputs",
    "GateResult",
    "evaluate_gate",
]
