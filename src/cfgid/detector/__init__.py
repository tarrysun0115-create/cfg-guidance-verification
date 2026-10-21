"""Detectors: empirical ROC/AUC primitives, the rho-sufficiency probe, the white-box LR test.

``empirical``: tie-correct Mann-Whitney AUC and ROC curves (the half-tie AUC convention of
Appendix C). ``sufficiency``: a shallow Fisher-LDA probe on the full inversion residual vs scalar
rho (the "Rho sufficiency" block of ``outputs/p0/decision_memo.md``). ``whitebox_lr`` (imported
directly): the Gaussian likelihood-ratio detector with an analytic FPR threshold, used by the
diagnostic scripts ``run_whitebox_lr.py`` and ``cross_decoder.py``.
"""

from cfgid.detector.empirical import auc, roc_curve
from cfgid.detector.sufficiency import RhoSufficiency, rho_sufficiency_gap

__all__ = ["auc", "roc_curve", "RhoSufficiency", "rho_sufficiency_gap"]
