"""Oracle-trace consistency check (inversion-artifact diagnostic).

When generation is run with a trace (``generate_image_with_trace`` in the backends), the true
per-branch noise predictions at one generation step are stored; the inversion-recovered per-patch
rho is compared with the oracle rho computed from that trace. Poor agreement would indicate that
the measured rho is an artifact of the inverter on different-CFG trajectories rather than an
image-channel signal. Used by ``cfgid.sweep.evidence.oracle_evidence`` (H1 rows, unblurred,
uncompressed) as a diagnostic, not as a reported paper quantity. Pure numpy / finite-guarded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cfgid.numerics import as_finite_float32


@dataclass(frozen=True)
class TraceConsistency:
    max_abs_err: float
    mean_abs_err: float
    pearson_r: float

    def is_consistent(self, *, abs_tol: float, min_r: float) -> bool:
        return self.max_abs_err <= abs_tol and self.pearson_r >= min_r


def rho_trace_consistency(inv_rho: np.ndarray, oracle_rho: np.ndarray) -> TraceConsistency:
    """Compare inversion-recovered per-patch rho to the oracle (generation-trajectory) rho."""
    a = as_finite_float32("inv_rho", inv_rho).reshape(-1)
    b = as_finite_float32("oracle_rho", oracle_rho).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"inv_rho {a.shape} != oracle_rho {b.shape}")
    if a.size == 0:
        raise ValueError("rho_trace_consistency received empty arrays")
    err = np.abs(a - b)
    # Pearson r is undefined for constant inputs; fall back to exact agreement (1.0) or none (0.0).
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        r = 1.0 if np.allclose(a, b) else 0.0
    else:
        r = float(np.corrcoef(a, b)[0, 1])
    return TraceConsistency(float(err.max()), float(err.mean()), r)
