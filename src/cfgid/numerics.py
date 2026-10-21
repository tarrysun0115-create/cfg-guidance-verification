"""Shared numeric guards: non-finite values (NaN/Inf) raise instead of propagating silently.

Used across the pipeline (feature store, align/bounds helpers, analysis scripts) so that a
diverged inversion or a corrupted artifact fails loudly. ``loads_strict_json`` is the strict JSON
reader the analysis scripts use for their input artifacts.
"""

from __future__ import annotations

import json
import math
from typing import NoReturn

import numpy as np


def as_finite_float32(name: str, values: np.ndarray, *, allow_nan: bool = False) -> np.ndarray:
    """Cast to float32 and reject non-finite values (Inf always; NaN unless ``allow_nan``)."""
    try:
        out = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    bad = np.isinf(out) if allow_nan else ~np.isfinite(out)
    if np.any(bad):
        raise ValueError(f"{name} contains non-finite values (NaN/Inf)")
    return out


def finite_float(name: str, value: object, *, positive: bool = False) -> float:
    """Coerce one JSON/numpy scalar to finite float, rejecting bool and NaN/Inf."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (str, int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be finite numeric")
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite numeric") from exc
    if not math.isfinite(out) or (positive and out <= 0.0):
        want = "positive finite" if positive else "finite"
        raise ValueError(f"{name} must be {want}, got {out}")
    return out


def finite_or_none(value: object, *, positive: bool = False) -> float | None:
    """Return a finite scalar float, or None for nonnumeric/nonfinite/invalid values."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (str, int, float, np.integer, np.floating)
    ):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(out) or (positive and out <= 0.0):
        return None
    return out


def auc_gate_passed(value: object, *, expected: object, tol: object) -> bool:
    """Finite-only absolute-tolerance gate for self-check AUCs and gate parameters."""
    auc_value = finite_or_none(value)
    expected_value = finite_or_none(expected)
    tol_value = finite_or_none(tol)
    return (
        auc_value is not None
        and expected_value is not None
        and tol_value is not None
        and tol_value >= 0.0
        and abs(auc_value - expected_value) <= tol_value
    )


def _reject_json_constant(token: str) -> NoReturn:
    raise ValueError(f"JSON contains non-finite token {token}; refusing to parse invalid JSON")


def _finite_json_float(token: str) -> float:
    """parse_float hook: reject finite-looking literals that overflow IEEE-754 double to +-inf."""
    value = float(token)
    if not math.isfinite(value):
        raise ValueError(
            f"JSON contains non-finite numeric literal {token} (overflows to {value}); "
            "refusing to parse invalid JSON"
        )
    return value


def _finite_json_int(token: str) -> int:
    """parse_int hook: reject integer literals too large for downstream float coercions."""
    value = int(token)
    try:
        as_float = float(value)
    except OverflowError as exc:
        raise ValueError(
            f"JSON contains integer literal {token} that overflows IEEE-754 double; "
            "refusing to parse invalid artifact"
        ) from exc
    if not math.isfinite(as_float):
        raise ValueError(
            f"JSON contains non-finite integer literal {token} (overflows to {as_float}); "
            "refusing to parse invalid artifact"
        )
    return value


def loads_strict_json(text: str) -> object:
    """Parse JSON while rejecting EVERY non-finite value the default ``json.loads`` accepts.

    Two leaks are closed: the bareword tokens ``NaN`` / ``Infinity`` / ``-Infinity`` (via
    ``parse_constant``) AND a finite-looking numeric literal that overflows a double to ``inf``
    (e.g. ``1e400``, which Python's number scanner converts to ``inf`` WITHOUT invoking
    ``parse_constant``; closed here via ``parse_float``). Integer literals that would later
    overflow the artifact readers' float coercions are rejected via ``parse_int``. All raise
    ``ValueError`` so a corrupted or hand-edited artifact fails loud rather than carrying a
    non-finite value into analysis.
    """
    return json.loads(
        text,
        parse_constant=_reject_json_constant,
        parse_float=_finite_json_float,
        parse_int=_finite_json_int,
    )
