"""Master per-cell table: w* bounds, empirical knee and sign-aligned rho AUC per alignment cell.

Produces: no printed paper number directly (intermediate product). ``master_table.json`` supplies
the per-cell sigma_eff that ``psf_grounding_features.py`` / ``feature_comparison.py`` use for the
exploratory PSF-order column of Table X (Section IV-E3), and the per-cell rho AUC_1 at sigma_blur=0
that ``cross_model_matrix_analyze.py`` uses as a loose reference.

Inputs: ``outputs/p0/alignment_cells.json`` (from ``cfgid.sweep.analyze``),
``outputs/p0/pooled_neff_scaling.json`` and ``outputs/p0/<decoder>/p0_table.json``. Pure numpy,
CPU only; no inversion, no model weights.

Outputs:
  * ``outputs/p0/master_table.json`` -- {"meta": {...}, "rows": [...]} (rows sorted by decoder, sb).
  * ``outputs/p0/master_table.md``   -- a GitHub-flavored markdown table (caption + header + one
    row per cell), numbers rounded to 3 significant figures, JSON null shown as the em-dash '—'
    (U+2014).

Usage: python scripts/analysis/master_table.py [--features-root outputs/p0] [--out-dir outputs/p0]

It joins three persisted products into one row per (decoder, sigma_blur) alignment cell:

  * alignment_cells.json -- the 12 distinct-VAE (decoder, sigma_blur) cells with their analytic /
    calibrated / covariance-weighted w* (Eq. (15)), the empirical knee + CI, E_tex, v_hat, g_sigma
    and the censoring flag.
  * pooled_neff_scaling.json -- the favorable-subset pooled sqrt(N_eff) result; its
    ``alive_cells`` list names the cells that passed the held-out sign-consistent gate (and at which
    w / sign), and its ``exponent`` is surfaced in the table meta.
  * the p0_table.json rows -- to RECOMPUTE the per-cell sign-aligned single-image AUC and sign (so
    this script does not depend on any other reanalysis output). For each admitted cell record the
    sign-aligned eval AUC is auc(sign*[zt1_eval, zt0_eval], labels); records are mapped back to
    cells by (decoder, sigma_blur).

A cell with no admitted record is reported with alive=False and null (JSON null) AUC / sign /
alive_w. Strict JSON throughout (allow_nan=False). Predictive track only: g_sigma is read from the
persisted cells, never fitted on H1 labels here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cfgid.detector.empirical import auc
from cfgid.numerics import loads_strict_json
from cfgid.paths import outputs_dir
from cfgid.sweep.neff_scaling import (
    _group_rows_by_cell,
    _load_p0_rows,
    filter_core_none_qf_rows,
    filter_to_first_n_prompts,
)
from cfgid.sweep.pooled_neff_scaling import CellRecord, collect_alive_cell_records

# Held-out sign-consistent alive gate; should match the pooled module so this table's alive
# set is the same as outputs/p0/pooled_neff_scaling.json (the default there is also 1.5). The pooled
# file does not record its own alive_t, so run() cross-checks its alive_cells against the recompute
# at this ALIVE_T and emits meta `pooled_alive_set_consistent` (+ a warning) on any mismatch.
ALIVE_T = 1.5

# The per-cell fields of the table (see _COLUMNS). Numeric scalars passed straight through from the
# alignment cell (validated finite or emitted as null); the detection block (alive/alive_w/sign/
# auc1/n_h0/n_h1) is recomputed from rows.
_NULL = "—"  # em-dash shown for a JSON-null cell in the markdown render

# Markdown column order + headers (kept in one place so render_markdown + the caption agree).
_COLUMNS: tuple[tuple[str, str], ...] = (
    ("decoder", "decoder"),
    ("sigma_blur", "sigma_blur"),
    ("sigma_eff", "sigma_eff"),
    ("alive", "alive"),
    ("alive_w", "alive_w"),
    ("sign", "sign"),
    ("auc1", "auc1"),
    ("n_h0", "n_h0"),
    ("n_h1", "n_h1"),
    ("knee", "knee"),
    ("knee_ci_lo", "knee_ci_lo"),
    ("knee_ci_hi", "knee_ci_hi"),
    ("wstar", "wstar"),
    ("wstar_calibrated", "wstar_calibrated"),
    ("wstar_cov", "wstar_cov"),
    ("e_tex", "e_tex"),
    ("v_hat", "v_hat"),
    ("g_sigma", "g_sigma"),
    ("censored", "censored"),
)


def _finite_or_none(value: object) -> float | None:
    """A finite float, or None for a missing / non-finite / non-numeric cell.

    Booleans are NOT numbers here (a stray bool in a numeric slot is a data error, not 0/1), so they
    return None. Used to pull alignment-cell scalars without ever letting a NaN/Inf reach json.dump.
    """
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    out = float(value)
    return out if np.isfinite(out) else None


def _require_bool(value: object, name: str) -> bool:
    """Strict bool extractor: the 'censored' flag must be a genuine boolean, fail loud otherwise."""
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"alignment-cell field '{name}' must be a boolean, got {value!r}")
    return bool(value)


def _require_str(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"alignment-cell field '{name}' must be a non-empty string, got {value!r}")
    return value


def _cell_key(decoder: str, sigma_blur: float) -> tuple[str, float]:
    """Join key shared by alignment cells and recomputed records: (decoder, rounded sigma_blur).

    sigma_blur is rounded to a fixed grid resolution so an alignment cell's stored float (e.g. 8.0)
    and a record's float compare equal under any harmless float drift, without a fragile == on raw
    floats. Note this round-to-6-DECIMALS key is its own join space; it is NOT meant to coincide
    with the cell grouper's ``f"sb={x:g}"`` key (6 SIGNIFICANT figures). For the actual main grid
    (small integer sigma_blur 0/8/16/32) both agree; the two schemes could only diverge for a
    sigma_blur with many significant digits, which this grid never has.
    """
    return decoder, round(float(sigma_blur), 6)


def recompute_alive_records(
    features_root: Path, *, max_prompts: int | None = None
) -> dict[tuple[str, float], dict]:
    """Recompute the per-cell sign-aligned single-image AUC + sign from the persisted rows.

    Loads every <decoder>/p0_table.json, routes through the shared core/QF filter, groups into
    (decoder, sigma_blur) cells, and runs the SAME held-out sign-consistent alive gate the
    pooled module uses (collect_alive_cell_records). For each survivor the sign-aligned fit/eval AUC
    is auc(concat(sign*zt1_eval, sign*zt0_eval), labels). Returns a dict keyed by (decoder, sb) so
    build_row can attach detection stats to its alignment cell; cells with no record are simply
    absent (build_row then marks them alive=False with null stats).

    ``max_prompts`` (None -> all) subsamples to the first-N prompts before the alive gate so this
    table can be recomputed at a convergence point; the per-prompt subset is nested.
    """
    rows: list[dict] = []
    for table_path in sorted(features_root.glob("*/p0_table.json")):
        rows.extend(filter_core_none_qf_rows(_load_p0_rows(table_path)))
    rows = filter_to_first_n_prompts(rows, max_prompts)
    if not rows:
        return {}
    cells = _group_rows_by_cell(rows)
    records: list[CellRecord] = collect_alive_cell_records(cells, alive_t=ALIVE_T)
    out: dict[tuple[str, float], dict] = {}
    for rec in records:
        auc1 = _single_image_auc(rec)
        out[_cell_key(rec.decoder, rec.sigma_blur)] = {
            "alive_w": int(rec.w),
            "sign": int(rec.sign),
            "auc1": auc1,
            "n_h0": int(rec.zt0_eval.size),
            "n_h1": int(rec.zt1_eval.size),
        }
    return out


def _single_image_auc(rec: CellRecord) -> float | None:
    """Sign-aligned single-image TEST-eval AUC for one cell record, or None if not computable.

    Aligns the cell's UNSIGNED eval z by its FIT sign so a correct sign sends H1 to a positive
    shift, then AUCs the pooled eval-half (H1 vs H0). Returns None when a class is empty (auc would
    raise) so the join stays null-safe rather than crashing on a degenerate cell.
    """
    z1 = rec.sign * np.asarray(rec.zt1_eval, dtype=np.float64).reshape(-1)
    z0 = rec.sign * np.asarray(rec.zt0_eval, dtype=np.float64).reshape(-1)
    if z1.size == 0 or z0.size == 0:
        return None
    scores = np.concatenate([z1, z0])
    labels = np.concatenate([np.ones(z1.size, dtype=int), np.zeros(z0.size, dtype=int)])
    if not np.all(np.isfinite(scores)):
        raise ValueError("recomputed eval z-scores contain non-finite values")
    value = float(auc(scores, labels))
    return value if np.isfinite(value) else None


def build_row(cell: dict, detection: dict | None) -> dict:
    """Assemble one master-table row from an alignment cell + its (optional) recomputed detection.

    ``cell`` is one record from alignment_cells.json; ``detection`` is the recompute_alive_records
    entry for the same (decoder, sigma_blur) or None when the cell has no alive record. Numeric
    alignment fields pass through _finite_or_none (non-finite -> JSON null); ``censored`` is a
    strict bool. When ``detection`` is None the cell is alive=False with null alive_w / sign /
    auc1 / n_h0 / n_h1; otherwise those come from the recomputed record.
    """
    decoder = _require_str(cell.get("decoder"), "decoder")
    sigma_blur = _finite_or_none(cell.get("sigma_blur"))
    if sigma_blur is None:
        raise ValueError(f"alignment-cell '{decoder}' has a non-finite sigma_blur")
    alive = detection is not None
    return {
        "decoder": decoder,
        "sigma_blur": sigma_blur,
        "sigma_eff": _finite_or_none(cell.get("sigma_eff")),
        "alive": alive,
        "alive_w": (detection["alive_w"] if alive else None),
        "sign": (detection["sign"] if alive else None),
        "auc1": (detection["auc1"] if alive else None),
        "n_h0": (detection["n_h0"] if alive else None),
        "n_h1": (detection["n_h1"] if alive else None),
        "knee": _finite_or_none(cell.get("knee")),
        "knee_ci_lo": _finite_or_none(cell.get("knee_ci_lo")),
        "knee_ci_hi": _finite_or_none(cell.get("knee_ci_hi")),
        "wstar": _finite_or_none(cell.get("wstar")),
        "wstar_calibrated": _finite_or_none(cell.get("wstar_calibrated")),
        "wstar_cov": _finite_or_none(cell.get("wstar_cov")),
        "e_tex": _finite_or_none(cell.get("e_tex")),
        "v_hat": _finite_or_none(cell.get("v_hat")),
        "g_sigma": _finite_or_none(cell.get("g_sigma")),
        "censored": _require_bool(cell.get("censored"), "censored"),
    }


def _fmt_cell(value: object) -> str:
    """Render one value for the markdown table: null -> em-dash, float -> 3 significant figures."""
    if value is None:
        return _NULL
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        v = float(value)
        if not np.isfinite(v):
            return _NULL
        if v == 0.0:
            return "0"
        return f"{v:.3g}"
    return str(value)


def render_markdown(rows: list[dict], *, caption: str) -> str:
    """A GitHub-flavored markdown table: a caption line, one header row, one row per input cell.

    Each numeric is rounded to 3 significant figures (via _fmt_cell); a JSON-null field is shown as
    the em-dash. The header / column order is fixed by _COLUMNS so the caption's source note and the
    columns stay in lockstep. The caption is emitted as a markdown blockquote above the table.
    """
    headers = [header for _key, header in _COLUMNS]
    lines = [
        f"> {caption}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        cells = [_fmt_cell(row.get(key)) for key, _header in _COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _load_pooled_meta(features_root: Path) -> tuple[float | None, list[str]]:
    """Read the pooled exponent + alive_cells list from pooled_neff_scaling.json (if present).

    Returns (pooled_exponent_or_None, alive_cells_list). The exponent is surfaced in the table meta;
    the alive_cells list is carried verbatim so a reader can see WHICH cells (and at which w / sign)
    the pooled √N_eff result rests on, alongside our independently-recomputed per-cell AUC.
    """
    pooled_path = features_root / "pooled_neff_scaling.json"
    if not pooled_path.exists():
        return None, []
    payload = loads_strict_json(pooled_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{pooled_path}: expected a JSON object")
    exponent = _finite_or_none(payload.get("exponent"))
    alive_raw = payload.get("alive_cells", [])
    if not isinstance(alive_raw, list):
        raise ValueError(f"{pooled_path}: 'alive_cells' must be a list")
    alive_cells = [str(c) for c in alive_raw]
    return exponent, alive_cells


def run(features_root: Path, out_dir: Path, *, max_prompts: int | None = None) -> int:
    """Assemble the master per-cell table from existing outputs/p0 products + recomputed AUCs.

    ``max_prompts`` (None -> all) recomputes the per-cell detection block on the first-N prompts
    (for a prompt-N convergence point). The pooled result (exponent + alive_cells) is read
    VERBATIM from pooled_neff_scaling.json, which is at FULL N -- so under a subsample the
    pooled-vs-recompute cross-check is apples-to-oranges and is reported as N/A (None) rather
    than a spurious mismatch.
    """
    cells_path = features_root / "alignment_cells.json"
    if not cells_path.exists():
        print(f"[skip] no alignment_cells.json under {features_root}; nothing to aggregate.")
        return 0
    raw = loads_strict_json(cells_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{cells_path}: alignment_cells.json must contain a list of cells")
    if not raw:
        print(f"[skip] alignment_cells.json under {features_root} is empty.")
        return 0

    detection = recompute_alive_records(features_root, max_prompts=max_prompts)
    pooled_exponent, alive_cells = _load_pooled_meta(features_root)

    rows: list[dict] = []
    for cell in raw:
        if not isinstance(cell, dict):
            raise ValueError(f"{cells_path}: each alignment cell must be a JSON object")
        decoder = _require_str(cell.get("decoder"), "decoder")
        sigma_blur = _finite_or_none(cell.get("sigma_blur"))
        if sigma_blur is None:
            raise ValueError(f"alignment-cell '{decoder}' has a non-finite sigma_blur")
        rows.append(build_row(cell, detection.get(_cell_key(decoder, sigma_blur))))
    rows.sort(key=lambda r: (r["decoder"], r["sigma_blur"]))

    n_alive = sum(1 for r in rows if r["alive"])
    # Coherence guard: the per-cell `alive` flags are recomputed here at ALIVE_T, while
    # `pooled_exponent` + `pooled_alive_cells` are read VERBATIM from pooled_neff_scaling.json
    # (which does not record the alive_t it was generated with). If that file was produced with a
    # different alive_t or is stale, its alive set differs from this recompute -- detect and flag it
    # so the table never silently presents a pooled result over a DIFFERENT selection.
    recomputed_alive_labels = {
        f"{dec}/sb={sb:g}/w={d['alive_w']}/sign={'+' if d['sign'] > 0 else '-'}"
        for (dec, sb), d in detection.items()
    }
    # Under a prompt subsample the recompute is at N=max_prompts while the pooled file is at full N;
    # the cross-check is then meaningless -> N/A (None), and the warning below is suppressed.
    pooled_alive_set_consistent: bool | None
    if max_prompts is not None:
        pooled_alive_set_consistent = None
    else:
        pooled_alive_set_consistent = (not alive_cells) or (
            set(alive_cells) == recomputed_alive_labels
        )
    # Orphan guard: every recomputed alive record should attach to an alignment-cell row. A
    # detection key absent from alignment_cells.json contributes to recomputed_alive_labels (and so
    # the pooled cross-check) but produces NO row and is NOT counted in n_alive -- which would let
    # the table advertise an alive count that omits a cell the pooled result rests on.
    alignment_keys = {_cell_key(r["decoder"], r["sigma_blur"]) for r in rows}
    orphan_alive = sorted(
        f"{dec}/sb={sb:g}" for (dec, sb) in detection if (dec, sb) not in alignment_keys
    )
    rows_cover_recompute = not orphan_alive
    meta = {
        "alive_t": ALIVE_T,
        "max_prompts": max_prompts,
        "pooled_exponent": pooled_exponent,
        "n_alive": n_alive,
        "n_cells": len(rows),
        "pooled_alive_cells": alive_cells,
        "pooled_alive_set_consistent": pooled_alive_set_consistent,
        "rows_cover_recompute": rows_cover_recompute,
        "source": "alignment_cells.json + pooled_neff_scaling.json + recomputed per-cell AUC",
    }
    if pooled_alive_set_consistent is False:
        print(
            "[warn] pooled_neff_scaling.json alive_cells differ from the recomputed "
            f"alive_t={ALIVE_T:g} selection; the pooled exponent/alive list and the per-cell rows "
            "rest on DIFFERENT selections (regenerate the pooled file at alive_t="
            f"{ALIVE_T:g})."
        )
    if orphan_alive:
        print(
            f"[warn] recomputed alive cell(s) {orphan_alive} have no alignment_cells.json row, so "
            f"they are absent from the table and excluded from n_alive={n_alive} though alive."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "master_table.json"
    json_path.write_text(
        json.dumps({"meta": meta, "rows": rows}, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    subsample_note = (
        f" Per-cell detection recomputed on the first {max_prompts} prompts; "
        "the pooled exponent is read at FULL N (cross-check N/A)."
        if max_prompts is not None
        else ""
    )
    caption = (
        f"Master per-cell table. Source: alignment_cells.json + pooled_neff_scaling.json + "
        f"recomputed sign-aligned per-cell AUC (alive_t={ALIVE_T:g}; "
        f"{n_alive}/{len(rows)} cells alive; pooled exponent="
        f"{_fmt_cell(pooled_exponent)}).{subsample_note} Numbers rounded to 3 significant figures; "
        f"'{_NULL}' = not defined."
    )
    md_path = out_dir / "master_table.md"
    md_path.write_text(render_markdown(rows, caption=caption), encoding="utf-8")

    print(f"\nMaster table: {len(rows)} cells ({n_alive} alive at alive_t={ALIVE_T:g})")
    exp = _fmt_cell(pooled_exponent)
    print(f"  pooled sqrt(N_eff) exponent = {exp}  (alive cells: {alive_cells})")
    for r in rows:
        auc1 = _fmt_cell(r["auc1"])
        wstar_c = _fmt_cell(r["wstar_calibrated"])
        knee = _fmt_cell(r["knee"])
        flag = "alive" if r["alive"] else "  -- "
        print(
            f"  [{flag}] {r['decoder']:5s} sb={r['sigma_blur']:>4g}  "
            f"auc1={auc1:>6s}  knee={knee:>6s}  wstar_calib={wstar_c:>8s}"
        )
    print(f"-> {json_path}")
    print(f"-> {md_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="master_table",
        description="Master per-cell table (alignment cells + pooled + recomputed AUC)",
    )
    ap.add_argument(
        "--features-root",
        default=None,
        help="dir holding alignment_cells.json + <decoder>/p0_table.json (default: outputs/p0)",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="output dir for master_table.json / .md (default: outputs/p0)",
    )
    ap.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="recompute per-cell detection on the first N prompts (convergence point); default all",
    )
    args = ap.parse_args(argv)
    features_root = Path(args.features_root) if args.features_root else (outputs_dir() / "p0")
    out_dir = Path(args.out_dir) if args.out_dir else (outputs_dir() / "p0")
    return run(features_root, out_dir, max_prompts=args.max_prompts)


if __name__ == "__main__":
    raise SystemExit(main())
