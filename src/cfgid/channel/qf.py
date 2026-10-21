"""JPEG quality-factor (QF) helpers: DQT table parsing and analytic quantization-noise terms.

``v(sigma, QF)`` is computed from the **actual** quantization tables (uniform-quantization
noise variance step^2/12), not a scalar "equivalent QF" (the latter is a display-only field
reverse-mapped from the luma table). This is the controlled-condition model
(``analytic_no_calibration`` provenance); it does not model the VAE, the 8-bit save or encoder
specifics. Consumers: ``params.vmodel`` (analytic v(QF)) and ``params.gsigma`` (JPEG-MTF gain).
The paper's main grid is uncompressed; JPEG enters only the post-processing evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# JPEG zigzag scan order: ZIGZAG[k] is the flat 8x8 index that receives the k-th coefficient.
_ZIGZAG = np.array(
    [
        0,
        1,
        8,
        16,
        9,
        2,
        3,
        10,
        17,
        24,
        32,
        25,
        18,
        11,
        4,
        5,
        12,
        19,
        26,
        33,
        40,
        48,
        41,
        34,
        27,
        20,
        13,
        6,
        7,
        14,
        21,
        28,
        35,
        42,
        49,
        56,
        57,
        50,
        43,
        36,
        29,
        22,
        15,
        23,
        30,
        37,
        44,
        51,
        58,
        59,
        52,
        45,
        38,
        31,
        39,
        46,
        53,
        60,
        61,
        54,
        47,
        55,
        62,
        63,
    ],
    dtype=np.int64,
)

# Annex-K standard JPEG luminance quantization table (QF=50 baseline), used only to reverse-map an
# approximate "equivalent QF" for display.
_STD_LUMA = np.array(
    [
        16,
        11,
        10,
        16,
        24,
        40,
        51,
        61,
        12,
        12,
        14,
        19,
        26,
        58,
        60,
        55,
        14,
        13,
        16,
        24,
        40,
        57,
        69,
        56,
        14,
        17,
        22,
        29,
        51,
        87,
        80,
        62,
        18,
        22,
        37,
        56,
        68,
        109,
        103,
        77,
        24,
        35,
        55,
        64,
        81,
        104,
        113,
        92,
        49,
        64,
        78,
        87,
        103,
        121,
        120,
        101,
        72,
        92,
        95,
        98,
        112,
        100,
        103,
        99,
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class DQTInfo:
    tables: tuple[np.ndarray, ...]  # one or more 8x8 int quantization tables (luma first)

    @property
    def luma(self) -> np.ndarray:
        return self.tables[0]


def _dezigzag(values: list[int]) -> np.ndarray:
    if len(values) != 64:
        raise ValueError(f"quant table must have 64 entries, got {len(values)}")
    table = np.zeros(64, dtype=np.int64)
    table[_ZIGZAG] = np.asarray(values, dtype=np.int64)
    return table.reshape(8, 8)


def parse_dqt(data: bytes) -> DQTInfo:
    """Parse all DQT tables from JPEG bytes (tables are defined before the SOS scan)."""
    if len(data) < 2 or data[0] != 0xFF or data[1] != 0xD8:
        raise ValueError("not a JPEG byte stream (missing SOI marker FFD8)")
    n = len(data)
    i = 2
    tables: dict[int, np.ndarray] = {}
    while i + 1 < n:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xD9, 0xDA):  # EOI / SOS -> quant tables already seen
            break
        if marker == 0xFF:  # fill byte (T.81 B.1.1.2): a marker may be preceded by any number of
            # 0xFF fill bytes. Skip only ONE so the next byte is re-examined as a marker code; a +2
            # skip would eat the leading 0xFF of the genuine marker that follows and, with an odd
            # fill run, silently mis-parse or drop the DQT.
            i += 1
            continue
        if marker == 0x01 or 0xD0 <= marker <= 0xD7:  # standalone markers (TEM / RSTn): no payload
            i += 2
            continue
        if i + 4 > n:
            break
        seg_len = (data[i + 2] << 8) | data[i + 3]
        if seg_len < 2:
            raise ValueError(f"invalid JPEG segment length {seg_len} at offset {i}")
        seg_end = i + 2 + seg_len
        if seg_end > n:
            raise ValueError(f"truncated JPEG segment at offset {i}")
        if marker == 0xDB:  # DQT
            p = i + 4
            while p < seg_end:
                pq, tq = data[p] >> 4, data[p] & 0x0F
                p += 1
                if pq == 0:
                    if p + 64 > seg_end:
                        raise ValueError("truncated 8-bit DQT table")
                    vals = list(data[p : p + 64])
                    p += 64
                elif pq == 1:
                    if p + 128 > seg_end:
                        raise ValueError("truncated 16-bit DQT table")
                    vals = [(data[p + 2 * k] << 8) | data[p + 2 * k + 1] for k in range(64)]
                    p += 128
                else:
                    raise ValueError(f"invalid DQT precision {pq}")
                tables[tq] = _dezigzag(vals)
        i = seg_end
    if not tables:
        raise ValueError("no DQT (quantization table) marker found in JPEG stream")
    return DQTInfo(tables=tuple(tables[k] for k in sorted(tables)))


def dqt_for_quality(qf: int) -> DQTInfo:
    """Reconstruct the STANDARD JPEG luma quantization table for integer quality ``qf`` (1..100).

    Uses the canonical libjpeg quality->scaling map (integer ``scale = 5000 // qf`` for qf<50,
    else ``200 - 2*qf``) and integer ``q[i] = clip((std_luma[i]*scale + 50)//100, 1, 255)``.
    This makes ``v(QF)`` fully analytic from the integer quality used at generation time — no need
    to persist the encoder's actual DQT bytes.
    """
    if not (isinstance(qf, (int, np.integer)) and 1 <= int(qf) <= 100):
        raise ValueError(f"qf must be an integer in [1, 100], got {qf}")
    q = int(qf)
    scale = 5000 // q if q < 50 else 200 - 2 * q
    std = _STD_LUMA.astype(np.int64)
    table = (std * scale + 50) // 100
    table = np.clip(table, 1, 255).astype(np.int64)
    return DQTInfo(tables=(table.reshape(8, 8),))


def jpeg_quant_noise_var(dqt: DQTInfo) -> float:
    """JPEG noise contribution to v: mean step^2/12 (uniform quant) over the luma table."""
    steps = dqt.luma.astype(np.float64)
    return float(np.mean(np.square(steps)) / 12.0)


# Zigzag indices of the high-frequency DCT band (drop the DC + lowest-frequency coefficients).
# Used by the JPEG-MTF attenuation proxy: the rho channel lives on high-band texture, so heavier
# high-band quantization (lower QF) attenuates the guidance signal more.
_HIGH_BAND_START = 32  # last 32 of the 64 zigzag positions = high-frequency half

# FIXED reference step (independent of QF): the standard table's mean high-band step (the QF=50
# baseline). The attenuation proxy compares a table's high-band steps against this constant, so it
# is monotone in QF — a per-table DC reference would itself scale with QF and break monotonicity.
_MTF_REF_STEP = float(np.mean(_STD_LUMA.reshape(64)[_ZIGZAG][_HIGH_BAND_START:]))


def jpeg_mtf_attenuation(dqt: DQTInfo) -> float:
    """High-band signal-preservation factor in (0, 1): ``ref / (ref + mean_high_band_step)``.

    A monotone-decreasing, bounded proxy for JPEG's attenuation of the high-frequency texture band
    the rho statistic uses: larger high-band quantization steps (lower QF) -> smaller factor (more
    attenuation). The reference ``_MTF_REF_STEP`` is the FIXED standard high-band mean (QF=50),
    so a QF=50 table gives ~0.5 and lighter compression -> closer to 1. PROVISIONAL (heuristic MTF
    proxy, like ``gsigma.plugin_gain``); its absolute scale is not calibrated.
    """
    flat = dqt.luma.astype(np.float64).reshape(64)[_ZIGZAG]  # zigzag order: index 0 = DC
    mean_high = float(np.mean(flat[_HIGH_BAND_START:]))
    if not (np.isfinite(mean_high) and mean_high >= 0.0):
        raise ValueError(f"invalid DQT high-band steps for MTF proxy (mean_high={mean_high})")
    return float(_MTF_REF_STEP / (_MTF_REF_STEP + mean_high))


def equivalent_qf(dqt: DQTInfo) -> float:
    """Display-only approximate quality factor reverse-mapped from the luma table (1..100)."""
    q = dqt.luma.astype(np.float64).reshape(64)
    ratios = q / _STD_LUMA
    usable = ratios[(q > 1) & (q < 255)]  # drop clamped entries
    scale = float(np.median(usable)) * 100.0 if usable.size else float(np.median(ratios)) * 100.0
    if scale <= 0:
        return 100.0
    qf = 5000.0 / scale if scale > 100.0 else (200.0 - scale) / 2.0
    return float(min(100.0, max(1.0, qf)))
