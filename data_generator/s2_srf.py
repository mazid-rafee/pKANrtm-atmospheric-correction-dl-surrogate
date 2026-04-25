import csv
import os
from pathlib import Path
from typing import List, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
# S2A | S2B | S2C — matches folders under data/sentinel_srf/s2_srf/
_MISSION = os.environ.get("S2_MISSION", "S2A")
DATA_DIR = _PROJECT_ROOT / "data" / "sentinel_srf" / "s2_srf" / _MISSION

_SRF_CACHE = {}


def _csv_stem_for_band(band: str) -> str:
    """Map generator band names (B2, …) to extract script filenames (B02, …)."""
    band = band.upper()
    if band == "B8A":
        return "B8A"
    if len(band) == 2 and band.startswith("B") and band[1].isdigit():
        return f"B0{band[1]}"
    return band


def load_srf(band: str) -> Tuple[List[float], List[float]]:
    """
    Load Sentinel-2 SRF for a band.

    Expects CSV under data/sentinel_srf/s2_srf/<S2MISSION>/<stem>.csv with columns:
    wavelength_nm, response
    """
    band = band.upper()
    if band in _SRF_CACHE:
        return _SRF_CACHE[band]

    stem = _csv_stem_for_band(band)
    path = DATA_DIR / f"{stem}.csv"
    if not path.exists():
        raise FileNotFoundError(f"SRF file not found for band {band}: {path}")

    wvl: List[float] = []
    rsp: List[float] = []

    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            # Skip header or commented lines
            if row[0].startswith("#") or row[0].lower() in ("wavelength_nm", "wavelength"):
                continue
            try:
                lam = float(row[0])
                val = float(row[1])
            except Exception:
                continue
            wvl.append(lam)
            rsp.append(val)

    if not wvl:
        raise ValueError(f"Empty SRF for band {band} at {path}")

    # Ensure sorted by wavelength
    pairs = sorted(zip(wvl, rsp), key=lambda p: p[0])
    wvl_sorted, rsp_sorted = zip(*pairs)
    wvl_list = list(wvl_sorted)
    rsp_list = list(rsp_sorted)

    _SRF_CACHE[band] = (wvl_list, rsp_list)
    return wvl_list, rsp_list


def band_wavelength_grid(band: str, step_nm: float = 1.0) -> List[float]:
    """Return a simple wavelength grid spanning the non-zero SRF support."""
    wvl, rsp = load_srf(band)
    support = [lam for lam, r in zip(wvl, rsp) if r > 0.0]
    if not support:
        raise ValueError(f"No positive SRF values for band {band}")
    lam_min = min(support)
    lam_max = max(support)
    if step_nm <= 0:
        raise ValueError("step_nm must be positive")
    n = int((lam_max - lam_min) / step_nm) + 1
    return [lam_min + i * step_nm for i in range(n)]


def _interp1d(x_src: List[float], y_src: List[float], xq: float) -> float:
    """Simple linear interpolation; returns 0 outside the source range."""
    if xq < x_src[0] or xq > x_src[-1]:
        return 0.0
    # Find interval
    lo = 0
    hi = len(x_src) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if x_src[mid] <= xq:
            lo = mid
        else:
            hi = mid
    x0, x1 = x_src[lo], x_src[hi]
    y0, y1 = y_src[lo], y_src[hi]
    if x1 == x0:
        return y0
    t = (xq - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def convolve_to_band(wavelengths_nm: List[float], values: List[float], band: str) -> float:
    """
    SRF-weighted average of 'values' defined at 'wavelengths_nm' for given band.
    """
    if len(wavelengths_nm) != len(values):
        raise ValueError("wavelengths_nm and values must have same length")

    w_srf, r_srf = load_srf(band)
    if not w_srf:
        raise ValueError(f"Empty SRF for band {band}")

    num = 0.0
    den = 0.0
    for lam, val in zip(wavelengths_nm, values):
        w = _interp1d(w_srf, r_srf, lam)
        if w <= 0.0:
            continue
        num += val * w
        den += w

    if den == 0.0:
        raise ValueError(f"SRF convolution denominator zero for band {band}")

    return num / den

