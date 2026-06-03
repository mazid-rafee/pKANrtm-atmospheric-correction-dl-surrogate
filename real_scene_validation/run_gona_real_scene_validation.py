#!/usr/bin/env python3
"""
Real-scene RadCalNet validation pipeline for pKANrtm atmospheric correction.

Performs band-limited validation over a Sentinel-2 / RadCalNet matchup
by running 6S, pKANrtm inference, and (optionally) direct libRadtran
comparisons.  Outputs CSVs, plots, and a comprehensive run_metadata.json.
"""

import argparse
import json
import math
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

USABLE_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
EXCLUDED_BANDS = ["B01", "B09", "B10"]

S2_BANDS_NM: Dict[str, float] = {
    "B01": 445.0, "B02": 520.0, "B03": 560.0, "B04": 654.0,
    "B05": 701.0, "B06": 743.0, "B07": 779.0, "B08": 789.0,
    "B8A": 871.0, "B09": 942.0, "B10": 1372.0, "B11": 1639.0,
    "B12": 2256.0,
}

S2_BAND_RESOLUTION: Dict[str, int] = {
    "B01": 60, "B02": 10, "B03": 10, "B04": 10,
    "B05": 20, "B06": 20, "B07": 20, "B08": 10,
    "B8A": 20, "B09": 60, "B10": 60, "B11": 20, "B12": 20,
}

TARGET_COLUMNS = ["rho_path", "T_total", "spher_alb"]
HIGH_FIDELITY_TARGET_COLUMNS = [f"{c}_lrt" for c in TARGET_COLUMNS]
LOW_FIDELITY_TARGET_COLUMNS  = [f"{c}_6s"  for c in TARGET_COLUMNS]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_LOG_LINES: List[str] = []


def log(msg: str, level: str = "INFO") -> None:
    line = f"[{level}] {msg}"
    _LOG_LINES.append(line)
    print(line, flush=True)

# ---------------------------------------------------------------------------
# Geometry / ROI helpers
# ---------------------------------------------------------------------------

def _build_roi_circle_geojson(lat: float, lon: float, radius_m: float,
                               n_points: int = 128) -> dict:
    from pyproj import Transformer
    from shapely.geometry import Point, mapping
    from shapely.ops import transform as shapely_transform

    utm_zone = int((lon + 180) / 6) + 1
    hemisphere = "north" if lat >= 0 else "south"
    epsg_utm = 32600 + utm_zone if hemisphere == "north" else 32700 + utm_zone
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_utm}", always_xy=True)
    to_wgs = Transformer.from_crs(f"EPSG:{epsg_utm}", "EPSG:4326", always_xy=True)
    pt_utm = shapely_transform(to_utm.transform, Point(lon, lat))
    circle_utm = pt_utm.buffer(radius_m, resolution=n_points)
    circle_wgs = shapely_transform(to_wgs.transform, circle_utm)
    return mapping(circle_wgs)


def _project_roi(geojson: dict, target_crs) -> dict:
    from pyproj import Transformer
    from shapely.geometry import mapping, shape
    from shapely.ops import transform as shapely_transform

    geom = shape(geojson)
    proj = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    return mapping(shapely_transform(proj.transform, geom))

# ---------------------------------------------------------------------------
# A. Sentinel-2 ROI reflectance
# ---------------------------------------------------------------------------

def _parse_quantification(safe_path: Path, product_level: str) -> Tuple[float, float]:
    """Return (quantification_value, radiometric_offset) from Sentinel metadata."""
    if product_level == "L1C":
        xml_name = "MTD_MSIL1C.xml"
    else:
        xml_name = "MTD_MSIL2A.xml"

    xml_path = safe_path / xml_name
    if not xml_path.exists():
        return 10000.0, 0.0

    import xml.etree.ElementTree as ET
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    ns = root.tag.split("}")[0] + "}" if "}" in root.tag else ""

    quant = 10000.0
    offset = 0.0

    for el in root.iter():
        tag = el.tag.split("}")[-1] if "}" in el.tag else el.tag
        if tag in ("QUANTIFICATION_VALUE", "BOA_QUANTIFICATION_VALUE"):
            try:
                quant = float(el.text)
            except (ValueError, TypeError):
                pass
        elif tag in ("BOA_ADD_OFFSET", "RADIO_ADD_OFFSET") and offset == 0.0:
            try:
                offset = float(el.text)
            except (ValueError, TypeError):
                pass

    return quant, offset


def _find_band_file(safe_path: Path, band: str, product_level: str) -> Optional[Path]:
    """Locate the JP2 file for a given band inside a SAFE folder."""
    import glob as globmod
    suffix = band if band != "B8A" else "B8A"
    if product_level == "L1C":
        pattern = str(safe_path / "GRANULE" / "*" / "IMG_DATA" / f"*_{suffix}.jp2")
    else:
        best_res = S2_BAND_RESOLUTION.get(band, 20)
        pattern = str(safe_path / "GRANULE" / "*" / "IMG_DATA" / f"R{best_res}m" / f"*_{suffix}_{best_res}m.jp2")

    matches = sorted(globmod.glob(pattern))
    if matches:
        return Path(matches[0])

    for res in [10, 20, 60]:
        pattern2 = str(safe_path / "GRANULE" / "*" / "IMG_DATA" / f"R{res}m" / f"*_{suffix}_{res}m.jp2")
        matches2 = sorted(globmod.glob(pattern2))
        if matches2:
            return Path(matches2[0])

    pattern3 = str(safe_path / "GRANULE" / "*" / "IMG_DATA" / f"*_{suffix}*.jp2")
    matches3 = sorted(globmod.glob(pattern3))
    if matches3:
        return Path(matches3[0])
    return None


def extract_sentinel_roi(l1c_safe: Path, l2a_safe: Path, roi_geojson: dict,
                          bands: List[str], metadata: dict) -> pd.DataFrame:
    import rasterio
    from rasterio.mask import mask as rio_mask

    quant_l1c, off_l1c = _parse_quantification(l1c_safe, "L1C")
    quant_l2a, off_l2a = _parse_quantification(l2a_safe, "L2A")

    log(f"L1C: quantification={quant_l1c}, offset={off_l1c}")
    log(f"L2A: quantification={quant_l2a}, offset={off_l2a}")

    metadata["l1c_quantification_value"] = quant_l1c
    metadata["l1c_radiometric_offset"] = off_l1c
    metadata["l2a_quantification_value"] = quant_l2a
    metadata["l2a_radiometric_offset"] = off_l2a

    rows = []
    for band in bands:
        l1c_file = _find_band_file(l1c_safe, band, "L1C")
        l2a_file = _find_band_file(l2a_safe, band, "L2A")

        row = {"band": band, "resolution_m": S2_BAND_RESOLUTION.get(band, 0)}

        for label, fpath, quant, off in [("l1c", l1c_file, quant_l1c, off_l1c),
                                          ("l2a", l2a_file, quant_l2a, off_l2a)]:
            prefix = f"{label}_"
            if fpath is None or not fpath.exists():
                log(f"{band} {label.upper()} file not found", "WARN")
                row[prefix + "count_inside"] = 0
                row[prefix + "count_valid"] = 0
                row[prefix + "mean_toa" if label == "l1c" else prefix + "mean_sr"] = np.nan
                row[prefix + "std_toa" if label == "l1c" else prefix + "std_sr"] = np.nan
                continue

            ds = rasterio.open(str(fpath))
            roi_proj = _project_roi(roi_geojson, ds.crs)
            try:
                out_image, _ = rio_mask(ds, [roi_proj], crop=True, filled=False)
            except Exception as e:
                log(f"{band} {label.upper()} masking failed: {e}", "WARN")
                ds.close()
                row[prefix + "count_inside"] = 0
                row[prefix + "count_valid"] = 0
                row[prefix + "mean_toa" if label == "l1c" else prefix + "mean_sr"] = np.nan
                row[prefix + "std_toa" if label == "l1c" else prefix + "std_sr"] = np.nan
                continue

            data = out_image[0]
            inside = ~np.ma.getmaskarray(data)
            raw = np.ma.getdata(data)
            n_inside = int(inside.sum())
            valid = inside & (raw > 0)
            v = raw[valid].astype(np.float64)
            n_valid = int(valid.sum())
            ds.close()

            if n_valid > 0:
                refl = (v + off) / quant
                mean_r = float(refl.mean())
                std_r = float(refl.std())
            else:
                mean_r = np.nan
                std_r = np.nan

            row[prefix + "count_inside"] = n_inside
            row[prefix + "count_valid"] = n_valid
            if label == "l1c":
                row[prefix + "mean_toa"] = mean_r
                row[prefix + "std_toa"] = std_r
            else:
                row[prefix + "mean_sr"] = mean_r
                row[prefix + "std_sr"] = std_r

        rows.append(row)
        log(f"  {band}: L1C valid={row.get('l1c_count_valid', 0)}, "
            f"L2A valid={row.get('l2a_count_valid', 0)}")

    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# B. RadCalNet
# ---------------------------------------------------------------------------

def read_radcalnet(nc_path: Path, sentinel_time_utc: str,
                   metadata: dict) -> Tuple[pd.DataFrame, dict]:
    import netCDF4 as nc

    ds = nc.Dataset(str(nc_path))
    time_grp = ds["Time"]
    years = time_grp["Year"][:].data.astype(int)
    doys = time_grp["DOY"][:].data.astype(int)
    hours = time_grp["Hour_UTC"][:].data.astype(int)
    mins = time_grp["Minutes_UTC"][:].data.astype(int)
    sza_arr = time_grp["Solar_Zenith_Angle"][:].data.astype(float)
    saa_arr = time_grp["Solar_Azimuth_Angle"][:].data.astype(float)

    target_dt = datetime.fromisoformat(sentinel_time_utc.replace("Z", "+00:00"))
    if target_dt.tzinfo is None:
        target_dt = target_dt.replace(tzinfo=timezone.utc)

    best_idx = 0
    best_diff = float("inf")
    slot_dts = []
    for i in range(len(years)):
        dt = datetime(int(years[i]), 1, 1, int(hours[i]), int(mins[i]),
                      tzinfo=timezone.utc) + __import__("datetime").timedelta(days=int(doys[i]) - 1)
        slot_dts.append(dt)
        diff = abs((dt - target_dt).total_seconds())
        if diff < best_diff:
            best_diff = diff
            best_idx = i

    selected_dt = slot_dts[best_idx]
    time_diff_min = best_diff / 60.0
    log(f"RadCalNet nearest slot: {selected_dt.isoformat()} "
        f"(Δ = {time_diff_min:.1f} min)")
    metadata["radcalnet_selected_timestamp"] = selected_dt.isoformat()
    metadata["radcalnet_time_diff_min"] = round(time_diff_min, 2)

    atm = ds["Atmosphere_data"]
    atm_vars: Dict[str, Any] = {}
    for name, subgrp_key, var_key in [
        ("aod550", "AOT", "AOT_data"),
        ("angstrom", "Angstrom", "Angstrom_data"),
        ("cwv", "WaterVapor", "WaterVapor_data"),
        ("ozone_du", "Ozone", "Ozone_data"),
        ("pressure", "Pressure", "Pressure_data"),
        ("temperature", "Temperature", "Temperature_data"),
    ]:
        try:
            arr = atm[subgrp_key][var_key][:].data
            val = float(arr.flat[best_idx]) if arr.size > best_idx else float("nan")
        except Exception:
            val = float("nan")
        atm_vars[name] = val

    atm_vars["ozone"] = atm_vars["ozone_du"] / 1000.0

    try:
        atype_arr = atm["AerosolType"]["AerosolType"][:].data
        atm_vars["aerosol_type_code"] = int(atype_arr.flat[best_idx]) if atype_arr.size > best_idx else None
    except Exception:
        atm_vars["aerosol_type_code"] = None

    atm_vars["sza_deg"] = float(sza_arr[best_idx])
    atm_vars["saa_deg"] = float(saa_arr[best_idx])

    refl = ds["Reflectance"]
    wvl = refl["Wavelength"][:].data.astype(float)
    boa_data = refl["Reflectance_BOA"]["Reflectance_BOA_data"][:].data
    toa_data = refl["Reflectance_TOA"]["Reflectance_TOA_data"][:].data

    try:
        boa_unc = refl["Reflectance_BOA"]["Reflectance_BOA_unc"][:].data
    except Exception:
        boa_unc = None

    if boa_data.ndim == 2:
        if boa_data.shape[0] == len(wvl):
            boa_spec = boa_data[:, best_idx]
            toa_spec = toa_data[:, best_idx]
            boa_unc_spec = boa_unc[:, best_idx] if boa_unc is not None else None
        else:
            boa_spec = boa_data[best_idx, :]
            toa_spec = toa_data[best_idx, :]
            boa_unc_spec = boa_unc[best_idx, :] if boa_unc is not None else None
    else:
        boa_spec = boa_data
        toa_spec = toa_data
        boa_unc_spec = boa_unc

    ds.close()

    boa_spec = np.where((boa_spec > 1.5) | (boa_spec < 0), np.nan, boa_spec)
    toa_spec = np.where((toa_spec > 1.5) | (toa_spec < 0), np.nan, toa_spec)

    spec_df = pd.DataFrame({
        "wavelength_nm": wvl,
        "boa_reflectance": boa_spec,
        "toa_reflectance": toa_spec,
    })
    if boa_unc_spec is not None:
        boa_unc_spec = np.where(np.asarray(boa_unc_spec) > 1.5, np.nan, boa_unc_spec)
        spec_df["boa_uncertainty"] = boa_unc_spec

    return spec_df, atm_vars

# ---------------------------------------------------------------------------
# C. SRF integration
# ---------------------------------------------------------------------------

def integrate_radcalnet_to_srf(spec_df: pd.DataFrame, srf_dir: Path,
                                bands: List[str]) -> pd.DataFrame:
    import csv as csv_mod

    rows = []
    for band in bands:
        stem = band if band == "B8A" else f"B{int(band[1:]):02d}"
        srf_path = srf_dir / f"{stem}.csv"
        if not srf_path.exists():
            srf_path = srf_dir / f"{band}.csv"
        if not srf_path.exists():
            log(f"SRF file not found for {band}: {srf_path}", "WARN")
            rows.append({"band": band, "center_nm": S2_BANDS_NM.get(band, np.nan),
                          "rho_radcalnet_srf": np.nan, "srf_file": "MISSING"})
            continue

        srf_wvl, srf_rsp = [], []
        with srf_path.open("r") as f:
            reader = csv_mod.reader(f)
            for line in reader:
                if not line or line[0].startswith("#") or line[0].lower().startswith("wave"):
                    continue
                try:
                    srf_wvl.append(float(line[0]))
                    srf_rsp.append(float(line[1]))
                except Exception:
                    continue

        rc_wvl = spec_df["wavelength_nm"].values
        rc_boa = spec_df["boa_reflectance"].values

        valid_mask = np.isfinite(rc_boa)
        rc_wvl_v = rc_wvl[valid_mask]
        rc_boa_v = rc_boa[valid_mask]

        num, den = 0.0, 0.0
        for lam, rsp in zip(srf_wvl, srf_rsp):
            if rsp <= 0:
                continue
            if len(rc_wvl_v) == 0:
                continue
            rc_val = np.interp(lam, rc_wvl_v, rc_boa_v)
            if np.isfinite(rc_val):
                num += rc_val * rsp
                den += rsp

        rho_int = num / den if den > 0 else np.nan
        srf_support = [w for w, r in zip(srf_wvl, srf_rsp) if r > 0]
        rows.append({
            "band": band,
            "center_nm": S2_BANDS_NM.get(band, np.nan),
            "rho_radcalnet_srf": rho_int,
            "srf_file": srf_path.name,
            "srf_wavelength_min_nm": min(srf_support) if srf_support else np.nan,
            "srf_wavelength_max_nm": max(srf_support) if srf_support else np.nan,
        })

    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# D. Run 6S
# ---------------------------------------------------------------------------

def _map_aerosol_for_site(site_name: str, aerosol_code: Optional[int]) -> str:
    site_lower = site_name.lower()
    if "gona" in site_lower or "gobabeb" in site_lower:
        return "desert"
    if aerosol_code is not None:
        mapping = {1: "continental", 2: "maritime", 3: "urban", 4: "desert"}
        if aerosol_code in mapping:
            return mapping[aerosol_code]
    return "continental"


def _map_atm_profile_for_site(lat: float) -> str:
    if abs(lat) < 23.5:
        return "tropical"
    return "midlatitude_summer"


def _band_to_project_name(band: str) -> str:
    """Convert B02 → B2, B8A → B8A for project 6S code compatibility."""
    if band == "B8A":
        return "B8A"
    num = band.lstrip("B0").lstrip("B")
    return f"B{num}"


def _nan_safe_convolve(wavelengths_nm, values, srf_wvl, srf_rsp):
    """SRF-weighted average that skips NaN values."""
    num, den = 0.0, 0.0
    skipped = 0
    for lam, val in zip(wavelengths_nm, values):
        if not math.isfinite(val):
            skipped += 1
            continue
        w = float(np.interp(lam, srf_wvl, srf_rsp))
        if w <= 0.0:
            continue
        num += val * w
        den += w
    return (num / den if den > 0 else float("nan")), skipped


def run_6s_for_bands(
    bands: List[str],
    atm_vars: dict,
    srf_dir: Path,
    vza: float,
    vaa: float,
    aerosol_model: str,
    atm_profile: str,
    metadata: dict,
) -> Optional[pd.DataFrame]:
    try:
        from Py6S import SixS as _SixS_check
    except ImportError:
        log("Py6S is not available. 6S/pKANrtm validation cannot proceed.", "FAIL")
        metadata["sixs_status"] = "not_run_missing_6s"
        metadata["pkanrtm_status"] = "not_run_6s_missing"
        return None

    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from data_generator.generate_6s_dataset import (
            _run_6s_reflectance_and_eglo_at_wavelength as sixs_at_wavelength,
        )
        from data_generator.s2_srf import load_srf
    except ImportError as e:
        log(f"Cannot import project 6S code: {e}", "FAIL")
        metadata["sixs_status"] = "not_run_import_error"
        metadata["pkanrtm_status"] = "not_run_6s_missing"
        return None

    sza = atm_vars["sza_deg"]
    saa = atm_vars["saa_deg"]
    raa = abs(saa - vaa) % 360
    if raa > 180:
        raa = 360 - raa

    aod550 = atm_vars["aod550"]
    cwv = atm_vars["cwv"]
    ozone = atm_vars["ozone"]
    pressure = atm_vars.get("pressure", 1013.25)
    elev_km = max(0.0, (1013.25 - pressure) * 0.008) if not math.isnan(pressure) else 0.0

    log(f"6S parameters: SZA={sza:.2f}, VZA={vza:.2f}, RAA={raa:.2f}, "
        f"AOD550={aod550:.4f}, CWV={cwv:.4f} g/cm2, "
        f"O3={atm_vars['ozone_du']:.1f} DU ({ozone:.4f} cm-atm), "
        f"elev_km={elev_km:.3f}, aerosol={aerosol_model}, atm={atm_profile}")

    state = {
        "sza_deg": float(sza),
        "vza_deg": float(vza),
        "raa_deg": float(raa),
        "aod550": float(aod550),
        "cwv_cm": float(cwv),
        "o3_cm": float(ozone),
        "elev_km": float(elev_km),
        "aerosol_type": aerosol_model,
        "atm_profile": atm_profile,
    }

    rows_out = []
    for band in bands:
        proj_band = _band_to_project_name(band)
        log(f"  6S running for {band} ({proj_band})...")

        try:
            srf_wvl, srf_rsp = load_srf(proj_band)
            samples_nm = [w for w, r in zip(srf_wvl, srf_rsp) if r > 0.0]

            band_r = {}
            for albedo in [0.0, 0.2, 0.6]:
                rho_samples = []
                for wvl_nm in samples_nm:
                    try:
                        rho, _ = sixs_at_wavelength(state, wvl_nm, albedo, False)
                    except Exception:
                        rho = float("nan")
                    rho_samples.append(rho)
                r_val, n_skip = _nan_safe_convolve(
                    samples_nm, rho_samples, srf_wvl, srf_rsp)
                band_r[albedo] = r_val
                if n_skip:
                    log(f"    {band} albedo={albedo}: skipped {n_skip}/{len(samples_nm)} NaN wavelengths")

            r0, r1, r2 = band_r[0.0], band_r[0.2], band_r[0.6]
            y1 = r1 - r0
            y2 = r2 - r0
            spher = (y1 / 0.2 - y2 / 0.6) / (y1 - y2 + 1e-30)
            t_total = y1 * (1.0 - spher * 0.2) / (0.2 + 1e-30)
            rho_path = r0

            status = "ok"
            if not all(math.isfinite(v) for v in [rho_path, t_total, spher]):
                status = "non_finite"
        except Exception as e:
            log(f"    6S failed for {band}: {e}", "WARN")
            rho_path, t_total, spher = np.nan, np.nan, np.nan
            status = f"error: {e}"

        rows_out.append({
            "band": band,
            "rho_path_6s": float(rho_path),
            "T_total_6s": float(t_total),
            "s_6s": float(spher),
            "sixs_status": status,
            "aod550": float(aod550),
            "cwv": float(cwv),
            "ozone": float(ozone),
            "pressure": float(pressure),
            "sza": float(sza),
            "saa": float(saa),
            "vza": float(vza),
            "vaa": float(vaa),
            "raa": float(raa),
            "aerosol_model": aerosol_model,
            "atmospheric_profile": atm_profile,
        })
        log(f"    {band}: rho_path={rho_path:.6f}, T_total={t_total:.6f}, "
            f"s={spher:.6f} [{status}]")

    metadata["sixs_status"] = "completed"
    metadata["sixs_note"] = (
        "SRF-weighted band-averaged 6S using project per-wavelength function "
        "(consistent with training). NaN wavelengths skipped in convolution."
    )
    return pd.DataFrame(rows_out)


def run_6s_center_wavelength(
    bands: List[str],
    atm_vars: dict,
    vza: float,
    vaa: float,
    aerosol_model: str,
    atm_profile: str,
    metadata: dict,
) -> Optional[pd.DataFrame]:
    """Run 6S at the band center wavelength only (no SRF convolution)."""
    try:
        from Py6S import SixS as _SixS_check
    except ImportError:
        log("Py6S not available for center-wavelength 6S.", "FAIL")
        metadata["sixs_center_status"] = "not_run_missing_6s"
        return None

    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from data_generator.generate_6s_dataset import (
            _run_6s_reflectance_and_eglo_at_wavelength as sixs_at_wavelength,
        )
    except ImportError as e:
        log(f"Cannot import project 6S code: {e}", "FAIL")
        metadata["sixs_center_status"] = "not_run_import_error"
        return None

    sza = atm_vars["sza_deg"]
    saa = atm_vars["saa_deg"]
    raa = abs(saa - vaa) % 360
    if raa > 180:
        raa = 360 - raa

    aod550 = atm_vars["aod550"]
    cwv = atm_vars["cwv"]
    ozone = atm_vars["ozone"]
    pressure = atm_vars.get("pressure", 1013.25)
    elev_km = max(0.0, (1013.25 - pressure) * 0.008) if not math.isnan(pressure) else 0.0

    state = {
        "sza_deg": float(sza), "vza_deg": float(vza), "raa_deg": float(raa),
        "aod550": float(aod550), "cwv_cm": float(cwv), "o3_cm": float(ozone),
        "elev_km": float(elev_km),
        "aerosol_type": aerosol_model, "atm_profile": atm_profile,
    }

    log(f"6S center-wavelength mode: same state as SRF-weighted, single wvl per band")

    rows_out = []
    for band in bands:
        center_nm = S2_BANDS_NM[band]
        log(f"  6S-center running for {band} @ {center_nm} nm ...")
        try:
            r0, _ = sixs_at_wavelength(state, center_nm, 0.0, False)
            r1, _ = sixs_at_wavelength(state, center_nm, 0.2, False)
            r2, _ = sixs_at_wavelength(state, center_nm, 0.6, False)

            y1 = r1 - r0
            y2 = r2 - r0
            spher = (y1 / 0.2 - y2 / 0.6) / (y1 - y2 + 1e-30)
            t_total = y1 * (1.0 - spher * 0.2) / (0.2 + 1e-30)
            rho_path = r0

            status = "ok"
            if not all(math.isfinite(v) for v in [rho_path, t_total, spher]):
                status = "non_finite"
        except Exception as e:
            log(f"    6S-center failed for {band}: {e}", "WARN")
            rho_path, t_total, spher = np.nan, np.nan, np.nan
            status = f"error: {e}"

        rows_out.append({
            "band": band,
            "center_nm": center_nm,
            "rho_path_6s_center": float(rho_path),
            "T_total_6s_center": float(t_total),
            "s_6s_center": float(spher),
            "sixs_center_status": status,
        })
        log(f"    {band}: rho_path={rho_path:.6f}, T_total={t_total:.6f}, "
            f"s={spher:.6f} [{status}]")

    metadata["sixs_center_status"] = "completed"
    metadata["sixs_center_note"] = (
        "6S at band center wavelength only (no SRF convolution). "
        "3-albedo anchor method at a single wavelength."
    )
    return pd.DataFrame(rows_out)


def run_libradtran_center_wavelength(
    bands: List[str],
    atm_vars: dict,
    vza: float,
    vaa: float,
    aerosol_model: str,
    atm_profile: str,
    metadata: dict,
    uvspec_bin: str = "uvspec",
) -> Optional[pd.DataFrame]:
    """Run libRadtran at the band center wavelength only (no SRF convolution)."""
    import shutil
    if not shutil.which(uvspec_bin):
        log(f"uvspec binary not found at '{uvspec_bin}'. Cannot run libRadtran.", "FAIL")
        metadata["libradtran_center_status"] = "not_run_uvspec_not_found"
        return None

    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from data_generator.generate_libradtran_dataset import (
            RtState, get_rho_toa, solve_params_reflectance,
        )
    except ImportError as e:
        log(f"Cannot import project libRadtran code: {e}", "FAIL")
        metadata["libradtran_center_status"] = "not_run_import_error"
        return None

    sza = atm_vars["sza_deg"]
    saa = atm_vars["saa_deg"]
    raa = abs(saa - vaa) % 360
    if raa > 180:
        raa = 360 - raa

    aod550 = atm_vars["aod550"]
    cwv = atm_vars["cwv"]
    ozone = atm_vars["ozone"]
    pressure = atm_vars.get("pressure", 1013.25)
    elev_km = max(0.0, (1013.25 - pressure) * 0.008) if not math.isnan(pressure) else 0.0

    st = RtState(
        sza_deg=float(sza), vza_deg=float(vza), raa_deg=float(raa),
        aod550=float(aod550), cwv_cm=float(cwv), o3_cm=float(ozone),
        elev_km=float(elev_km),
        aerosol_type=aerosol_model, atm_profile=atm_profile,
    )

    log(f"libRadtran center-wavelength mode: uvspec={uvspec_bin}")
    log(f"  State: SZA={sza:.2f}, VZA={vza:.2f}, RAA={raa:.2f}, "
        f"AOD={aod550:.4f}, CWV={cwv:.4f}, O3={ozone:.4f} cm-atm, "
        f"elev={elev_km:.3f} km, aerosol={aerosol_model}, atm={atm_profile}")

    rows_out = []
    for band in bands:
        center_nm = S2_BANDS_NM[band]
        log(f"  libRadtran-center running for {band} @ {center_nm} nm ...")
        try:
            r0 = get_rho_toa(st, center_nm, 0.0)
            r1 = get_rho_toa(st, center_nm, 0.2)
            r2 = get_rho_toa(st, center_nm, 0.6)
            rho_path, t_total, spher = solve_params_reflectance(r0, r1, r2, 0.2, 0.6)

            status = "ok"
            if not all(math.isfinite(v) for v in [rho_path, t_total, spher]):
                status = "non_finite"
        except Exception as e:
            log(f"    libRadtran-center failed for {band}: {e}", "WARN")
            rho_path, t_total, spher = np.nan, np.nan, np.nan
            status = f"error: {e}"

        rows_out.append({
            "band": band,
            "center_nm": center_nm,
            "rho_path_lrt_center": float(rho_path),
            "T_total_lrt_center": float(t_total),
            "s_lrt_center": float(spher),
            "lrt_center_status": status,
        })
        log(f"    {band}: rho_path={rho_path:.6f}, T_total={t_total:.6f}, "
            f"s={spher:.6f} [{status}]")

    metadata["libradtran_center_status"] = "completed"
    metadata["libradtran_center_note"] = (
        "libRadtran at band center wavelength only (no SRF convolution). "
        "3-albedo anchor method (get_rho_toa) at a single wavelength."
    )
    return pd.DataFrame(rows_out)


# ---------------------------------------------------------------------------
# E. Run pKANrtm
# ---------------------------------------------------------------------------

def _search_artifact(search_dirs: List[Path], candidates: List[str]) -> Optional[Path]:
    for d in search_dirs:
        if not d.exists():
            continue
        for name in candidates:
            p = d / name
            if p.exists():
                return p
    return None


def run_pkanrtm(
    sixs_df: pd.DataFrame,
    atm_vars: dict,
    checkpoint_path: Path,
    preprocessor_path: Optional[Path],
    scaler_path: Optional[Path],
    bands: List[str],
    aerosol_model: str,
    atm_profile: str,
    metadata: dict,
) -> Optional[pd.DataFrame]:
    try:
        import torch
        import joblib
    except ImportError as e:
        log(f"Missing dependency for pKANrtm: {e}", "WARN")
        metadata["pkanrtm_status"] = "not_run_missing_artifacts"
        return None

    if not checkpoint_path.exists():
        log(f"Checkpoint not found: {checkpoint_path}", "WARN")
        metadata["pkanrtm_status"] = "not_run_missing_artifacts"
        return None

    if preprocessor_path is None or not preprocessor_path.exists():
        log(f"Preprocessor not found: {preprocessor_path}", "WARN")
        metadata["pkanrtm_status"] = "not_run_missing_artifacts"
        return None

    if scaler_path is None or not scaler_path.exists():
        log(f"Target scaler not found: {scaler_path}", "WARN")
        metadata["pkanrtm_status"] = "not_run_missing_artifacts"
        return None

    try:
        preprocessor = joblib.load(preprocessor_path)
        target_scaler = joblib.load(scaler_path)
        feature_names = list(preprocessor.feature_names_in_)
        log(f"pKANrtm preprocessor features: {feature_names}")
        log(f"pKANrtm target scaler: mean={target_scaler.mean_}, scale={target_scaler.scale_}")
    except Exception as e:
        log(f"Failed to load scalers: {e}", "WARN")
        metadata["pkanrtm_status"] = f"not_run_inference_error: {e}"
        return None

    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from surrogate_pipeline.models import build_model

        ckpt = torch.load(str(checkpoint_path), map_location="cpu")

        model_kwargs = {
            "mlp_arch": "baseline", "kan_arch": "small",
            "activation": "relu", "use_layernorm": False,
            "deployable_stage_sizing": False,
        }
        for parent in [checkpoint_path.parent] + list(checkpoint_path.parent.parents)[:6]:
            cfg_path = parent / "run_config.json"
            if cfg_path.exists():
                try:
                    cfg = json.loads(cfg_path.read_text())
                    for k in model_kwargs:
                        if k in cfg:
                            model_kwargs[k] = cfg[k]
                    log(f"Loaded model config from {cfg_path.name}: kan_arch={model_kwargs['kan_arch']}")
                    metadata["pkanrtm_run_config"] = str(cfg_path)
                except Exception:
                    pass
                break

        in_dim = None
        if "model_state_dict" in ckpt:
            first_key = list(ckpt["model_state_dict"].keys())[0]
            first_param = ckpt["model_state_dict"][first_key]
            if first_param.dim() >= 2:
                in_dim = first_param.shape[-1]

        if in_dim is None:
            sample_row = _build_pkanrtm_feature_row(
                bands[0], sixs_df, atm_vars, aerosol_model, atm_profile, feature_names)
            x_sample = preprocessor.transform(pd.DataFrame([sample_row])[
                [c for c in feature_names if c in sample_row]])
            if hasattr(x_sample, "toarray"):
                x_sample = x_sample.toarray()
            in_dim = x_sample.shape[1]

        model, _ = build_model(
            "pkan", in_dim=in_dim,
            out_dim=len(TARGET_COLUMNS),
            kan_arch=model_kwargs["kan_arch"],
            activation=model_kwargs["activation"],
            use_layernorm=model_kwargs["use_layernorm"],
            deployable_stage_sizing=model_kwargs["deployable_stage_sizing"],
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        log(f"pKANrtm model loaded: {sum(p.numel() for p in model.parameters())} parameters")
    except Exception as e:
        log(f"Failed to load pKANrtm model: {e}", "WARN")
        metadata["pkanrtm_status"] = f"not_run_inference_error: {e}"
        return None

    rows_out = []
    try:
        feature_rows = []
        for band in bands:
            fr = _build_pkanrtm_feature_row(
                band, sixs_df, atm_vars, aerosol_model, atm_profile, feature_names)
            feature_rows.append(fr)

        input_df = pd.DataFrame(feature_rows)
        available_cols = [c for c in feature_names if c in input_df.columns]
        missing_cols = [c for c in feature_names if c not in input_df.columns]
        if missing_cols:
            log(f"pKANrtm missing columns (will be NaN): {missing_cols}", "WARN")
            for mc in missing_cols:
                input_df[mc] = np.nan

        x = preprocessor.transform(input_df[feature_names])
        if hasattr(x, "toarray"):
            x = x.toarray()
        x = np.asarray(x, dtype=np.float32)

        with torch.no_grad():
            pred_scaled = model(torch.from_numpy(x)).numpy()

        pred = target_scaler.inverse_transform(pred_scaled)

        metadata["pkanrtm_prediction_convention"] = (
            "oracle_residual: model directly predicts high-fidelity (libRadtran-equivalent) "
            "coefficients [rho_path_lrt, T_total_lrt, spher_alb_lrt] given scene state + 6S coefficients as input"
        )

        for i, band in enumerate(bands):
            rows_out.append({
                "band": band,
                "rho_path_pkanrtm": float(pred[i, 0]),
                "T_total_pkanrtm": float(pred[i, 1]),
                "s_pkanrtm": float(pred[i, 2]),
                "pkanrtm_status": "ok",
            })
            log(f"  pKANrtm {band}: rho_path={pred[i,0]:.6f}, "
                f"T_total={pred[i,1]:.6f}, s={pred[i,2]:.6f}")

        metadata["pkanrtm_status"] = "completed"
    except Exception as e:
        log(f"pKANrtm inference failed: {e}", "WARN")
        metadata["pkanrtm_status"] = f"not_run_inference_error: {e}"
        return None

    return pd.DataFrame(rows_out)


def _build_pkanrtm_feature_row(band: str, sixs_df: pd.DataFrame,
                                 atm_vars: dict, aerosol_model: str,
                                 atm_profile: str,
                                 feature_names: List[str]) -> dict:
    sixs_row = sixs_df[sixs_df["band"] == band].iloc[0]

    base_vals = {
        "wvl_nm": S2_BANDS_NM.get(band, np.nan),
        "sza_deg": atm_vars["sza_deg"],
        "vza_deg": float(sixs_row["vza"]),
        "raa_deg": float(sixs_row["raa"]),
        "aod550": atm_vars["aod550"],
        "cwv_cm": atm_vars["cwv"],
        "o3_cm": atm_vars["ozone"],
        "elev_km": max(0.0, (1013.25 - atm_vars.get("pressure", 1013.25)) * 0.008),
        "rho_path_6s": float(sixs_row["rho_path_6s"]),
        "T_total_6s": float(sixs_row["T_total_6s"]),
        "spher_alb_6s": float(sixs_row["s_6s"]),
        "band": band,
        "aerosol_type": aerosol_model,
        "atm_profile": atm_profile,
    }

    row = {}
    for fname in feature_names:
        clean = fname
        for suffix in ("_lrt", "_6s"):
            if clean.endswith(suffix) and clean not in base_vals:
                clean = clean[: -len(suffix)]
                break
        if fname in base_vals:
            row[fname] = base_vals[fname]
        elif clean in base_vals:
            row[fname] = base_vals[clean]
        else:
            row[fname] = np.nan

    return row

# ---------------------------------------------------------------------------
# F. Retrieve surface reflectance
# ---------------------------------------------------------------------------

def retrieve_surface_reflectance(
    sentinel_df: pd.DataFrame,
    sixs_df: Optional[pd.DataFrame],
    pkanrtm_df: Optional[pd.DataFrame],
    libradtran_df: Optional[pd.DataFrame],
    bands: List[str],
    sixs_center_df: Optional[pd.DataFrame] = None,
    lrt_center_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    rows = []
    for band in bands:
        s_row = sentinel_df[sentinel_df["band"] == band]
        if s_row.empty:
            continue
        rho_toa = float(s_row["l1c_mean_toa"].iloc[0])

        result = {"band": band, "rho_toa_l1c": rho_toa}

        sources = [
            ("6s", sixs_df, "rho_path_6s", "T_total_6s", "s_6s"),
            ("pkanrtm", pkanrtm_df, "rho_path_pkanrtm", "T_total_pkanrtm", "s_pkanrtm"),
            ("libradtran", libradtran_df, "rho_path_libradtran", "T_total_libradtran", "s_libradtran"),
            ("6s_center", sixs_center_df, "rho_path_6s_center", "T_total_6s_center", "s_6s_center"),
            ("lrt_center", lrt_center_df, "rho_path_lrt_center", "T_total_lrt_center", "s_lrt_center"),
        ]

        for label, df, rp_col, tt_col, s_col in sources:
            if df is None:
                result[f"rho_{label}"] = np.nan
                continue

            row_df = df[df["band"] == band]
            if row_df.empty:
                result[f"rho_{label}"] = np.nan
                continue

            rp = float(row_df[rp_col].iloc[0])
            tt = float(row_df[tt_col].iloc[0])
            ss = float(row_df[s_col].iloc[0])

            numerator = rho_toa - rp
            denominator = tt + ss * (rho_toa - rp)

            if not np.isfinite(denominator) or abs(denominator) < 1e-12:
                result[f"rho_{label}"] = np.nan
                log(f"  {band} {label}: unstable denominator ({denominator})", "WARN")
            else:
                result[f"rho_{label}"] = numerator / denominator

        rows.append(result)

    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# G. libRadtran (optional)
# ---------------------------------------------------------------------------

def run_libradtran_for_bands(args, bands, atm_vars, srf_dir, metadata):
    log("libRadtran requested but not yet implemented in this pipeline.", "WARN")
    metadata["libradtran_status"] = "not_implemented"
    return None

# ---------------------------------------------------------------------------
# H. Comparison table
# ---------------------------------------------------------------------------

def build_band_table(
    sentinel_df: pd.DataFrame,
    srf_df: pd.DataFrame,
    refl_df: pd.DataFrame,
    bands: List[str],
    has_libradtran: bool,
    has_6s_center: bool = False,
    has_lrt_center: bool = False,
) -> pd.DataFrame:
    rows = []
    for band in bands:
        s = sentinel_df[sentinel_df["band"] == band]
        srf = srf_df[srf_df["band"] == band]
        r = refl_df[refl_df["band"] == band]

        rho_toa = float(s["l1c_mean_toa"].iloc[0]) if not s.empty else np.nan
        rho_l2a = float(s["l2a_mean_sr"].iloc[0]) if not s.empty else np.nan
        rho_rc = float(srf["rho_radcalnet_srf"].iloc[0]) if not srf.empty else np.nan
        rho_6s = float(r["rho_6s"].iloc[0]) if not r.empty and "rho_6s" in r.columns else np.nan
        rho_pk = float(r["rho_pkanrtm"].iloc[0]) if not r.empty and "rho_pkanrtm" in r.columns else np.nan

        row = {
            "band": band,
            "center_nm": S2_BANDS_NM.get(band, np.nan),
            "rho_toa_l1c": rho_toa,
            "rho_l2a_sen2cor": rho_l2a,
            "rho_radcalnet_srf": rho_rc,
            "rho_6s_only": rho_6s,
            "rho_6s_pkanrtm": rho_pk,
        }

        if has_libradtran and not r.empty and "rho_libradtran" in r.columns:
            row["rho_libradtran"] = float(r["rho_libradtran"].iloc[0])

        if has_6s_center and not r.empty and "rho_6s_center" in r.columns:
            row["rho_6s_center"] = float(r["rho_6s_center"].iloc[0])

        if has_lrt_center and not r.empty and "rho_lrt_center" in r.columns:
            row["rho_lrt_center"] = float(r["rho_lrt_center"].iloc[0])

        def _err(a, b):
            return a - b if np.isfinite(a) and np.isfinite(b) else np.nan

        def _rel(a, b):
            if np.isfinite(a) and np.isfinite(b) and abs(b) > 1e-8:
                return 100.0 * (a - b) / b
            return np.nan

        row["err_l2a_vs_radcalnet"] = _err(rho_l2a, rho_rc)
        row["err_6s_vs_radcalnet"] = _err(rho_6s, rho_rc)
        row["err_pkanrtm_vs_radcalnet"] = _err(rho_pk, rho_rc)
        row["rel_l2a_vs_radcalnet_percent"] = _rel(rho_l2a, rho_rc)
        row["rel_6s_vs_radcalnet_percent"] = _rel(rho_6s, rho_rc)
        row["rel_pkanrtm_vs_radcalnet_percent"] = _rel(rho_pk, rho_rc)

        if has_libradtran and "rho_libradtran" in row:
            rho_lr = row["rho_libradtran"]
            row["err_libradtran_vs_radcalnet"] = _err(rho_lr, rho_rc)
            row["rel_libradtran_vs_radcalnet_percent"] = _rel(rho_lr, rho_rc)

        if has_6s_center and "rho_6s_center" in row:
            rho_6c = row["rho_6s_center"]
            row["err_6s_center_vs_radcalnet"] = _err(rho_6c, rho_rc)
            row["rel_6s_center_vs_radcalnet_percent"] = _rel(rho_6c, rho_rc)

        if has_lrt_center and "rho_lrt_center" in row:
            rho_lc = row["rho_lrt_center"]
            row["err_lrt_center_vs_radcalnet"] = _err(rho_lc, rho_rc)
            row["rel_lrt_center_vs_radcalnet_percent"] = _rel(rho_lc, rho_rc)

        rows.append(row)

    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# I. Metrics
# ---------------------------------------------------------------------------

def compute_metrics(pred: np.ndarray, ref: np.ndarray, label: str) -> dict:
    mask = np.isfinite(pred) & np.isfinite(ref)
    p, r = pred[mask], ref[mask]
    n = len(p)
    if n == 0:
        return {"comparison": label, "n_bands": 0, "rmse": np.nan,
                "mae": np.nan, "mean_rel_diff_pct": np.nan,
                "mean_abs_rel_diff_pct": np.nan}

    diff = p - r
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    rel = 100.0 * diff / np.where(np.abs(r) > 1e-8, r, np.nan)
    mrd = float(np.nanmean(rel))
    mard = float(np.nanmean(np.abs(rel)))

    return {"comparison": label, "n_bands": n, "rmse": rmse,
            "mae": mae, "mean_rel_diff_pct": mrd,
            "mean_abs_rel_diff_pct": mard}


def compute_all_metrics(
    band_table: pd.DataFrame,
    has_libradtran: bool,
    has_6s_center: bool = False,
    has_lrt_center: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    rc = band_table["rho_radcalnet_srf"].values
    l2a = band_table["rho_l2a_sen2cor"].values
    s6 = band_table["rho_6s_only"].values
    pk = band_table["rho_6s_pkanrtm"].values if "rho_6s_pkanrtm" in band_table.columns else np.full(len(rc), np.nan)

    rc_metrics = [
        compute_metrics(l2a, rc, "L2A vs RadCalNet"),
        compute_metrics(s6, rc, "6S-SRF vs RadCalNet"),
        compute_metrics(pk, rc, "6S+pKANrtm vs RadCalNet"),
    ]

    l2a_metrics = [
        compute_metrics(s6, l2a, "6S-SRF vs L2A"),
        compute_metrics(pk, l2a, "6S+pKANrtm vs L2A"),
    ]

    if has_6s_center and "rho_6s_center" in band_table.columns:
        s6c = band_table["rho_6s_center"].values
        rc_metrics.append(compute_metrics(s6c, rc, "6S-center vs RadCalNet"))
        l2a_metrics.append(compute_metrics(s6c, l2a, "6S-center vs L2A"))

    if has_lrt_center and "rho_lrt_center" in band_table.columns:
        lrc = band_table["rho_lrt_center"].values
        rc_metrics.append(compute_metrics(lrc, rc, "libRadtran-center vs RadCalNet"))
        l2a_metrics.append(compute_metrics(lrc, l2a, "libRadtran-center vs L2A"))

    lrt_metrics = None
    if has_libradtran and "rho_libradtran" in band_table.columns:
        lr = band_table["rho_libradtran"].values
        lrt_metrics_list = [
            compute_metrics(lr, rc, "libRadtran vs RadCalNet"),
            compute_metrics(lr, l2a, "libRadtran vs L2A"),
            compute_metrics(pk, lr, "pKANrtm vs libRadtran"),
            compute_metrics(s6, lr, "6S-only vs libRadtran"),
        ]
        lrt_metrics = pd.DataFrame(lrt_metrics_list)

    return pd.DataFrame(rc_metrics), pd.DataFrame(l2a_metrics), lrt_metrics

# ---------------------------------------------------------------------------
# J. Plots
# ---------------------------------------------------------------------------

def create_plots(band_table: pd.DataFrame, output_dir: Path,
                  has_pkanrtm: bool, has_libradtran: bool,
                  has_6s_center: bool = False, has_lrt_center: bool = False) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wvl = band_table["center_nm"].values
    order = np.argsort(wvl)
    wvl_s = wvl[order]

    # --- Spectral curves ---
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(wvl_s, band_table["rho_radcalnet_srf"].values[order], "ko-",
            label="RadCalNet SRF-integrated", markersize=7, linewidth=2)
    ax.plot(wvl_s, band_table["rho_l2a_sen2cor"].values[order], "bs-",
            label="Sentinel-2 L2A/Sen2Cor", markersize=5)
    if "rho_6s_only" in band_table.columns:
        ax.plot(wvl_s, band_table["rho_6s_only"].values[order], "r^-",
                label="6S-SRF retrieval", markersize=5)
    if has_pkanrtm and "rho_6s_pkanrtm" in band_table.columns:
        ax.plot(wvl_s, band_table["rho_6s_pkanrtm"].values[order], "gD-",
                label="6S+pKANrtm retrieval", markersize=5)
    if has_libradtran and "rho_libradtran" in band_table.columns:
        ax.plot(wvl_s, band_table["rho_libradtran"].values[order], "mv-",
                label="libRadtran-SRF retrieval", markersize=5)
    if has_6s_center and "rho_6s_center" in band_table.columns:
        ax.plot(wvl_s, band_table["rho_6s_center"].values[order], "r+--",
                label="6S-center retrieval", markersize=8, linewidth=1, alpha=0.7)
    if has_lrt_center and "rho_lrt_center" in band_table.columns:
        ax.plot(wvl_s, band_table["rho_lrt_center"].values[order], "mx--",
                label="libRadtran-center retrieval", markersize=8, linewidth=1, alpha=0.7)

    for i, b in enumerate(band_table["band"].values[order]):
        ax.annotate(b, (wvl_s[i], band_table["rho_radcalnet_srf"].values[order][i]),
                    textcoords="offset points", xytext=(0, 10), fontsize=7,
                    ha="center", alpha=0.7)

    ax.set_xlabel("Wavelength (nm)", fontsize=12)
    ax.set_ylabel("Surface Reflectance", fontsize=12)
    ax.set_title("Spectral Comparison: GONA 2018-05-25 Real-Scene Validation", fontsize=13)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.savefig(str(output_dir / "spectral_curves.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Relative difference ---
    fig, ax = plt.subplots(figsize=(14, 6))
    x_pos = np.arange(len(wvl_s))
    band_labels = band_table["band"].values[order]

    items = [("rel_l2a_vs_radcalnet_percent", "L2A", "blue")]
    if "rel_6s_vs_radcalnet_percent" in band_table.columns:
        items.append(("rel_6s_vs_radcalnet_percent", "6S-SRF", "red"))
    if has_pkanrtm and "rel_pkanrtm_vs_radcalnet_percent" in band_table.columns:
        items.append(("rel_pkanrtm_vs_radcalnet_percent", "pKANrtm", "green"))
    if has_libradtran and "rel_libradtran_vs_radcalnet_percent" in band_table.columns:
        items.append(("rel_libradtran_vs_radcalnet_percent", "libRadtran-SRF", "purple"))
    if has_6s_center and "rel_6s_center_vs_radcalnet_percent" in band_table.columns:
        items.append(("rel_6s_center_vs_radcalnet_percent", "6S-center", "salmon"))
    if has_lrt_center and "rel_lrt_center_vs_radcalnet_percent" in band_table.columns:
        items.append(("rel_lrt_center_vs_radcalnet_percent", "lRT-center", "orchid"))

    n_bars = len(items)
    width = min(0.18, 0.8 / max(n_bars, 1))
    start = -(n_bars - 1) * width / 2
    for j, (col, lbl, clr) in enumerate(items):
        vals = band_table[col].values[order]
        ax.bar(x_pos + start + j * width, vals, width=width, label=lbl, color=clr, alpha=0.7)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(band_labels, rotation=45)
    ax.set_ylabel("Relative Difference vs RadCalNet (%)", fontsize=12)
    ax.set_title("Relative Difference Against RadCalNet by Band", fontsize=13)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(0, color="black", linewidth=0.8)
    fig.savefig(str(output_dir / "relative_difference_radcalnet.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Scatter ---
    fig, ax = plt.subplots(figsize=(8, 8))
    rc = band_table["rho_radcalnet_srf"].values

    ax.scatter(rc, band_table["rho_l2a_sen2cor"].values, marker="s",
               color="blue", s=50, label="L2A", zorder=3)
    if "rho_6s_only" in band_table.columns:
        ax.scatter(rc, band_table["rho_6s_only"].values, marker="^",
                   color="red", s=50, label="6S-SRF", zorder=3)
    if has_pkanrtm and "rho_6s_pkanrtm" in band_table.columns:
        ax.scatter(rc, band_table["rho_6s_pkanrtm"].values, marker="D",
                   color="green", s=50, label="pKANrtm", zorder=3)
    if has_libradtran and "rho_libradtran" in band_table.columns:
        ax.scatter(rc, band_table["rho_libradtran"].values, marker="v",
                   color="purple", s=50, label="libRadtran-SRF", zorder=3)
    if has_6s_center and "rho_6s_center" in band_table.columns:
        ax.scatter(rc, band_table["rho_6s_center"].values, marker="+",
                   color="salmon", s=70, label="6S-center", zorder=3)
    if has_lrt_center and "rho_lrt_center" in band_table.columns:
        ax.scatter(rc, band_table["rho_lrt_center"].values, marker="x",
                   color="orchid", s=70, label="lRT-center", zorder=3)

    lims = [0, max(0.5, np.nanmax(rc) * 1.15)]
    ax.plot(lims, lims, "k--", alpha=0.5, label="1:1 line")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("RadCalNet SRF-integrated Reflectance", fontsize=12)
    ax.set_ylabel("Retrieved / Reference Reflectance", fontsize=12)
    ax.set_title("Scatter: Methods vs RadCalNet", fontsize=13)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    fig.savefig(str(output_dir / "scatter_radcalnet.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    log("Plots saved: spectral_curves.png, relative_difference_radcalnet.png, scatter_radcalnet.png")

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    data_dir = SCRIPT_DIR / "data" / "GONA 25th May 2018"

    p = argparse.ArgumentParser(
        description="Real-scene RadCalNet validation for pKANrtm atmospheric correction.")

    p.add_argument("--l1c-safe", type=str,
                   default=str(data_dir / "S2A_MSIL1C_20180525T084601_N0500_R107_T33KWP_20230903T183659.SAFE"))
    p.add_argument("--l2a-safe", type=str,
                   default=str(data_dir / "S2A_MSIL2A_20180525T084601_N0500_R107_T33KWP_20230904T024512.SAFE"))
    p.add_argument("--radcalnet-nc", type=str,
                   default=str(data_dir / "GONA01_2018_145_v04.09.nc"))
    p.add_argument("--srf-dir", type=str,
                   default=str(PROJECT_ROOT / "data" / "sentinel_srf" / "s2_srf" / "S2A"))
    p.add_argument("--checkpoint", type=str,
                   default=str(PROJECT_ROOT / "runs" / "fig05_surrogate_pkan" / "oracle_residual" /
                               "splits" / "ood_aod_cwv" / "models" / "pkan" / "best.pt"))
    p.add_argument("--feature-scaler", type=str, default=None)
    p.add_argument("--target-scaler", type=str, default=None)
    p.add_argument("--feature-order", type=str, default=None)
    p.add_argument("--output-dir", type=str,
                   default=str(SCRIPT_DIR / "results" / "gona_20180525_real_validation"))

    p.add_argument("--site-name", default="GONA")
    p.add_argument("--site-long-name", default="Gobabeb, Namibia")
    p.add_argument("--roi-center-lat", type=float, default=-23.599990)
    p.add_argument("--roi-center-lon", type=float, default=15.119215)
    p.add_argument("--roi-radius-m", type=float, default=30.0)
    p.add_argument("--sentinel-time-utc", default="2018-05-25T08:46:01")
    p.add_argument("--vza", type=float, default=3.0,
                   help="View zenith angle (degrees). Near-nadir for Sentinel-2.")
    p.add_argument("--vaa", type=float, default=105.0,
                   help="View azimuth angle (degrees).")

    p.add_argument("--run-libradtran", action="store_true", default=False)
    p.add_argument("--libradtran-bin", type=str, default=None)
    p.add_argument("--libradtran-data-dir", type=str, default=None)
    p.add_argument("--libradtran-template-dir", type=str, default=None)
    p.add_argument("--libradtran-work-dir", type=str, default=None)

    p.add_argument("--run-6s-center", action="store_true", default=False,
                   help="Run 6S at band center wavelengths only (fast, no SRF convolution).")
    p.add_argument("--run-libradtran-center", action="store_true", default=False,
                   help="Run libRadtran at band center wavelengths only (fast, no SRF convolution).")
    p.add_argument("--uvspec-bin", type=str,
                   default="/aul/homes/mmazi007/rtm/libradtran_install/bin/uvspec",
                   help="Path to uvspec binary for libRadtran center-wl mode.")

    return p.parse_args()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata: Dict[str, Any] = {
        "pipeline": "real_scene_radcalnet_validation",
        "pipeline_version": "1.0.0",
        "site_name": args.site_name,
        "site_long_name": args.site_long_name,
        "roi_center_lat": args.roi_center_lat,
        "roi_center_lon": args.roi_center_lon,
        "roi_radius_m": args.roi_radius_m,
        "sentinel_time_utc": args.sentinel_time_utc,
        "l1c_safe": args.l1c_safe,
        "l2a_safe": args.l2a_safe,
        "radcalnet_nc": args.radcalnet_nc,
        "srf_dir": args.srf_dir,
        "checkpoint": args.checkpoint,
        "bands_used": USABLE_BANDS,
        "bands_excluded": EXCLUDED_BANDS,
        "exclusion_reason": (
            "Band-limited validation: B01 and B09 excluded due to no valid pixels "
            "inside the 30 m RadCalNet ROI; B10 excluded as cirrus."
        ),
        "geometry_assumptions": {
            "vza_deg": args.vza,
            "vaa_deg": args.vaa,
            "note": "Sentinel-2 near-nadir assumed if view angles not extracted from metadata",
        },
        "sixs_status": "pending",
        "pkanrtm_status": "pending",
        "libradtran_status": "not_requested" if not args.run_libradtran else "pending",
        "coefficient_definitions": {
            "rho_path": "Atmospheric path reflectance (r0 from albedo=0 anchor)",
            "T_total": "Total transmittance product = y1*(1 - s*rho1)/rho1",
            "spher_alb": "Spherical albedo s = (y1/rho1 - y2/rho2)/(y1 - y2)",
            "anchor_albedos": "rho0=0.0, rho1=0.2, rho2=0.6",
            "surface_retrieval": "rho_surface = (rho_TOA - rho_path) / (T_total + s*(rho_TOA - rho_path))",
        },
    }

    log("=" * 70)
    log(f"Real-Scene RadCalNet Validation: {args.site_name} ({args.site_long_name})")
    log(f"Date: {args.sentinel_time_utc}")
    log("=" * 70)

    # --- A: Sentinel-2 ROI reflectance ---
    log("\n--- A. Extract Sentinel-2 ROI reflectance ---")
    roi_geojson = _build_roi_circle_geojson(args.roi_center_lat, args.roi_center_lon,
                                             args.roi_radius_m)

    l1c_safe = Path(args.l1c_safe)
    l2a_safe = Path(args.l2a_safe)
    if not l1c_safe.exists():
        log(f"L1C SAFE not found: {l1c_safe}", "FAIL")
        _save_metadata(metadata, output_dir)
        return
    if not l2a_safe.exists():
        log(f"L2A SAFE not found: {l2a_safe}", "FAIL")
        _save_metadata(metadata, output_dir)
        return

    sentinel_df = extract_sentinel_roi(l1c_safe, l2a_safe, roi_geojson,
                                        USABLE_BANDS, metadata)
    sentinel_df.to_csv(output_dir / "sentinel_roi_reflectance.csv", index=False)
    log(f"Saved sentinel_roi_reflectance.csv ({len(sentinel_df)} bands)")

    # --- B: Read RadCalNet ---
    log("\n--- B. Read RadCalNet ---")
    nc_path = Path(args.radcalnet_nc)
    if not nc_path.exists():
        log(f"RadCalNet NC not found: {nc_path}", "FAIL")
        _save_metadata(metadata, output_dir)
        return

    spec_df, atm_vars = read_radcalnet(nc_path, args.sentinel_time_utc, metadata)
    spec_df.to_csv(output_dir / "radcalnet_selected_spectrum.csv", index=False)

    atm_json = {k: (float(v) if isinstance(v, (int, float, np.floating)) and np.isfinite(v)
                     else v) for k, v in atm_vars.items()}
    with open(output_dir / "radcalnet_atmosphere.json", "w") as f:
        json.dump(atm_json, f, indent=2, default=str)
    log(f"Saved radcalnet_selected_spectrum.csv ({len(spec_df)} wavelengths)")
    log(f"Atmosphere: AOD={atm_vars['aod550']:.4f}, CWV={atm_vars['cwv']:.4f} g/cm2, "
        f"O3={atm_vars['ozone_du']:.1f} DU ({atm_vars['ozone']:.4f} cm-atm), "
        f"P={atm_vars.get('pressure', '?')} hPa, SZA={atm_vars['sza_deg']:.2f}")
    metadata["atmosphere_variables"] = atm_json

    # --- C: SRF integration ---
    log("\n--- C. Integrate RadCalNet to SRFs ---")
    srf_dir = Path(args.srf_dir)
    srf_df = integrate_radcalnet_to_srf(spec_df, srf_dir, USABLE_BANDS)
    srf_df.to_csv(output_dir / "radcalnet_srf_integrated.csv", index=False)
    log(f"Saved radcalnet_srf_integrated.csv")
    for _, r in srf_df.iterrows():
        log(f"  {r['band']}: RadCalNet SRF = {r['rho_radcalnet_srf']:.6f}")

    # --- D: Run 6S ---
    log("\n--- D. Run 6S ---")
    aerosol_model = _map_aerosol_for_site(args.site_name,
                                           atm_vars.get("aerosol_type_code"))
    atm_profile = _map_atm_profile_for_site(args.roi_center_lat)
    metadata["aerosol_model_selected"] = aerosol_model
    metadata["atmospheric_profile_selected"] = atm_profile

    sixs_df = run_6s_for_bands(USABLE_BANDS, atm_vars, srf_dir,
                                args.vza, args.vaa, aerosol_model,
                                atm_profile, metadata)

    if sixs_df is not None:
        sixs_df.to_csv(output_dir / "sixs_coefficients.csv", index=False)
        log("Saved sixs_coefficients.csv")
    else:
        log("6S FAILED — saving partial outputs and stopping RTM pipeline.", "FAIL")
        _save_metadata(metadata, output_dir)
        return

    # --- E: Run pKANrtm ---
    log("\n--- E. Run pKANrtm ---")
    checkpoint_path = Path(args.checkpoint)
    split_dir = checkpoint_path.parent.parent.parent

    search_dirs = [
        split_dir / "preprocess",
        split_dir,
        checkpoint_path.parent,
        checkpoint_path.parent.parent,
    ]

    if args.feature_scaler:
        preprocessor_path = Path(args.feature_scaler)
    else:
        preprocessor_path = _search_artifact(search_dirs, ["preprocessor.joblib"])

    if args.target_scaler:
        target_scaler_path = Path(args.target_scaler)
    else:
        target_scaler_path = _search_artifact(search_dirs,
                                               ["target_scaler.joblib", "scaler.joblib",
                                                "y_scaler.pkl", "target_scaler.pkl"])

    log(f"Checkpoint:    {checkpoint_path}")
    log(f"Preprocessor:  {preprocessor_path}")
    log(f"Target scaler: {target_scaler_path}")
    metadata["preprocessor_path"] = str(preprocessor_path) if preprocessor_path else None
    metadata["target_scaler_path"] = str(target_scaler_path) if target_scaler_path else None

    pkanrtm_df = run_pkanrtm(sixs_df, atm_vars, checkpoint_path,
                              preprocessor_path, target_scaler_path,
                              USABLE_BANDS, aerosol_model, atm_profile, metadata)

    if pkanrtm_df is not None:
        pkanrtm_df.to_csv(output_dir / "pkanrtm_coefficients.csv", index=False)
        log("Saved pkanrtm_coefficients.csv")

    # --- G: libRadtran (optional) ---
    libradtran_df = None
    has_libradtran = False
    if args.run_libradtran:
        log("\n--- G. Run libRadtran ---")
        libradtran_df = run_libradtran_for_bands(args, USABLE_BANDS, atm_vars,
                                                  srf_dir, metadata)
        has_libradtran = libradtran_df is not None
        if has_libradtran:
            libradtran_df.to_csv(output_dir / "libradtran_coefficients.csv", index=False)
    else:
        metadata["libradtran_status"] = "not_requested"

    # --- G2: Center-wavelength 6S (optional) ---
    sixs_center_df = None
    has_6s_center = False
    if args.run_6s_center:
        log("\n--- G2. Run 6S at center wavelengths ---")
        sixs_center_df = run_6s_center_wavelength(
            USABLE_BANDS, atm_vars, args.vza, args.vaa,
            aerosol_model, atm_profile, metadata)
        has_6s_center = sixs_center_df is not None
        if has_6s_center:
            sixs_center_df.to_csv(output_dir / "sixs_center_coefficients.csv", index=False)
            log("Saved sixs_center_coefficients.csv")
    else:
        metadata["sixs_center_status"] = "not_requested"

    # --- G3: Center-wavelength libRadtran (optional) ---
    lrt_center_df = None
    has_lrt_center = False
    if args.run_libradtran_center:
        log("\n--- G3. Run libRadtran at center wavelengths ---")
        lrt_center_df = run_libradtran_center_wavelength(
            USABLE_BANDS, atm_vars, args.vza, args.vaa,
            aerosol_model, atm_profile, metadata,
            uvspec_bin=args.uvspec_bin)
        has_lrt_center = lrt_center_df is not None
        if has_lrt_center:
            lrt_center_df.to_csv(output_dir / "libradtran_center_coefficients.csv", index=False)
            log("Saved libradtran_center_coefficients.csv")
    else:
        metadata["libradtran_center_status"] = "not_requested"

    # --- F: Retrieve surface reflectance ---
    log("\n--- F. Retrieve surface reflectance ---")
    refl_df = retrieve_surface_reflectance(
        sentinel_df, sixs_df, pkanrtm_df, libradtran_df, USABLE_BANDS,
        sixs_center_df=sixs_center_df, lrt_center_df=lrt_center_df)
    refl_df.to_csv(output_dir / "retrieved_surface_reflectance.csv", index=False)
    log("Saved retrieved_surface_reflectance.csv")
    for _, r in refl_df.iterrows():
        parts = [f"{r['band']}: TOA={r['rho_toa_l1c']:.4f}"]
        if np.isfinite(r.get("rho_6s", np.nan)):
            parts.append(f"6S-SRF={r['rho_6s']:.4f}")
        if np.isfinite(r.get("rho_pkanrtm", np.nan)):
            parts.append(f"pKANrtm={r['rho_pkanrtm']:.4f}")
        if np.isfinite(r.get("rho_6s_center", np.nan)):
            parts.append(f"6S-ctr={r['rho_6s_center']:.4f}")
        if np.isfinite(r.get("rho_lrt_center", np.nan)):
            parts.append(f"lRT-ctr={r['rho_lrt_center']:.4f}")
        log("  " + ", ".join(parts))

    # --- H: Comparison table ---
    log("\n--- H. Build comparison table ---")
    has_pkanrtm = pkanrtm_df is not None
    band_table = build_band_table(sentinel_df, srf_df, refl_df,
                                   USABLE_BANDS, has_libradtran,
                                   has_6s_center=has_6s_center,
                                   has_lrt_center=has_lrt_center)
    band_table.to_csv(output_dir / "band_table.csv", index=False)
    log("Saved band_table.csv")

    # --- I: Metrics ---
    log("\n--- I. Compute metrics ---")
    metrics_rc, metrics_l2a, metrics_lrt = compute_all_metrics(
        band_table, has_libradtran,
        has_6s_center=has_6s_center, has_lrt_center=has_lrt_center)
    metrics_rc.to_csv(output_dir / "metrics_radcalnet.csv", index=False)
    metrics_l2a.to_csv(output_dir / "metrics_l2a.csv", index=False)
    if metrics_lrt is not None:
        metrics_lrt.to_csv(output_dir / "metrics_libradtran.csv", index=False)

    log("Saved metrics_radcalnet.csv, metrics_l2a.csv")
    log("\nMetrics vs RadCalNet:")
    for _, r in metrics_rc.iterrows():
        log(f"  {r['comparison']}: RMSE={r['rmse']:.6f}, MAE={r['mae']:.6f}, "
            f"MRD={r['mean_rel_diff_pct']:.2f}%, MARD={r['mean_abs_rel_diff_pct']:.2f}%")
    log("\nMetrics vs L2A:")
    for _, r in metrics_l2a.iterrows():
        log(f"  {r['comparison']}: RMSE={r['rmse']:.6f}, MAE={r['mae']:.6f}, "
            f"MRD={r['mean_rel_diff_pct']:.2f}%, MARD={r['mean_abs_rel_diff_pct']:.2f}%")

    # --- J: Plots ---
    log("\n--- J. Generate plots ---")
    create_plots(band_table, output_dir, has_pkanrtm, has_libradtran,
                 has_6s_center=has_6s_center, has_lrt_center=has_lrt_center)

    # --- K/L: Metadata ---
    if args.run_libradtran:
        metadata["libradtran_note"] = (
            "Direct libRadtran is included as an RTM consistency comparison. "
            "RadCalNet SRF-integrated BOA reflectance remains the observational reference."
        )

    metadata["output_files"] = [f.name for f in output_dir.iterdir() if f.is_file()]
    _save_metadata(metadata, output_dir)

    log("\n" + "=" * 70)
    log("VALIDATION COMPLETE")
    log(f"Outputs: {output_dir}")
    log("=" * 70)


def _save_metadata(metadata: dict, output_dir: Path):
    def _default(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj) if np.isfinite(obj) else None
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return str(obj)

    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=_default)
    log("Saved run_metadata.json")


if __name__ == "__main__":
    main()
