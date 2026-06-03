#!/usr/bin/env python3
"""
Quality-check script for Sentinel-2 L1C + L2A + RadCalNet matchup.

Verifies whether a given matchup (L1C, L2A, RadCalNet) is suitable for
pKANrtm real-scene RadCalNet atmospheric correction validation.

All site parameters are configurable via CLI arguments (defaults: RVUS).
"""

import argparse
import json
import math
import os
import re
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

_MISSING = []
for _pkg in ("rasterio", "shapely", "pyproj", "netCDF4", "matplotlib"):
    try:
        __import__(_pkg)
    except ImportError:
        _MISSING.append(_pkg)
if _MISSING:
    print(
        f"ERROR: Missing packages in current environment: {', '.join(_MISSING)}\n"
        f"Install with:\n"
        f"  conda activate pylrt && pip install {' '.join(_MISSING)}"
    )
    sys.exit(1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

import netCDF4 as nc
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.enums import Resampling
from pyproj import Transformer
from shapely.geometry import shape, box, Point
from shapely.ops import transform as shapely_transform

# ===================================================================
# CONSTANTS (not site-specific)
# ===================================================================
REQUIRED_BANDS = [
    "B01", "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B09", "B11", "B12",
]
EXCLUDED_BANDS = ["B10"]

BAND_ORDER_INDEX = {
    "B01": 0, "B02": 1, "B03": 2, "B04": 3, "B05": 4,
    "B06": 5, "B07": 6, "B08": 7, "B8A": 8, "B09": 9,
    "B10": 10, "B11": 11, "B12": 12,
}

S2A_BAND_CENTER_NM = {
    "B01": 443, "B02": 490, "B03": 560, "B04": 665,
    "B05": 705, "B06": 740, "B07": 783, "B08": 842,
    "B8A": 865, "B09": 945, "B10": 1375,
    "B11": 1610, "B12": 2190,
}

SCL_LABELS = {
    0: "NO_DATA", 1: "SATURATED_DEFECTIVE", 2: "CAST_SHADOWS",
    3: "CLOUD_SHADOWS", 4: "VEGETATION", 5: "BARE_SOILS",
    6: "WATER", 7: "CLOUD_LOW_PROB", 8: "CLOUD_MED_PROB",
    9: "CLOUD_HIGH_PROB", 10: "THIN_CIRRUS", 11: "SNOW_ICE",
    255: "NO_DATA_FILL",
}
SCL_IDEAL = {5}
SCL_ACCEPTABLE = {4}
SCL_BAD = {0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 255}

RADCALNET_SWIR_BANDS = {"B11", "B12"}

L2A_NATIVE_RES = {
    "B02": "R10m", "B03": "R10m", "B04": "R10m", "B08": "R10m",
    "B05": "R20m", "B06": "R20m", "B07": "R20m", "B8A": "R20m",
    "B11": "R20m", "B12": "R20m",
    "B01": "R60m", "B09": "R60m",
}


# ===================================================================
# CLI
# ===================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Quality-check a Sentinel-2 L1C+L2A+RadCalNet matchup.")

    p.add_argument("--site-name", default="RVUS")
    p.add_argument("--site-long-name", default="Railroad Valley Playa")
    p.add_argument("--expected-platform", default="S2A")
    p.add_argument("--expected-date", default="2018-06-28")
    p.add_argument("--expected-time", default="18:29:21")
    p.add_argument("--expected-tile", default="T11SPC")
    p.add_argument("--roi-center-lat", type=float, default=38.497)
    p.add_argument("--roi-center-lon", type=float, default=-115.690)
    p.add_argument("--roi-radius-m", type=float, default=30)
    p.add_argument("--radcalnet-nc", default=None,
                   help="Path to RadCalNet NetCDF. Default: <data-dir>/RVUS00_2018_179_v04.06.nc")
    p.add_argument("--data-dir", default=None,
                   help="Directory containing SAFE folders and NetCDF. "
                        "Default: <script_dir>/data")
    p.add_argument("--output-dir", default=None,
                   help="Output directory. Default: "
                        "<data-dir>/../results/quality_check_<SITE>_<DATE>_<RADIUS>m")
    return p.parse_args()


# ===================================================================
# HELPERS
# ===================================================================

def _build_roi_circle_geojson(center_lon, center_lat, radius_m, n_segments=64):
    utm_zone = int((center_lon + 180) / 6) + 1
    hemisphere = "north" if center_lat >= 0 else "south"
    utm_crs = f"+proj=utm +zone={utm_zone} +{hemisphere} +datum=WGS84"

    to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    to_4326 = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)

    cx, cy = to_utm.transform(center_lon, center_lat)
    circle_utm = Point(cx, cy).buffer(radius_m, resolution=n_segments)
    circle_4326 = shapely_transform(to_4326.transform, circle_utm)

    coords = list(circle_4326.exterior.coords)
    return {"type": "Polygon", "coordinates": [coords]}


def _build_visual_aoi(center_lon, center_lat, half_side_deg_lon=0.00574,
                      half_side_deg_lat=0.00450):
    return {
        "type": "Polygon",
        "coordinates": [[
            [center_lon - half_side_deg_lon, center_lat - half_side_deg_lat],
            [center_lon + half_side_deg_lon, center_lat - half_side_deg_lat],
            [center_lon + half_side_deg_lon, center_lat + half_side_deg_lat],
            [center_lon - half_side_deg_lon, center_lat + half_side_deg_lat],
            [center_lon - half_side_deg_lon, center_lat - half_side_deg_lat],
        ]],
    }


def _find_safe_folder(data_dir, pattern_keyword, expected_tile, expected_date,
                      expected_time):
    candidates = sorted(
        data_dir.glob(f"S2*_{pattern_keyword}_*_{expected_tile}_*.SAFE"))
    date_compact = expected_date.replace("-", "")
    time_compact = expected_time.replace(":", "")
    for c in candidates:
        if date_compact in c.name and time_compact in c.name:
            return c
    return candidates[0] if candidates else None


def _parse_product_name(safe_path):
    parts = safe_path.name.replace(".SAFE", "").split("_")
    dt_parsed = datetime.strptime(parts[2], "%Y%m%dT%H%M%S")
    return {
        "platform": parts[0], "level": parts[1],
        "date": dt_parsed.strftime("%Y-%m-%d"),
        "time": dt_parsed.strftime("%H:%M:%S"),
        "tile": parts[5],
        "datetime": dt_parsed.replace(tzinfo=timezone.utc),
    }


def _read_xml_metadata(safe_path, level):
    xml_name = "MTD_MSIL1C.xml" if "L1C" in level.upper() else "MTD_MSIL2A.xml"
    xml_path = safe_path / xml_name
    info = {"quantification_value": 10000, "offsets": {}}
    if not xml_path.exists():
        return info
    tree = ET.parse(str(xml_path))
    for elem in tree.getroot().iter():
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        if tag in ("QUANTIFICATION_VALUE", "BOA_QUANTIFICATION_VALUE"):
            info["quantification_value"] = float(elem.text)
        elif tag in ("RADIO_ADD_OFFSET", "BOA_ADD_OFFSET"):
            info["offsets"][int(elem.attrib.get("band_id", -1))] = float(elem.text)
    return info


def _find_band_files(safe_path, level, bands):
    tile_dir = list((safe_path / "GRANULE").iterdir())[0]
    if "L1C" in level.upper():
        img_dir = tile_dir / "IMG_DATA"
        return {b: c[0] for b in bands
                if (c := list(img_dir.glob(f"*_{b}.jp2")))}

    files = {}
    for b in bands:
        res = L2A_NATIVE_RES.get(b, "R20m")
        for try_res in [res, "R20m", "R60m", "R10m"]:
            d = tile_dir / "IMG_DATA" / try_res
            if d.exists() and (c := list(d.glob(f"*_{b}_*.jp2"))):
                files[b] = c[0]
                break
    return files


def _find_scl(safe_path):
    for res in ["R20m", "R60m"]:
        if c := list(safe_path.rglob(f"*_SCL_*{res.replace('R', '')}*.jp2")):
            return c[0]
    if c := list(safe_path.rglob("*SCL*.jp2")):
        return c[0]
    return None


def _project_roi(roi_geojson, target_crs):
    t = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    coords = [t.transform(x, y) for x, y in roi_geojson["coordinates"][0]]
    return {"type": "Polygon", "coordinates": [coords]}


def _roi_inside_raster(roi_proj, ds):
    return box(*ds.bounds).contains(shape(roi_proj))


def _compute_roi_stats(ds, roi_proj, band_idx=1):
    """Stats only for pixels whose centers fall inside the geometry."""
    try:
        out_image, _ = rio_mask(ds, [roi_proj], crop=True, filled=False)
    except Exception:
        return None
    data = out_image[band_idx - 1]
    inside = ~np.ma.getmaskarray(data)
    raw = np.ma.getdata(data)
    n_inside = int(inside.sum())
    if n_inside == 0:
        return dict(count_inside_geometry=0, count_total=0, count_valid=0,
                    count_nodata_inside_geometry=0, valid_percent=0.0,
                    min=np.nan, max=np.nan, mean=np.nan, median=np.nan,
                    std=np.nan, p01=np.nan, p99=np.nan)
    valid = inside & (raw > 0)
    v = raw[valid]
    n_valid = int(valid.sum())
    n_nodata = n_inside - n_valid
    if n_valid == 0:
        return dict(count_inside_geometry=n_inside, count_total=n_inside,
                    count_valid=0, count_nodata_inside_geometry=n_nodata,
                    valid_percent=0.0, min=np.nan, max=np.nan, mean=np.nan,
                    median=np.nan, std=np.nan, p01=np.nan, p99=np.nan)
    return dict(
        count_inside_geometry=n_inside, count_total=n_inside,
        count_valid=n_valid, count_nodata_inside_geometry=n_nodata,
        valid_percent=round(100.0 * n_valid / n_inside, 4),
        min=float(v.min()), max=float(v.max()),
        mean=float(v.mean()), median=float(np.median(v)),
        std=float(v.std()), p01=float(np.percentile(v, 1)),
        p99=float(np.percentile(v, 99)))


def _dn_to_reflectance(dn_stats, quant, offset):
    _keep = {"count_total", "count_valid", "valid_percent",
             "count_inside_geometry", "count_nodata_inside_geometry"}
    return {k: (v if k in _keep else (np.nan if np.isnan(v)
            else round((v + offset) / quant, 6)))
            for k, v in dn_stats.items()}


# ===================================================================
# MAIN
# ===================================================================

def main():
    args = parse_args()

    SITE_NAME = args.site_name
    SITE_LONG_NAME = args.site_long_name
    EXPECTED_PLATFORM = args.expected_platform
    EXPECTED_DATE = args.expected_date
    EXPECTED_TIME = args.expected_time
    EXPECTED_TILE = args.expected_tile
    ROI_CENTER_LAT = args.roi_center_lat
    ROI_CENTER_LON = args.roi_center_lon
    ROI_RADIUS_M = args.roi_radius_m

    SENTINEL_DATETIME = datetime.strptime(
        f"{EXPECTED_DATE}T{EXPECTED_TIME}", "%Y-%m-%dT%H:%M:%S"
    ).replace(tzinfo=timezone.utc)

    SCRIPT_DIR = Path(__file__).resolve().parent
    DATA_DIR = Path(args.data_dir) if args.data_dir else SCRIPT_DIR / "data"

    if args.radcalnet_nc:
        RADCALNET_NC = Path(args.radcalnet_nc)
        if not RADCALNET_NC.is_absolute():
            RADCALNET_NC = Path.cwd() / RADCALNET_NC
    else:
        RADCALNET_NC = DATA_DIR / "RVUS00_2018_179_v04.06.nc"

    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
    else:
        date_compact = EXPECTED_DATE.replace("-", "")
        OUTPUT_DIR = (DATA_DIR.parent / "results" /
                      f"quality_check_{SITE_NAME}_{date_compact}_{int(ROI_RADIUS_M)}m")

    issues = []
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_lines = []

    def log(msg, severity="INFO"):
        prefix = f"[{severity}]" if severity != "INFO" else "      "
        line = f"{prefix} {msg}"
        report_lines.append(line)
        print(line)
        if severity in ("WARN", "FAIL"):
            issues.append((severity, msg))

    ROI_GEOJSON = _build_roi_circle_geojson(ROI_CENTER_LON, ROI_CENTER_LAT,
                                            ROI_RADIUS_M)
    VISUAL_AOI = _build_visual_aoi(ROI_CENTER_LON, ROI_CENTER_LAT)

    log("=" * 72)
    log(f"Matchup Quality Check: {SITE_NAME} ({SITE_LONG_NAME})  {EXPECTED_DATE}")
    log("=" * 72)
    log(f"Validation ROI: {ROI_RADIUS_M} m radius disk at "
        f"({ROI_CENTER_LAT:.6f}N, {ROI_CENTER_LON:.6f}E)")
    log("Visual context AOI: ~1 km x 1 km (quicklook only)")
    log("")

    # --- SECTION 1: Product Identity ---
    log("--- SECTION 1: Product Identity ---")
    l1c_safe = _find_safe_folder(DATA_DIR, "MSIL1C", EXPECTED_TILE,
                                 EXPECTED_DATE, EXPECTED_TIME)
    l2a_safe = _find_safe_folder(DATA_DIR, "MSIL2A", EXPECTED_TILE,
                                 EXPECTED_DATE, EXPECTED_TIME)

    if l1c_safe is None or not l1c_safe.exists():
        log("L1C SAFE folder not found in data directory.", "FAIL")
        _write_outputs(report_lines, issues, OUTPUT_DIR, SITE_NAME,
                       SITE_LONG_NAME, EXPECTED_PLATFORM, EXPECTED_DATE,
                       EXPECTED_TIME, EXPECTED_TILE, ROI_CENTER_LAT,
                       ROI_CENTER_LON, ROI_RADIUS_M)
        return
    if l2a_safe is None or not l2a_safe.exists():
        log("L2A SAFE folder not found in data directory.", "FAIL")
        _write_outputs(report_lines, issues, OUTPUT_DIR, SITE_NAME,
                       SITE_LONG_NAME, EXPECTED_PLATFORM, EXPECTED_DATE,
                       EXPECTED_TIME, EXPECTED_TILE, ROI_CENTER_LAT,
                       ROI_CENTER_LON, ROI_RADIUS_M)
        return

    log(f"L1C SAFE: {l1c_safe.name}")
    log(f"L2A SAFE: {l2a_safe.name}")

    l1c_info = _parse_product_name(l1c_safe)
    l2a_info = _parse_product_name(l2a_safe)

    if "L1C" not in l1c_info["level"].upper():
        log(f"L1C product level is {l1c_info['level']}, expected MSIL1C.", "FAIL")
    else:
        log("L1C product confirmed MSIL1C.", "PASS")

    if "L2A" not in l2a_info["level"].upper():
        log(f"L2A product level is {l2a_info['level']}, expected MSIL2A.", "FAIL")
    else:
        log("L2A product confirmed MSIL2A.", "PASS")

    match_ok = True
    for field in ("platform", "date", "time", "tile"):
        if l1c_info[field] != l2a_info[field]:
            log(f"L1C/L2A mismatch on {field}: "
                f"{l1c_info[field]} vs {l2a_info[field]}.", "FAIL")
            match_ok = False
    if match_ok:
        log("L1C and L2A match on platform, date, time, tile.", "PASS")

    for field, expected in [("platform", EXPECTED_PLATFORM),
                             ("date", EXPECTED_DATE),
                             ("time", EXPECTED_TIME),
                             ("tile", EXPECTED_TILE)]:
        if l1c_info[field] != expected:
            log(f"Expected {field}={expected}, got {l1c_info[field]}.", "FAIL")

    log(f"Platform: {l1c_info['platform']}  Date: {l1c_info['date']}  "
        f"Time: {l1c_info['time']}  Tile: {l1c_info['tile']}")
    log("")

    # --- SECTION 2: Band Inventory ---
    log("--- SECTION 2: Band Inventory ---")
    l1c_bands = _find_band_files(l1c_safe, "L1C", REQUIRED_BANDS)
    l2a_bands = _find_band_files(l2a_safe, "L2A", REQUIRED_BANDS)

    if missing := [b for b in REQUIRED_BANDS if b not in l1c_bands]:
        log(f"L1C missing bands: {missing}", "FAIL")
    else:
        log(f"L1C: all {len(REQUIRED_BANDS)} required bands found.", "PASS")
    if missing := [b for b in REQUIRED_BANDS if b not in l2a_bands]:
        log(f"L2A missing bands: {missing}", "FAIL")
    else:
        log(f"L2A: all {len(REQUIRED_BANDS)} required bands found.", "PASS")
    log("B10 excluded from all validation calculations.")
    log("")

    # --- SECTION 3: SCL Mask ---
    log("--- SECTION 3: SCL Mask ---")
    scl_path = _find_scl(l2a_safe)
    if scl_path is None:
        log("L2A SCL mask not found.", "FAIL")
    else:
        log(f"SCL mask found: {scl_path.name}", "PASS")
    log("")

    # --- Metadata ---
    l1c_meta = _read_xml_metadata(l1c_safe, "L1C")
    l2a_meta = _read_xml_metadata(l2a_safe, "L2A")
    log(f"L1C quantification value: {l1c_meta['quantification_value']}")
    log(f"L2A quantification value: {l2a_meta['quantification_value']}")
    if l1c_meta["offsets"]:
        log(f"L1C radiometric offsets found "
            f"(e.g. band 0 = {l1c_meta['offsets'].get(0, 'N/A')}).")
    if l2a_meta["offsets"]:
        log(f"L2A BOA offsets found "
            f"(e.g. band 0 = {l2a_meta['offsets'].get(0, 'N/A')}).")
    log("")

    # --- SECTION 4: ROI band stats ---
    log(f"--- SECTION 4: ROI Spatial & Band Statistics ({int(ROI_RADIUS_M)} m radius) ---")
    log("Note: Outside-geometry pixels are excluded from ROI statistics.")

    l1c_stats_rows, l2a_stats_rows = [], []
    nodata_affects_roi = False
    nodata_fail = False
    roi_outside_any = False

    def _process_bands(band_dict, level_label, meta_info, stats_rows):
        nonlocal nodata_affects_roi, nodata_fail, roi_outside_any
        for bname in REQUIRED_BANDS:
            if bname not in band_dict:
                stats_rows.append({"band": bname, "status": "MISSING"})
                continue
            try:
                ds = rasterio.open(str(band_dict[bname]))
            except Exception as e:
                if any(k in str(e).lower() for k in ("jp2", "jpeg2000", "openjpeg")):
                    log("Rasterio/GDAL cannot read JP2. Install GDAL/rasterio "
                        "with JPEG2000 support in the pylrt environment.", "FAIL")
                else:
                    log(f"Cannot open {band_dict[bname].name}: {e}", "FAIL")
                stats_rows.append({"band": bname, "status": "READ_ERROR"})
                continue

            roi_proj = _project_roi(ROI_GEOJSON, ds.crs)
            if not _roi_inside_raster(roi_proj, ds):
                log(f"{level_label} {bname}: {SITE_NAME} ROI is NOT inside "
                    "the raster.", "FAIL")
                roi_outside_any = True
                stats_rows.append({"band": bname, "status": "ROI_OUTSIDE"})
                ds.close()
                continue

            stats = _compute_roi_stats(ds, roi_proj)
            ds.close()
            if stats is None:
                stats_rows.append({"band": bname, "status": "STATS_ERROR"})
                continue

            nodata_pct = round(100.0 - stats["valid_percent"], 2)
            if nodata_pct > 1.0:
                log(f"{level_label} {bname}: {nodata_pct}% no-data/black "
                    f"pixels in {SITE_NAME} ROI.", "FAIL")
                nodata_affects_roi = True
                nodata_fail = True
            elif nodata_pct > 0:
                log(f"{level_label} {bname}: {nodata_pct}% no-data/black "
                    f"pixels in {SITE_NAME} ROI.", "WARN")
                nodata_affects_roi = True

            bid = BAND_ORDER_INDEX.get(bname, -1)
            offset = meta_info["offsets"].get(bid, 0.0)
            quant = meta_info["quantification_value"]
            refl = _dn_to_reflectance(stats, quant, offset)

            row = {"band": bname, "status": "OK"}
            row.update({f"dn_{k}": v for k, v in stats.items()})
            row.update({f"refl_{k}": v for k, v in refl.items()})
            stats_rows.append(row)

    _process_bands(l1c_bands, "L1C", l1c_meta, l1c_stats_rows)
    _process_bands(l2a_bands, "L2A", l2a_meta, l2a_stats_rows)

    l1c_ok = sum(1 for r in l1c_stats_rows if r.get("status") == "OK")
    l2a_ok = sum(1 for r in l2a_stats_rows if r.get("status") == "OK")
    log(f"L1C bands with valid ROI stats: {l1c_ok}/{len(REQUIRED_BANDS)}")
    log(f"L2A bands with valid ROI stats: {l2a_ok}/{len(REQUIRED_BANDS)}")

    if nodata_affects_roi:
        if nodata_fail:
            log(f"Black cut-off / no-data pixels affect the "
                f"{SITE_NAME} ROI (> 1%).", "FAIL")
        else:
            log(f"Minor no-data pixels in {SITE_NAME} ROI (<= 1%).", "WARN")
    else:
        log(f"No black cut-off / no-data issue in {SITE_NAME} ROI.", "PASS")

    if roi_outside_any:
        log(f"{SITE_NAME} ROI falls outside at least one raster.", "FAIL")
    else:
        log(f"{SITE_NAME} ROI is inside all checked raster extents.", "PASS")
    log("")

    # --- SECTION 5: SCL ---
    log(f"--- SECTION 5: SCL Classification in ROI ({int(ROI_RADIUS_M)} m radius) ---")
    log("Note: Outside-geometry pixels are excluded from ROI statistics.")
    scl_counts = {}
    scl_verdict = "N/A"
    if scl_path is not None:
        try:
            ds_scl = rasterio.open(str(scl_path))
            roi_proj_scl = _project_roi(ROI_GEOJSON, ds_scl.crs)
            scl_crop, _ = rio_mask(ds_scl, [roi_proj_scl], crop=True,
                                   filled=False)
            scl_ma = scl_crop[0]
            inside = ~np.ma.getmaskarray(scl_ma)
            scl_inside = np.ma.getdata(scl_ma)[inside]
            ds_scl.close()

            unique, counts = np.unique(scl_inside, return_counts=True)
            total_px = len(scl_inside)
            for u, c in zip(unique, counts):
                scl_counts[int(u)] = int(c)

            log(f"SCL pixels inside {SITE_NAME} ROI geometry: {total_px}")
            for cls_id in sorted(scl_counts):
                pct = 100.0 * scl_counts[cls_id] / total_px
                label = SCL_LABELS.get(cls_id, f"UNKNOWN_{cls_id}")
                marker = (" [IDEAL]" if cls_id in SCL_IDEAL
                          else " [ACCEPTABLE]" if cls_id in SCL_ACCEPTABLE
                          else " [BAD/INVALID]" if cls_id in SCL_BAD else "")
                log(f"  SCL {cls_id:3d} ({label}): "
                    f"{scl_counts[cls_id]:6d} px  ({pct:6.2f}%){marker}")

            bad_px = sum(scl_counts.get(c, 0) for c in SCL_BAD)
            bad_pct = 100.0 * bad_px / total_px if total_px else 0.0

            if bad_pct > 1.0:
                scl_verdict = "FAIL"
                log(f"Bad/invalid SCL pixels: {bad_pct:.3f}% (> 1%) -- FAIL",
                    "FAIL")
            else:
                scl_verdict = "PASS"
                log(f"Bad/invalid SCL pixels: {bad_pct:.3f}% (<= 1%) -- PASS",
                    "PASS")
        except Exception as e:
            log(f"Error reading SCL: {e}", "FAIL")
    log("")

    # --- SECTION 6: RadCalNet ---
    log("--- SECTION 6: RadCalNet Data ---")
    radcalnet_summary = {}
    radcalnet_time_verdict = "N/A"
    boa_exists = False

    if not RADCALNET_NC.exists():
        log(f"RadCalNet file not found: {RADCALNET_NC}", "FAIL")
    else:
        ds_nc = nc.Dataset(str(RADCALNET_NC))
        log("RadCalNet NetCDF variables (top-level + groups):")
        _print_nc_vars(ds_nc, report_lines, indent=2)

        tg = ds_nc.groups["Time"]
        hours_utc = tg.variables["Hour_UTC"][:]
        mins_utc = tg.variables["Minutes_UTC"][:]
        hours_local = tg.variables["Hour_local"][:]
        mins_local = tg.variables["Minutes_local"][:]
        year_arr = tg.variables["Year"][:]
        doy_arr = tg.variables["DOY"][:]

        rc_dts = [datetime(int(year_arr[i]), 1, 1, int(hours_utc[i]),
                           int(mins_utc[i]), 0, tzinfo=timezone.utc)
                  + timedelta(days=int(doy_arr[i]) - 1)
                  for i in range(len(hours_utc))]

        diffs = [abs((dt - SENTINEL_DATETIME).total_seconds()) for dt in rc_dts]
        bi = int(np.argmin(diffs))
        best_diff_min = diffs[bi] / 60.0
        best_dt = rc_dts[bi]

        log(f"Sentinel time (UTC): {SENTINEL_DATETIME:%Y-%m-%d %H:%M:%S}")
        log(f"Nearest RadCalNet slot (UTC): {best_dt:%Y-%m-%d %H:%M:%S}")
        log(f"  Local time: {int(hours_local[bi]):02d}:{int(mins_local[bi]):02d}")
        log(f"  Time difference: {best_diff_min:.1f} minutes")

        if best_diff_min <= 30:
            radcalnet_time_verdict = "PASS"
            log(f"RadCalNet time match: {best_diff_min:.1f} min "
                "(<= 30 min) -- PASS", "PASS")
        elif best_diff_min <= 60:
            radcalnet_time_verdict = "WARN"
            log(f"RadCalNet time match: {best_diff_min:.1f} min "
                "(30-60 min) -- WARN", "WARN")
        else:
            radcalnet_time_verdict = "FAIL"
            log(f"RadCalNet time match: {best_diff_min:.1f} min "
                "(> 60 min) -- FAIL", "FAIL")

        atm = ds_nc.groups["Atmosphere_data"]
        atm_data = {}
        for sub in ["Pressure", "Temperature", "WaterVapor", "Ozone",
                     "AOT", "Angstrom"]:
            sg = atm.groups[sub]
            for vn, vv in sg.variables.items():
                if "_unc" not in vn:
                    val = float(vv[bi])
                    units = getattr(vv, "units", "")
                    atm_data[sub] = {"value": val, "units": units}
                    log(f"  {sub}: {val} {units}")

        radcalnet_summary.update({
            "AOD550": atm_data.get("AOT", {}).get("value", np.nan),
            "Angstrom": atm_data.get("Angstrom", {}).get("value", np.nan),
            "CWV_g_cm2": atm_data.get("WaterVapor", {}).get("value", np.nan),
            "Ozone_DU": atm_data.get("Ozone", {}).get("value", np.nan),
            "Pressure_mbar": atm_data.get("Pressure", {}).get("value", np.nan),
            "Temperature_K": atm_data.get("Temperature", {}).get("value", np.nan),
        })

        sza = float(tg.variables["Solar_Zenith_Angle"][bi])
        saa = float(tg.variables["Solar_Azimuth_Angle"][bi])
        log(f"  Solar Zenith Angle: {sza:.2f} deg")
        log(f"  Solar Azimuth Angle: {saa:.2f} deg")
        radcalnet_summary["SZA_deg"] = sza
        radcalnet_summary["SAA_deg"] = saa

        refl_grp = ds_nc.groups["Reflectance"]
        wavelengths = refl_grp.variables["Wavelength"][:]
        boa = refl_grp.groups["Reflectance_BOA"].variables[
            "Reflectance_BOA_data"][:, bi]
        toa = refl_grp.groups["Reflectance_TOA"].variables[
            "Reflectance_TOA_data"][:, bi]

        vb = (boa < 9000) & (boa >= 0)
        vt = (toa < 9000) & (toa >= 0)

        if vb.sum() > 0:
            boa_exists = True
            log(f"BOA reflectance: {int(vb.sum())} valid wavelengths "
                f"({float(wavelengths[vb].min()):.0f}–"
                f"{float(wavelengths[vb].max()):.0f} nm)", "PASS")
        else:
            log("BOA reflectance: NO valid data.", "FAIL")

        if vt.sum() > 0:
            log(f"TOA reflectance: {int(vt.sum())} valid wavelengths "
                f"({float(wavelengths[vt].min()):.0f}–"
                f"{float(wavelengths[vt].max()):.0f} nm)")
        else:
            log("TOA reflectance: NO valid data.")

        wl_lo = float(wavelengths[vb].min()) if vb.sum() else 9999
        wl_hi = float(wavelengths[vb].max()) if vb.sum() else 0

        vnir = [b for b in REQUIRED_BANDS if b not in RADCALNET_SWIR_BANDS]
        sup_v = [b for b in vnir if wl_lo <= S2A_BAND_CENTER_NM[b] <= wl_hi]
        unsup_v = [b for b in vnir if b not in sup_v]
        sup_s = [b for b in RADCALNET_SWIR_BANDS
                 if wl_lo <= S2A_BAND_CENTER_NM[b] <= wl_hi]
        unsup_s = [b for b in RADCALNET_SWIR_BANDS if b not in sup_s]

        if len(sup_v) >= len(vnir):
            log(f"RadCalNet wavelength coverage supports all VNIR bands: "
                f"{sup_v}", "PASS")
        else:
            log(f"RadCalNet wavelength coverage missing VNIR bands: "
                f"{unsup_v}", "WARN")
        if sup_s:
            log(f"SWIR bands supported by RadCalNet: {sup_s}")
        if unsup_s:
            log(f"SWIR bands NOT supported by RadCalNet: {unsup_s}")

        rc_rows = [{"wavelength_nm": float(wavelengths[i]),
                     "BOA_reflectance": float(boa[i]) if boa[i] < 9000 else np.nan,
                     "TOA_reflectance": float(toa[i]) if toa[i] < 9000 else np.nan}
                    for i in range(len(wavelengths))]
        pd.DataFrame(rc_rows).to_csv(OUTPUT_DIR / "radcalnet_summary.csv",
                                     index=False)

        radcalnet_summary.update({
            "time_diff_min": round(best_diff_min, 2),
            "nearest_utc": f"{best_dt:%Y-%m-%d %H:%M:%S}",
            "nearest_local": f"{int(hours_local[bi]):02d}:"
                             f"{int(mins_local[bi]):02d}",
            "supported_vnir_bands": sup_v,
            "unsupported_vnir_bands": unsup_v,
            "swir_supported": sup_s,
            "swir_unsupported": unsup_s,
        })
        ds_nc.close()
    log("")

    # --- SECTION 7: Quicklook ---
    log("--- SECTION 7: Quicklook ---")
    _make_quicklook(l1c_bands, ROI_GEOJSON, VISUAL_AOI, OUTPUT_DIR, log,
                    SITE_NAME, SITE_LONG_NAME, EXPECTED_DATE, ROI_RADIUS_M)
    log("")

    # --- Save CSVs ---
    if l1c_stats_rows:
        pd.DataFrame(l1c_stats_rows).to_csv(
            OUTPUT_DIR / "l1c_roi_stats.csv", index=False)
    if l2a_stats_rows:
        pd.DataFrame(l2a_stats_rows).to_csv(
            OUTPUT_DIR / "l2a_roi_stats.csv", index=False)
    if scl_counts:
        total_scl = sum(scl_counts.values())
        pd.DataFrame([{
            "scl_class": c,
            "label": SCL_LABELS.get(c, f"UNKNOWN_{c}"),
            "count": scl_counts[c],
            "percent": round(100.0 * scl_counts[c] / total_scl, 4),
            "category": ("IDEAL" if c in SCL_IDEAL
                         else "ACCEPTABLE" if c in SCL_ACCEPTABLE
                         else "BAD" if c in SCL_BAD else "OTHER"),
        } for c in sorted(scl_counts)]).to_csv(
            OUTPUT_DIR / "l2a_scl_roi_counts.csv", index=False)

    # --- SECTION 8: Final Verdict ---
    log("--- SECTION 8: Final Verdict ---")
    pass_criteria = {
        "no-data/black <= 1%": not nodata_fail,
        "bad/unknown SCL <= 1%": scl_verdict == "PASS",
        "RadCalNet time <= 30 min": radcalnet_time_verdict == "PASS",
        "L1C/L2A match": match_ok,
        "RadCalNet BOA spectrum exists": boa_exists,
    }
    log("")
    all_pass = True
    for crit, passed in pass_criteria.items():
        if not passed:
            all_pass = False
        log(f"  {crit}: {'PASS' if passed else 'FAIL'}")

    fail_count = sum(1 for s, _ in issues if s == "FAIL")
    warn_count = sum(1 for s, _ in issues if s == "WARN")
    log("")
    log(f"FAIL issues: {fail_count}")
    log(f"WARN issues: {warn_count}")

    if all_pass and fail_count == 0:
        verdict = "WARN" if warn_count > 0 else "PASS"
        recommendation = ("USE WITH CAVEATS." if warn_count > 0
                          else "USE this matchup.")
    else:
        verdict = "FAIL"
        recommendation = "DO NOT USE this matchup."

    band_status = {}
    for r in l1c_stats_rows:
        band_status.setdefault(r["band"], {})["l1c"] = r.get("status")
    for r in l2a_stats_rows:
        band_status.setdefault(r["band"], {})["l2a"] = r.get("status")

    usable_bands = []
    excluded_bands_list = list(EXCLUDED_BANDS)
    for b in REQUIRED_BANDS:
        bs = band_status.get(b, {})
        if bs.get("l1c") == "OK" and bs.get("l2a") == "OK":
            sup_all = (radcalnet_summary.get("supported_vnir_bands", []) +
                       radcalnet_summary.get("swir_supported", []))
            if b in sup_all:
                usable_bands.append(b)
            elif b in radcalnet_summary.get("swir_unsupported", []):
                excluded_bands_list.append(b)
                log(f"Band {b}: L1C/L2A OK but RadCalNet SWIR not "
                    "covered -- exclude.")
            elif b in radcalnet_summary.get("unsupported_vnir_bands", []):
                excluded_bands_list.append(b)
                log(f"Band {b}: L1C/L2A OK but RadCalNet coverage "
                    "gap -- exclude.")
            else:
                usable_bands.append(b)
        else:
            excluded_bands_list.append(b)

    log("")
    log(f"Usable bands for validation: {usable_bands}")
    log(f"Bands to exclude:            {excluded_bands_list}")
    log("")

    nodata_summary = (
        f"No black cut-off / no-data issue in {SITE_NAME} ROI."
        if not nodata_affects_roi
        else f"WARNING: black cut-off / no-data pixels affect the "
             f"{SITE_NAME} ROI.")
    match_summary = ("L1C and L2A products match." if match_ok
                     else "L1C and L2A products DO NOT match.")
    scl_summary = f"SCL verdict: {scl_verdict}"
    rcn_summary = (f"RadCalNet time verdict: {radcalnet_time_verdict} "
                   f"(diff = {radcalnet_summary.get('time_diff_min', 'N/A')} min)")

    log(nodata_summary)
    log(match_summary)
    log(scl_summary)
    log(rcn_summary)
    log("")
    log(f"VERDICT: {verdict}")
    log(f"RECOMMENDATION: {recommendation}")
    log("=" * 72)

    _write_outputs(report_lines, issues, OUTPUT_DIR,
                   SITE_NAME, SITE_LONG_NAME, EXPECTED_PLATFORM,
                   EXPECTED_DATE, EXPECTED_TIME, EXPECTED_TILE,
                   ROI_CENTER_LAT, ROI_CENTER_LON, ROI_RADIUS_M,
                   verdict=verdict, recommendation=recommendation,
                   nodata_summary=nodata_summary, match_summary=match_summary,
                   match_ok=match_ok,
                   scl_summary=scl_summary, scl_verdict=scl_verdict,
                   rcn_summary=rcn_summary,
                   radcalnet_time_verdict=radcalnet_time_verdict,
                   usable_bands=usable_bands,
                   excluded_bands=excluded_bands_list,
                   radcalnet_summary=radcalnet_summary,
                   l1c_safe=l1c_safe, l2a_safe=l2a_safe,
                   pass_criteria=pass_criteria)


# ===================================================================
# OUTPUT HELPERS
# ===================================================================

def _print_nc_vars(ds_nc, report_lines, indent=0):
    prefix = " " * indent
    for vn in ds_nc.variables:
        line = f"{prefix}{vn}: shape={ds_nc.variables[vn].shape}"
        report_lines.append(f"       {line}")
        print(f"       {line}")
    for gn, grp in ds_nc.groups.items():
        line = f"{prefix}[group] {gn}/"
        report_lines.append(f"       {line}")
        print(f"       {line}")
        _print_nc_vars(grp, report_lines, indent + 2)


def _make_quicklook(l1c_bands, roi_geojson, visual_aoi, output_dir, log,
                    site_name, site_long_name, expected_date, roi_radius_m):
    rgb_bands = ["B04", "B03", "B02"]
    rgb_paths = [l1c_bands.get(b) for b in rgb_bands]
    if any(p is None for p in rgb_paths):
        log("Cannot create quicklook: missing RGB bands.", "WARN")
        return
    try:
        datasets = [rasterio.open(str(p)) for p in rgb_paths]
    except Exception as e:
        log(f"Cannot open bands for quicklook: {e}", "WARN")
        return

    ref_ds = datasets[0]
    visual_proj = _project_roi(visual_aoi, ref_ds.crs)
    roi_proj = _project_roi(roi_geojson, ref_ds.crs)
    vb = shape(visual_proj).bounds

    pad = 2000
    wb = (vb[0] - pad, vb[1] - pad, vb[2] + pad, vb[3] + pad)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    rgb_arrays = []
    target_shape = None
    for ds in datasets:
        window = rasterio.windows.from_bounds(*wb, transform=ds.transform)
        window = window.round_lengths().round_offsets()
        r, c = max(0, int(window.row_off)), max(0, int(window.col_off))
        h = min(int(window.height), ds.height - r)
        w = min(int(window.width), ds.width - c)
        aw = rasterio.windows.Window(c, r, w, h)
        data = ds.read(1, window=aw).astype(float)
        if target_shape is None:
            target_shape = data.shape
        elif data.shape != target_shape:
            from rasterio.warp import reproject
            out = np.empty(target_shape, dtype=np.float64)
            reproject(data, out,
                      src_transform=ds.window_transform(aw),
                      src_crs=ds.crs,
                      dst_transform=ref_ds.window_transform(
                          rasterio.windows.Window(
                              max(0, int(rasterio.windows.from_bounds(
                                  *wb, transform=ref_ds.transform
                              ).round_offsets().col_off)),
                              max(0, int(rasterio.windows.from_bounds(
                                  *wb, transform=ref_ds.transform
                              ).round_offsets().row_off)),
                              target_shape[1], target_shape[0])),
                      dst_crs=ref_ds.crs,
                      resampling=Resampling.bilinear)
            data = out
        rgb_arrays.append(data)

    rgb = np.stack(rgb_arrays, axis=-1)
    p2, p98 = (np.percentile(rgb[rgb > 0], [2, 98])
               if (rgb > 0).any() else (0, 1))
    rgb_s = np.clip((rgb - p2) / (p98 - p2 + 1e-10), 0, 1)

    extent = [wb[0], wb[2], wb[1], wb[3]]
    ax.imshow(rgb_s, extent=extent, origin="upper", aspect="equal")

    aoi_c = np.array(visual_proj["coordinates"][0])
    ax.add_patch(MplPolygon(aoi_c, closed=True, fill=False,
                            edgecolor="cyan", linewidth=1.5, linestyle="--",
                            label="~1 km visual AOI"))
    roi_c = np.array(roi_proj["coordinates"][0])
    ax.add_patch(MplPolygon(roi_c, closed=True, fill=False,
                            edgecolor="red", linewidth=2.5, linestyle="-",
                            label=f"{int(roi_radius_m)} m validation ROI"))

    ax.legend(loc="upper right", fontsize=10)
    ax.set_title(f"RGB Quicklook (B04/B03/B02) – {site_name} "
                 f"({site_long_name})\n{expected_date}  |  "
                 f"ROI: {int(roi_radius_m)} m radius", fontsize=12)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")

    fig.savefig(str(output_dir / "quicklook_roi_check.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    log(f"Quicklook saved: quicklook_roi_check.png", "PASS")
    for ds in datasets:
        ds.close()


def _write_outputs(report_lines, issues, output_dir,
                   site_name, site_long_name, expected_platform,
                   expected_date, expected_time, expected_tile,
                   roi_center_lat, roi_center_lon, roi_radius_m,
                   **kwargs):
    output_dir.mkdir(parents=True, exist_ok=True)
    verdict = kwargs.get("verdict", "FAIL")
    recommendation = kwargs.get("recommendation", "DO NOT USE this matchup.")

    with open(output_dir / "verdict.txt", "w") as f:
        f.write(f"VERDICT: {verdict}\nRECOMMENDATION: {recommendation}\n")

    md = [
        "# Matchup Quality Report", "",
        f"**Site:** {site_name} ({site_long_name})",
        f"**Date:** {expected_date}",
        f"**Platform:** {expected_platform}",
        f"**Tile:** {expected_tile}",
        f"**Sentinel Time (UTC):** {expected_time}",
        f"**Validation ROI:** {roi_radius_m} m radius disk at "
        f"({roi_center_lat:.6f}N, {roi_center_lon:.6f}E)", "",
    ]
    for key in ("l1c_safe", "l2a_safe"):
        if s := kwargs.get(key):
            label = "L1C" if "l1c" in key else "L2A"
            md.append(f"**{label} product:** `{s.name}`")
    md.append("")
    md += ["## Summary", "",
           f"- {kwargs.get('nodata_summary', 'N/A')}",
           f"- {kwargs.get('match_summary', 'N/A')}",
           f"- {kwargs.get('scl_summary', 'N/A')}",
           f"- {kwargs.get('rcn_summary', 'N/A')}", ""]

    if pc := kwargs.get("pass_criteria", {}):
        md += ["## PASS Criteria", "", "| Criterion | Status |",
               "|-----------|--------|"]
        for c, p in pc.items():
            md.append(f"| {c} | {'PASS' if p else 'FAIL'} |")
        md.append("")

    usable = kwargs.get("usable_bands", [])
    excluded = kwargs.get("excluded_bands", [])
    md += [f"**Usable bands:** {', '.join(usable) if usable else 'None'}",
           f"**Excluded bands:** {', '.join(excluded) if excluded else 'None'}",
           "", "## Verdict", "",
           f"**{verdict}** -- {recommendation}", ""]

    if issues:
        md += ["## Issues", ""]
        for sev, msg in issues:
            md.append(f"- **[{sev}]** {msg}")
        md.append("")

    md += ["## Detailed Log", "", "```"] + report_lines + ["```"]

    with open(output_dir / "matchup_quality_report.md", "w") as f:
        f.write("\n".join(md))

    summary = {
        "site": site_name, "site_long_name": site_long_name,
        "date": expected_date, "platform": expected_platform,
        "tile": expected_tile, "sentinel_time_utc": expected_time,
        "roi_center_lat": roi_center_lat, "roi_center_lon": roi_center_lon,
        "roi_radius_m": roi_radius_m,
        "verdict": verdict, "recommendation": recommendation,
        "pass_criteria": dict(kwargs.get("pass_criteria", {})),
        "fail_count": sum(1 for s, _ in issues if s == "FAIL"),
        "warn_count": sum(1 for s, _ in issues if s == "WARN"),
        "nodata_affects_roi": kwargs.get("nodata_summary", "").startswith("WARNING"),
        "l1c_l2a_match": kwargs.get("match_ok", False),
        "scl_verdict": kwargs.get("scl_verdict", "N/A"),
        "radcalnet_time_verdict": kwargs.get("radcalnet_time_verdict", "N/A"),
        "usable_bands": usable, "excluded_bands": excluded,
        "radcalnet": kwargs.get("radcalnet_summary", {}),
        "issues": [{"severity": s, "message": m} for s, m in issues],
    }

    class _Enc(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, np.integer): return int(o)
            if isinstance(o, np.floating): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            return super().default(o)

    with open(output_dir / "matchup_quality_summary.json", "w") as f:
        json.dump(summary, f, indent=2, cls=_Enc)


if __name__ == "__main__":
    main()
