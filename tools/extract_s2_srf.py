from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Vendor spreadsheet: prefer data/original/; fallback to data/ (repo root data folder)
_XLSX_NAME = "COPE-GSEG-EOPG-TN-15-0007 - Sentinel-2 Spectral Response Functions 2024 - 4.0.xlsx"
_xlsx_primary = PROJECT_ROOT / "data" / "original" / _XLSX_NAME
_xlsx_fallback = PROJECT_ROOT / "data" / _XLSX_NAME
xlsx = _xlsx_primary if _xlsx_primary.exists() else _xlsx_fallback

sheet_map = {
    "S2A": "Spectral Responses (S2A)",
    "S2B": "Spectral Responses (S2B)",
    "S2C": "Spectral Responses (S2C)",
}

col_prefix = {
    "S2A": "S2A_SR_AV_",
    "S2B": "S2B_SR_AV_",
    "S2C": "S2C_SR_AV_",
}

# Matches data_generator/s2_srf.py: data/processed/s2_srf/<S2A|S2B|S2C>/*.csv
outroot = PROJECT_ROOT / "data" / "processed" / "s2_srf"
outroot.mkdir(parents=True, exist_ok=True)

band_map = {
    "B1":  "B01",
    "B2":  "B02",
    "B3":  "B03",
    "B4":  "B04",
    "B5":  "B05",
    "B6":  "B06",
    "B7":  "B07",
    "B8":  "B08",
    "B8A": "B8A",
    "B9":  "B09",
    "B10": "B10",
    "B11": "B11",
    "B12": "B12",
}

wl_col = "SR_WL"

if not xlsx.exists():
    raise FileNotFoundError(
        f"SRF spreadsheet not found. Place it at:\n  {_xlsx_primary}\n"
        f"or:\n  {_xlsx_fallback}"
    )

total_written = 0
for platform in ("S2A", "S2B", "S2C"):
    df = pd.read_excel(xlsx, sheet_name=sheet_map[platform])
    outdir = outroot / platform
    outdir.mkdir(parents=True, exist_ok=True)

    for src_band, out_band in band_map.items():
        sr_col = f"{col_prefix[platform]}{src_band}"
        tmp = df[[wl_col, sr_col]].copy()
        tmp.columns = ["wavelength_nm", "response"]

        tmp["wavelength_nm"] = pd.to_numeric(tmp["wavelength_nm"], errors="coerce")
        tmp["response"] = pd.to_numeric(tmp["response"], errors="coerce")
        tmp = tmp.dropna()

        tmp = tmp[tmp["response"] > 0].copy()
        tmp = tmp.sort_values("wavelength_nm").reset_index(drop=True)

        tmp.to_csv(outdir / f"{out_band}.csv", index=False)
        total_written += 1

print(f"Done. Wrote {total_written} files under {outroot.resolve()}")