# Matchup Quality Report

**Site:** GONA (Gobabeb, Namibia)
**Date:** 2018-05-25
**Platform:** S2A
**Tile:** T33KWP
**Sentinel Time (UTC):** 08:46:01
**Validation ROI:** 30.0 m radius disk at (-23.599990N, 15.119215E)

**L1C product:** `S2A_MSIL1C_20180525T084601_N0500_R107_T33KWP_20230903T183659.SAFE`
**L2A product:** `S2A_MSIL2A_20180525T084601_N0500_R107_T33KWP_20230904T024512.SAFE`

## Summary

- WARNING: black cut-off / no-data pixels affect the GONA ROI.
- L1C and L2A products match.
- SCL verdict: PASS
- RadCalNet time verdict: PASS (diff = 13.98 min)

## PASS Criteria

| Criterion | Status |
|-----------|--------|
| no-data/black <= 1% | FAIL |
| bad/unknown SCL <= 1% | PASS |
| RadCalNet time <= 30 min | PASS |
| L1C/L2A match | PASS |
| RadCalNet BOA spectrum exists | PASS |

**Usable bands:** B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12
**Excluded bands:** B10

## Verdict

**FAIL** -- DO NOT USE this matchup.

## Issues

- **[FAIL]** L1C B01: 100.0% no-data/black pixels in GONA ROI.
- **[FAIL]** L1C B09: 100.0% no-data/black pixels in GONA ROI.
- **[FAIL]** L2A B01: 100.0% no-data/black pixels in GONA ROI.
- **[FAIL]** L2A B09: 100.0% no-data/black pixels in GONA ROI.
- **[FAIL]** Black cut-off / no-data pixels affect the GONA ROI (> 1%).

## Detailed Log

```
       ========================================================================
       Matchup Quality Check: GONA (Gobabeb, Namibia)  2018-05-25
       ========================================================================
       Validation ROI: 30.0 m radius disk at (-23.599990N, 15.119215E)
       Visual context AOI: ~1 km x 1 km (quicklook only)
       
       --- SECTION 1: Product Identity ---
       L1C SAFE: S2A_MSIL1C_20180525T084601_N0500_R107_T33KWP_20230903T183659.SAFE
       L2A SAFE: S2A_MSIL2A_20180525T084601_N0500_R107_T33KWP_20230904T024512.SAFE
[PASS] L1C product confirmed MSIL1C.
[PASS] L2A product confirmed MSIL2A.
[PASS] L1C and L2A match on platform, date, time, tile.
       Platform: S2A  Date: 2018-05-25  Time: 08:46:01  Tile: T33KWP
       
       --- SECTION 2: Band Inventory ---
[PASS] L1C: all 12 required bands found.
[PASS] L2A: all 12 required bands found.
       B10 excluded from all validation calculations.
       
       --- SECTION 3: SCL Mask ---
[PASS] SCL mask found: T33KWP_20180525T084601_SCL_20m.jp2
       
       L1C quantification value: 10000.0
       L2A quantification value: 10000.0
       L1C radiometric offsets found (e.g. band 0 = -1000.0).
       L2A BOA offsets found (e.g. band 0 = -1000.0).
       
       --- SECTION 4: ROI Spatial & Band Statistics (30 m radius) ---
       Note: Outside-geometry pixels are excluded from ROI statistics.
[FAIL] L1C B01: 100.0% no-data/black pixels in GONA ROI.
[FAIL] L1C B09: 100.0% no-data/black pixels in GONA ROI.
[FAIL] L2A B01: 100.0% no-data/black pixels in GONA ROI.
[FAIL] L2A B09: 100.0% no-data/black pixels in GONA ROI.
       L1C bands with valid ROI stats: 12/12
       L2A bands with valid ROI stats: 12/12
[FAIL] Black cut-off / no-data pixels affect the GONA ROI (> 1%).
[PASS] GONA ROI is inside all checked raster extents.
       
       --- SECTION 5: SCL Classification in ROI (30 m radius) ---
       Note: Outside-geometry pixels are excluded from ROI statistics.
       SCL pixels inside GONA ROI geometry: 7
         SCL   5 (BARE_SOILS):      7 px  (100.00%) [IDEAL]
[PASS] Bad/invalid SCL pixels: 0.000% (<= 1%) -- PASS
       
       --- SECTION 6: RadCalNet Data ---
       RadCalNet NetCDF variables (top-level + groups):
         SiteName: shape=(4,)
         Instrument: shape=(2,)
         Longitude: shape=(1,)
         Latitude: shape=(1,)
         Altitude: shape=(1,)
         [group] Product_Version/
           Version_product: shape=(6,)
           Version_IN: shape=(2,)
           Version_OUT: shape=(2,)
         [group] Time/
           Year: shape=(13,)
           DOY: shape=(13,)
           Hour_UTC: shape=(13,)
           Minutes_UTC: shape=(13,)
           DOY_L: shape=(13,)
           Hour_local: shape=(13,)
           Minutes_local: shape=(13,)
           DecYear_local: shape=(13,)
           Solar_Zenith_Angle: shape=(13,)
           Solar_Azimuth_Angle: shape=(13,)
           Earth_Sun_Distance: shape=(13,)
         [group] Atmosphere_data/
           [group] Pressure/
             Pressure_data: shape=(13,)
             Pressure_unc: shape=(13,)
           [group] Temperature/
             Temperature_data: shape=(13,)
             Temperature_unc: shape=(13,)
           [group] WaterVapor/
             WaterVapor_data: shape=(13,)
             WaterVapor_unc: shape=(13,)
           [group] Ozone/
             Ozone_data: shape=(13,)
             Ozone_unc: shape=(13,)
           [group] AOT/
             AOT_data: shape=(13,)
             AOT_unc: shape=(13,)
           [group] Angstrom/
             Angstrom_data: shape=(13,)
             Angstrom_unc: shape=(13,)
           [group] AerosolType/
             AerosolType: shape=(13,)
         [group] Reflectance/
           Wavelength: shape=(211,)
           [group] Reflectance_BOA/
             Reflectance_BOA_data: shape=(211, 13)
             Reflectance_BOA_unc: shape=(211, 13)
           [group] Reflectance_TOA/
             Reflectance_TOA_data: shape=(211, 13)
             Reflectance_TOA_unc: shape=(211, 13)
       Sentinel time (UTC): 2018-05-25 08:46:01
       Nearest RadCalNet slot (UTC): 2018-05-25 09:00:00
         Local time: 10:00
         Time difference: 14.0 minutes
[PASS] RadCalNet time match: 14.0 min (<= 30 min) -- PASS
         Pressure: 958.58 mbar
         Temperature: 304.4 K
         WaterVapor: 1.33 g/cm2
         Ozone: 319.0 Dobsons
         AOT: 0.037 dimensionless
         Angstrom: 1.058 dimensionless
         Solar Zenith Angle: 52.74 deg
         Solar Azimuth Angle: 34.80 deg
[PASS] BOA reflectance: 181 valid wavelengths (400–2300 nm)
       TOA reflectance: 181 valid wavelengths (400–2300 nm)
[PASS] RadCalNet wavelength coverage supports all VNIR bands: ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09']
       SWIR bands supported by RadCalNet: ['B11', 'B12']
       
       --- SECTION 7: Quicklook ---
[PASS] Quicklook saved: quicklook_roi_check.png
       
       --- SECTION 8: Final Verdict ---
       
         no-data/black <= 1%: FAIL
         bad/unknown SCL <= 1%: PASS
         RadCalNet time <= 30 min: PASS
         L1C/L2A match: PASS
         RadCalNet BOA spectrum exists: PASS
       
       FAIL issues: 5
       WARN issues: 0
       
       Usable bands for validation: ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
       Bands to exclude:            ['B10']
       
       WARNING: black cut-off / no-data pixels affect the GONA ROI.
       L1C and L2A products match.
       SCL verdict: PASS
       RadCalNet time verdict: PASS (diff = 13.98 min)
       
       VERDICT: FAIL
       RECOMMENDATION: DO NOT USE this matchup.
       ========================================================================
```