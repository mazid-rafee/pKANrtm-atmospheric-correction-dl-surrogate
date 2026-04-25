import argparse
import datetime
import json
import math
import shutil
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
try:
    from .s2_srf import convolve_to_band, load_srf
except ImportError:
    from s2_srf import convolve_to_band, load_srf

try:
    from Py6S import AeroProfile, AtmosProfile, Geometry, GroundReflectance, SixS, Wavelength
except Exception as exc:  # pragma: no cover - runtime dependency check
    AeroProfile = None
    AtmosProfile = None
    Geometry = None
    GroundReflectance = None
    SixS = None
    Wavelength = None
    _PY6S_IMPORT_ERROR = exc
else:
    _PY6S_IMPORT_ERROR = None


S2_BANDS_NM: Dict[str, float] = {
    "B1": 445.0,
    "B2": 520.0,
    "B3": 560.0,
    "B4": 654.0,
    "B5": 701.0,
    "B6": 743.0,
    "B7": 779.0,
    "B8": 789.0,
    "B8A": 871.0,
    "B9": 942.0,
    "B10": 1372.0,
    "B11": 1639.0,
    "B12": 2256.0,
}

DEFAULT_BANDS = ",".join(S2_BANDS_NM.keys())

NORMALIZED_AEROSOL_TYPES = (
    "continental",
    "maritime",
    "urban",
    "desert",
)

NORMALIZED_ATM_PROFILES = (
    "tropical",
    "midlatitude_summer",
    "midlatitude_winter",
    "subarctic_summer",
    "subarctic_winter",
    "us_standard",
)

LIBRADTRAN_ATMOSPHERE_MAP = {
    "tropical": "afglt",
    "midlatitude_summer": "afglms",
    "midlatitude_winter": "afglmw",
    "subarctic_summer": "afglss",
    "subarctic_winter": "afglsw",
    "us_standard": "afglus",
}

LIBRADTRAN_AEROSOL_LINE_MAP = {
    "continental": "aerosol_default",
    "maritime": "aerosol_default maritime",
    "urban": "aerosol_default urban",
    "desert": "aerosol_default desert",
}

USE_PREDEFINED_ATM_PROFILE_DEFAULT = True


def _ensure_py6s_available() -> None:
    if SixS is None:
        raise RuntimeError(
            "Py6S is not available. Install with:\n"
            "  python -m pip install Py6S\n"
            "and ensure the 6S executable is available for Py6S."
        ) from _PY6S_IMPORT_ERROR


def _parse_csv_list(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _normalize_label(value: object, field_name: str) -> str:
    text = str(value).strip().lower()
    if not text:
        raise ValueError(f"Missing required normalized label for `{field_name}`.")
    return text


def _normalize_and_validate_state_labels(state: Dict) -> Dict:
    aerosol = _normalize_label(state.get("aerosol_type"), "aerosol_type")
    profile = _normalize_label(state.get("atm_profile"), "atm_profile")
    if aerosol not in NORMALIZED_AEROSOL_TYPES:
        raise ValueError(
            f"Unsupported aerosol_type `{aerosol}`. Supported: {list(NORMALIZED_AEROSOL_TYPES)}"
        )
    if profile not in NORMALIZED_ATM_PROFILES:
        raise ValueError(
            f"Unsupported atm_profile `{profile}`. Supported: {list(NORMALIZED_ATM_PROFILES)}"
        )
    state["aerosol_type"] = aerosol
    state["atm_profile"] = profile
    return state


def _load_manifest(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            try:
                rows.append(_normalize_and_validate_state_labels(row))
            except ValueError as exc:
                sid = row.get("state_id", "<missing>")
                raise ValueError(
                    f"Invalid normalized labels in {path} line {line_no} (state_id={sid}): {exc}"
                ) from exc
    if not rows:
        raise ValueError(f"No states found in manifest: {path}")
    return rows


def map_atm_profile_6s(profile: str):
    normalized = _normalize_label(profile, "atm_profile")
    mapping = {
        "tropical": AtmosProfile.Tropical,
        "midlatitude_summer": AtmosProfile.MidlatitudeSummer,
        "midlatitude_winter": AtmosProfile.MidlatitudeWinter,
        "subarctic_summer": AtmosProfile.SubarcticSummer,
        "subarctic_winter": AtmosProfile.SubarcticWinter,
        "us_standard": AtmosProfile.USStandard1962,
    }
    if normalized not in mapping:
        raise ValueError(
            f"Unsupported atm_profile `{normalized}` for 6S. Supported: {list(mapping.keys())}"
        )
    return mapping[normalized]


def map_aerosol_6s(aerosol: str):
    normalized = _normalize_label(aerosol, "aerosol_type")
    mapping = {
        "continental": AeroProfile.Continental,
        "maritime": AeroProfile.Maritime,
        "urban": AeroProfile.Urban,
        "desert": AeroProfile.Desert,
    }
    if normalized not in mapping:
        raise ValueError(
            f"Unsupported aerosol_type `{normalized}` for 6S. Supported: {list(mapping.keys())}"
        )
    return mapping[normalized]


def _solve_params_reflectance(
    r0: float,
    r1: float,
    r2: float,
    rho1: float,
    rho2: float,
) -> Tuple[float, float, float]:
    y1 = r1 - r0
    y2 = r2 - r0
    s = (y1 / rho1 - y2 / rho2) / (y1 - y2 + 1e-30)
    t_total = y1 * (1 - s * rho1) / (rho1 + 1e-30)
    rho_path = r0
    return float(rho_path), float(t_total), float(s)


def _run_6s_reflectance_and_eglo_at_wavelength(
    state: Dict,
    wavelength_nm: float,
    albedo: float,
    use_predefined_atm_profile: bool,
) -> Tuple[float, float]:
    s = SixS()
    s.geometry = Geometry.User()
    s.geometry.solar_z = float(state["sza_deg"])
    s.geometry.view_z = float(state["vza_deg"])
    s.geometry.solar_a = 0.0
    s.geometry.view_a = float(state["raa_deg"])
    s.altitudes.set_sensor_satellite_level()
    s.altitudes.set_target_custom_altitude(float(state["elev_km"]))
    s.aot550 = float(state["aod550"])
    if use_predefined_atm_profile:
        # Py6S does not provide a clean combined mode with both predefined atmospheric profile
        # and explicit user CWV/O3 overrides in one setting, so this mode preserves profile choice.
        s.atmos_profile = AtmosProfile.PredefinedType(map_atm_profile_6s(state["atm_profile"]))
    else:
        # Explicit constituent mode: directly set water vapor and ozone.
        s.atmos_profile = AtmosProfile.UserWaterAndOzone(float(state["cwv_cm"]), float(state["o3_cm"]))
    s.aero_profile = AeroProfile.PredefinedType(map_aerosol_6s(state["aerosol_type"]))
    s.ground_reflectance = GroundReflectance.HomogeneousLambertian(float(albedo))
    s.wavelength = Wavelength(float(wavelength_nm) / 1000.0)
    s.run()
    rho = float(s.outputs.apparent_reflectance)
    direct = getattr(s.outputs, "direct_solar_irradiance", None)
    diffuse = getattr(s.outputs, "diffuse_solar_irradiance", None)
    if direct is None or diffuse is None:
        # Keep output schema consistent with libRadtran; use NaN if irradiance terms are unavailable.
        eglo = float("nan")
    else:
        eglo = float(direct) + float(diffuse)
    return rho, eglo


def _run_6s_reflectance_and_eglo_band_averaged(
    state: Dict,
    band: str,
    albedo: float,
    use_predefined_atm_profile: bool,
) -> Tuple[float, float]:
    wavelengths_nm, srf_response = load_srf(band)
    samples_nm = [lam for lam, rsp in zip(wavelengths_nm, srf_response) if rsp > 0.0]
    if not samples_nm:
        raise ValueError(f"No positive SRF support for band {band}.")

    rho_samples: List[float] = []
    eglo_samples: List[float] = []
    for wavelength_nm in samples_nm:
        rho, eglo = _run_6s_reflectance_and_eglo_at_wavelength(
            state=state,
            wavelength_nm=wavelength_nm,
            albedo=albedo,
            use_predefined_atm_profile=use_predefined_atm_profile,
        )
        rho_samples.append(rho)
        eglo_samples.append(eglo)

    rho_band = convolve_to_band(samples_nm, rho_samples, band)
    eglo_band = convolve_to_band(samples_nm, eglo_samples, band)
    return float(rho_band), float(eglo_band)




def _qa_flags_from_row(row: Dict) -> Tuple[bool, bool, bool, bool, bool]:
    monotonicity_failed = not (row["r0"] < row["r1"] < row["r2"])
    t_clamped = (row["T_total_raw"] < 0.0) or (row["T_total_raw"] > 1.0)
    rp_clamped = (row["rho_path_raw"] < 0.0) or (row["rho_path_raw"] > 1.0)
    s_clamped = (row["spher_alb_raw"] < 0.0) or (row["spher_alb_raw"] > 1.0)
    finite = all(
        math.isfinite(float(row[k]))
        for k in ("r0", "r1", "r2", "rho_path", "T_total", "spher_alb")
    )
    qa_valid = finite and not monotonicity_failed and not (t_clamped or rp_clamped or s_clamped)
    return qa_valid, monotonicity_failed, t_clamped, rp_clamped, s_clamped


def _failure_row(state: Dict, band: str, error_msg: str, rho1: float, rho2: float) -> Dict:
    return {
        "band": band,
        "wvl_nm": S2_BANDS_NM[band],
        "sza_deg": float(state["sza_deg"]),
        "vza_deg": float(state["vza_deg"]),
        "raa_deg": float(state["raa_deg"]),
        "aod550": float(state["aod550"]),
        "cwv_cm": float(state["cwv_cm"]),
        "o3_cm": float(state["o3_cm"]),
        "elev_km": float(state["elev_km"]),
        "aerosol_type": str(state["aerosol_type"]),
        "atm_profile": str(state["atm_profile"]),
        "solver": str(state.get("solver", "6s")),
        "nstr": int(state.get("nstr", 0)),
        "rho1": float(rho1),
        "rho2": float(rho2),
        "Eg0": None,
        "Eg1": None,
        "Eg2": None,
        "r0": None,
        "r1": None,
        "r2": None,
        "rho_path": None,
        "T_total": None,
        "spher_alb": None,
        "T_total_raw": None,
        "rho_path_raw": None,
        "spher_alb_raw": None,
        "state_id": state.get("state_id", ""),
        "split": state.get("split", "train"),
        "qa_valid": False,
        "monotonicity_failed": True,
        "T_total_was_clamped": False,
        "rho_path_was_clamped": False,
        "spher_alb_was_clamped": False,
        "__error__": error_msg,
    }


def _compute_row(task: Tuple[Dict, str, float, float, bool]) -> Dict:
    _ensure_py6s_available()
    state, band, rho1, rho2, use_predefined_atm_profile = task
    try:
        center_nm = float(S2_BANDS_NM[band])
        r0, eg0 = _run_6s_reflectance_and_eglo_band_averaged(
            state=state,
            band=band,
            albedo=0.0,
            use_predefined_atm_profile=use_predefined_atm_profile,
        )
        r1, eg1 = _run_6s_reflectance_and_eglo_band_averaged(
            state=state,
            band=band,
            albedo=rho1,
            use_predefined_atm_profile=use_predefined_atm_profile,
        )
        r2, eg2 = _run_6s_reflectance_and_eglo_band_averaged(
            state=state,
            band=band,
            albedo=rho2,
            use_predefined_atm_profile=use_predefined_atm_profile,
        )

        rho_path, t_total, spher_alb = _solve_params_reflectance(r0=r0, r1=r1, r2=r2, rho1=rho1, rho2=rho2)
        row = {
            "band": band,
            "wvl_nm": S2_BANDS_NM[band],
            "sza_deg": float(state["sza_deg"]),
            "vza_deg": float(state["vza_deg"]),
            "raa_deg": float(state["raa_deg"]),
            "aod550": float(state["aod550"]),
            "cwv_cm": float(state["cwv_cm"]),
            "o3_cm": float(state["o3_cm"]),
            "elev_km": float(state["elev_km"]),
            "aerosol_type": str(state["aerosol_type"]),
            "atm_profile": str(state["atm_profile"]),
            "solver": str(state.get("solver", "6s")),
            "nstr": int(state.get("nstr", 0)),
            "rho1": float(rho1),
            "rho2": float(rho2),
            "Eg0": float(eg0),
            "Eg1": float(eg1),
            "Eg2": float(eg2),
            "r0": float(r0),
            "r1": float(r1),
            "r2": float(r2),
            "rho_path": float(rho_path),
            "T_total": float(t_total),
            "spher_alb": float(spher_alb),
            "T_total_raw": float(t_total),
            "rho_path_raw": float(rho_path),
            "spher_alb_raw": float(spher_alb),
            "state_id": state.get("state_id", ""),
            "split": state.get("split", "train"),
        }
        qa_valid, mono_fail, t_flag, rp_flag, s_flag = _qa_flags_from_row(row)
        row["qa_valid"] = qa_valid
        row["monotonicity_failed"] = mono_fail
        row["T_total_was_clamped"] = t_flag
        row["rho_path_was_clamped"] = rp_flag
        row["spher_alb_was_clamped"] = s_flag
        return row
    except Exception as exc:  # pragma: no cover - runtime external engine
        return _failure_row(state=state, band=band, error_msg=str(exc), rho1=rho1, rho2=rho2)


def _iter_tasks(
    states: Iterable[Dict],
    bands: List[str],
    rho1: float,
    rho2: float,
    use_predefined_atm_profile: bool,
):
    for st in states:
        for b in bands:
            yield (st, b, rho1, rho2, use_predefined_atm_profile)


def _print_dry_run_mapping(state: Dict, use_predefined_atm_profile: bool) -> None:
    profile = state["atm_profile"]
    aerosol = state["aerosol_type"]
    mode = "predefined_atmosphere_profile" if use_predefined_atm_profile else "user_water_ozone"
    print("Dry-run mapping check (6S + libRadtran):")
    print(
        json.dumps(
            {
                "normalized_state": {
                    "state_id": state.get("state_id", ""),
                    "aerosol_type": aerosol,
                    "atm_profile": profile,
                    "cwv_cm": state.get("cwv_cm"),
                    "o3_cm": state.get("o3_cm"),
                },
                "mapping_mode": mode,
                "mapping_6s": {
                    "atm_profile_input": profile,
                    "atm_profile_mapped": str(map_atm_profile_6s(profile)),
                    "aerosol_input": aerosol,
                    "aerosol_mapped": str(map_aerosol_6s(aerosol)),
                },
                "mapping_libradtran": {
                    "atm_profile_input": profile,
                    "atmosphere_file": LIBRADTRAN_ATMOSPHERE_MAP[profile],
                    "aerosol_input": aerosol,
                    "aerosol_line": LIBRADTRAN_AEROSOL_LINE_MAP[aerosol],
                },
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run 6S rows from a shared normalized state manifest.\n\n"
            "Intended workflow:\n"
            "  1) python data_generator/generate_shared_states.py ...\n"
            "  2) python data_generator/generate_libradtran_dataset.py --states_manifest <manifest> ...\n"
            "  3) python data_generator/generate_6s_dataset.py --states_manifest <manifest> ..."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--states_manifest", required=True, help="Path to existing state_manifest JSONL.")
    p.add_argument("--output_dir", default="data/generated_6s_from_manifest", help="Separate output directory for 6S rows.")
    p.add_argument("--bands", default=DEFAULT_BANDS, help="Comma-separated Sentinel-2 bands (default: all 13).")
    p.add_argument("--rho1", type=float, default=0.2)
    p.add_argument("--rho2", type=float, default=0.6)
    p.add_argument("--max_workers", type=int, default=4)
    p.add_argument("--max_in_flight", type=int, default=32, help="Maximum queued futures.")
    p.add_argument("--progress_every", type=int, default=10, help="Progress print interval in completed rows.")
    p.add_argument("--max_states", type=int, default=None, help="Optional cap for a small smoke run (uses first N states).")
    p.add_argument("--copy_manifest", action="store_true", help="Copy input manifest into output_dir/state_manifest.jsonl.")
    p.add_argument(
        "--use_predefined_atm_profile",
        dest="use_predefined_atm_profile",
        action="store_true",
        help="Use mapped predefined atmospheric profile labels in 6S (uses atm_profile; ignores explicit CWV/O3 override).",
    )
    p.add_argument(
        "--no_use_predefined_atm_profile",
        dest="use_predefined_atm_profile",
        action="store_false",
        help="Use AtmosProfile.UserWaterAndOzone(CWV,O3) instead of predefined profile mapping.",
    )
    p.add_argument(
        "--dry_run_mapping",
        action="store_true",
        help="Print normalized state and mapped 6S/libRadtran atmosphere+aerosol values, then exit.",
    )
    p.set_defaults(use_predefined_atm_profile=USE_PREDEFINED_ATM_PROFILE_DEFAULT)
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    _ensure_py6s_available()

    manifest_path = Path(args.states_manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bands = _parse_csv_list(args.bands)
    invalid = [b for b in bands if b not in S2_BANDS_NM]
    if invalid:
        raise ValueError(f"Unknown bands: {invalid}. Supported bands: {list(S2_BANDS_NM.keys())}")

    states = _load_manifest(manifest_path)
    original_state_count = len(states)
    if args.max_states is not None:
        if args.max_states <= 0:
            raise ValueError("--max_states must be a positive integer when provided.")
        states = states[: args.max_states]
        print(f"Using first {len(states)} states from manifest (original: {original_state_count}).")

    if args.dry_run_mapping:
        _print_dry_run_mapping(states[0], use_predefined_atm_profile=bool(args.use_predefined_atm_profile))
        return

    # Fail fast before launching large parallel runs when 6S binary is missing.
    probe = _compute_row((states[0], bands[0], args.rho1, args.rho2, bool(args.use_predefined_atm_profile)))
    if probe.get("__error__") and "6S executable not found" in str(probe.get("__error__")):
        raise RuntimeError(
            "6S executable not found by Py6S. Install/provide a 6S binary (for example `sixs` or `sixsV1.1` on PATH), "
            "then rerun this script."
        )

    rows_path = output_dir / "dataset_rows.jsonl"
    summary_path = output_dir / "summary.json"
    errors_path = output_dir / "errors.jsonl"
    total_tasks = len(states) * len(bands)

    if args.copy_manifest:
        shutil.copyfile(manifest_path, output_dir / "state_manifest.jsonl")

    start = datetime.datetime.utcnow()
    completed = 0
    error_count = 0
    rows_per_band = {b: 0 for b in bands}

    pending = set()
    task_iter = iter(
        _iter_tasks(
            states=states,
            bands=bands,
            rho1=args.rho1,
            rho2=args.rho2,
            use_predefined_atm_profile=bool(args.use_predefined_atm_profile),
        )
    )
    max_workers = max(1, int(args.max_workers))
    max_in_flight = max(1, int(args.max_in_flight))

    with rows_path.open("w", encoding="utf-8") as rf, errors_path.open("w", encoding="utf-8") as ef, ProcessPoolExecutor(
        max_workers=max_workers
    ) as ex:
        while len(pending) < max_in_flight:
            try:
                pending.add(ex.submit(_compute_row, next(task_iter)))
            except StopIteration:
                break

        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                row = fut.result()
                row_error = row.pop("__error__", None)
                rf.write(json.dumps(row) + "\n")
                completed += 1
                band = row.get("band")
                if band in rows_per_band:
                    rows_per_band[band] += 1
                if not row.get("qa_valid", False):
                    error_count += 1
                    ef.write(json.dumps({"state_id": row.get("state_id"), "band": row.get("band"), "error": row_error}) + "\n")

                if (completed % args.progress_every == 0) or (completed == total_tasks):
                    rf.flush()
                    ef.flush()
                    elapsed = (datetime.datetime.utcnow() - start).total_seconds()
                    pct = (100.0 * completed / total_tasks) if total_tasks else 100.0
                    print(f"Progress: {completed}/{total_tasks} rows ({pct:.2f}%), elapsed={elapsed:.1f}s")

            while len(pending) < max_in_flight:
                try:
                    pending.add(ex.submit(_compute_row, next(task_iter)))
                except StopIteration:
                    break

    elapsed_total = (datetime.datetime.utcnow() - start).total_seconds()
    summary = {
        "source_model": "6s",
        "states_manifest": str(manifest_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "original_unique_states_in_manifest": original_state_count,
        "unique_states": len(states),
        "bands": bands,
        "total_rows": completed,
        "rows_per_band": rows_per_band,
        "qa_valid_false_count": error_count,
        "rho1": float(args.rho1),
        "rho2": float(args.rho2),
        "max_workers": max_workers,
        "use_predefined_atm_profile": bool(args.use_predefined_atm_profile),
        "elapsed_sec": elapsed_total,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {rows_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {errors_path}")


if __name__ == "__main__":
    main()
