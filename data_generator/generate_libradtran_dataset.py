import os, math, json, sys, tempfile, subprocess, hashlib, datetime, argparse, random
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Set
try:
    from .s2_srf import convolve_to_band, load_srf
except ImportError:
    from s2_srf import convolve_to_band, load_srf

_PKG_DIR = Path(__file__).resolve().parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

PROJECT_ROOT = _PKG_DIR.parent

# Repo layout (under PROJECT_ROOT):
#   data/original/              vendor inputs (e.g. ESA SRF .xlsx) — optional; see tools/extract_s2_srf.py
#   data/sentinel_srf/s2_srf/   SRF CSVs per mission: S2A/, S2B/, S2C/
#   data/generated/lut/         JSONL shards from this script
#   data/generated/debug_cases/ optional RT debug dumps

# Per-band diagnostics for clamping / soft issues
CLAMP_STATS: Dict[str, Dict[str, float]] = {}

# Nominal Sentinel-2 band centers (nm) used for single-wavelength calculations.
# B1 center is set to 445 nm.
S2_BANDS_NM = {
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
)

LIBRADTRAN_ATMOSPHERE_MAP = {
    "tropical": "tropics",
    "midlatitude_summer": "midlatitude_summer",
    "midlatitude_winter": "midlatitude_winter",
    "subarctic_summer": "subarctic_summer",
    "subarctic_winter": "subarctic_winter",
}

LIBRADTRAN_AEROSOL_LINE_MAP = {
    "continental": "aerosol_default",
    "maritime": "aerosol_default",
    "urban": "aerosol_default",
    "desert": "aerosol_default",
}

SIXS_ATM_PROFILE_MAP = {
    "tropical": "AtmosProfile.Tropical",
    "midlatitude_summer": "AtmosProfile.MidlatitudeSummer",
    "midlatitude_winter": "AtmosProfile.MidlatitudeWinter",
    "subarctic_summer": "AtmosProfile.SubarcticSummer",
    "subarctic_winter": "AtmosProfile.SubarcticWinter",
}

SIXS_AEROSOL_MAP = {
    "continental": "AeroProfile.Continental",
    "maritime": "AeroProfile.Maritime",
    "urban": "AeroProfile.Urban",
    "desert": "AeroProfile.Desert",
}


def normalize_label(value: object, field_name: str) -> str:
    text = str(value).strip().lower()
    if not text:
        raise ValueError(f"Missing required normalized label for `{field_name}`.")
    return text


def map_atm_profile_libradtran(profile: str) -> str:
    normalized = normalize_label(profile, "atm_profile")
    if normalized not in LIBRADTRAN_ATMOSPHERE_MAP:
        raise ValueError(
            f"Unsupported atm_profile `{normalized}`. Supported: {list(NORMALIZED_ATM_PROFILES)}"
        )
    return LIBRADTRAN_ATMOSPHERE_MAP[normalized]


def map_aerosol_libradtran(aerosol: str) -> str:
    normalized = normalize_label(aerosol, "aerosol_type")
    if normalized not in LIBRADTRAN_AEROSOL_LINE_MAP:
        raise ValueError(
            f"Unsupported aerosol_type `{normalized}`. Supported: {list(NORMALIZED_AEROSOL_TYPES)}"
        )
    return LIBRADTRAN_AEROSOL_LINE_MAP[normalized]


def normalize_and_validate_state_labels(state: "RtState") -> "RtState":
    aerosol = normalize_label(state.aerosol_type, "aerosol_type")
    profile = normalize_label(state.atm_profile, "atm_profile")
    if aerosol not in NORMALIZED_AEROSOL_TYPES:
        raise ValueError(
            f"Unsupported aerosol_type `{aerosol}`. Supported: {list(NORMALIZED_AEROSOL_TYPES)}"
        )
    if profile not in NORMALIZED_ATM_PROFILES:
        raise ValueError(
            f"Unsupported atm_profile `{profile}`. Supported: {list(NORMALIZED_ATM_PROFILES)}"
        )
    return RtState(
        sza_deg=state.sza_deg,
        vza_deg=state.vza_deg,
        raa_deg=state.raa_deg,
        aod550=state.aod550,
        cwv_cm=state.cwv_cm,
        o3_cm=state.o3_cm,
        elev_km=state.elev_km,
        aerosol_type=aerosol,
        atm_profile=profile,
        solver=state.solver,
        nstr=state.nstr,
    )


@dataclass(frozen=True)
class RtState:
    sza_deg: float
    vza_deg: float
    raa_deg: float
    aod550: float
    cwv_cm: float  # column water vapour in cm
    o3_cm: float
    elev_km: float
    aerosol_type: str = "continental"
    atm_profile: str = "midlatitude_summer"
    solver: str = "disort"
    nstr: int = 8

def run_uvspec(input_text: str, uvspec_bin: str = "uvspec") -> str:
    """Run uvspec; raise on non-zero exit. Returns stdout (with stderr appended as comments)."""
    stdout_txt, stderr_txt, returncode = run_uvspec_raw(input_text, uvspec_bin)
    if returncode != 0:
        raise RuntimeError(
            f"uvspec failed with exit code {returncode}.\n"
            f"STDERR output:\n{stderr_txt}"
        )
    if stderr_txt.strip():
        commented_err = "\n".join("# " + line for line in stderr_txt.splitlines())
        stdout_txt = stdout_txt + ("\n" if stdout_txt else "") + commented_err + "\n"
    return stdout_txt


def run_uvspec_raw(input_text: str, uvspec_bin: str = "uvspec") -> Tuple[str, str, int]:
    """Run uvspec and return (stdout, stderr, returncode) for debugging."""
    with tempfile.TemporaryDirectory() as td:
        inp = os.path.join(td, "in.inp")
        with open(inp, "w") as f:
            f.write(input_text)
        with open(inp, "rb") as fin:
            p = subprocess.run(
                [uvspec_bin],
                stdin=fin,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        stdout_txt = p.stdout.decode("utf-8", errors="replace")
        stderr_txt = p.stderr.decode("utf-8", errors="replace")
        return stdout_txt, stderr_txt, p.returncode

def parse_two_col_table(txt: str) -> List[Tuple[float, float]]:
    out = []
    for line in txt.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            x = float(parts[0])
            y = float(parts[1])
            out.append((x, y))
        except:
            continue
    return out

def parse_three_col_table(txt: str) -> List[Tuple[float, float, float]]:
    out = []
    for line in txt.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            x = float(parts[0])
            y = float(parts[1])
            z = float(parts[2])
            out.append((x, y, z))
        except:
            continue
    return out

def mk_common_block(st: RtState, wvl_nm) -> str:
    mu = math.cos(math.radians(st.vza_deg))
    phi = st.raa_deg

    aerosol_line = map_aerosol_libradtran(st.aerosol_type)
    atmosphere_file = map_atm_profile_libradtran(st.atm_profile)

    # Convert CWV from cm to mm of precipitable water for MM units
    h2o_mm = st.cwv_cm * 10.0

    if isinstance(wvl_nm, (tuple, list)) and len(wvl_nm) == 2:
        wvl_line = f"wavelength {wvl_nm[0]:.3f} {wvl_nm[1]:.3f}"
    else:
        wvl_line = f"wavelength {float(wvl_nm):.3f}"

    return "\n".join([
        f"atmosphere_file {atmosphere_file}",
        # Use Kurucz spectrum with sufficient wavelength coverage
        "source solar kurudz_1.0nm.dat",
        wvl_line,
        f"sza {st.sza_deg:.6f}",
        "phi0 0.0",
        f"umu {mu:.8f}",
        f"phi {phi:.6f}",
        f"zout 100.0",
        f"rte_solver {st.solver}",
        f"number_of_streams {st.nstr}",
        # Use REPTRAN with coarse grid (data now found via LIBRADTRAN_DATA_FILES)
        "mol_abs_param reptran coarse",
        aerosol_line,
        f"aerosol_set_tau_at_wvl 550 {st.aod550:.8f}",
        f"mol_modify H2O {h2o_mm:.8f} MM",
        # O3 column: convert from cm-atm to Dobson Units (1 DU = 1e-3 cm-atm)
        f"mol_modify O3 {st.o3_cm * 1000.0:.8f} DU",
        f"altitude {st.elev_km:.8f}",
        # Need both uu (radiance) and edir (direct irradiance) at TOA
        "output_user lambda uu edir",
    ]) + "\n"

def mk_surface_flux_block(st: RtState, wvl_nm, albedo: float) -> str:
    aerosol_line = map_aerosol_libradtran(st.aerosol_type)
    atmosphere_file = map_atm_profile_libradtran(st.atm_profile)

    h2o_mm = st.cwv_cm * 10.0

    if isinstance(wvl_nm, (tuple, list)) and len(wvl_nm) == 2:
        wvl_line = f"wavelength {wvl_nm[0]:.3f} {wvl_nm[1]:.3f}"
    else:
        wvl_line = f"wavelength {float(wvl_nm):.3f}"

    return "\n".join([
        f"atmosphere_file {atmosphere_file}",
        "source solar kurudz_1.0nm.dat",
        wvl_line,
        f"sza {st.sza_deg:.6f}",
        "phi0 0.0",
        f"zout 0.0",
        f"rte_solver {st.solver}",
        f"number_of_streams {st.nstr}",
        "mol_abs_param reptran coarse",
        aerosol_line,
        f"aerosol_set_tau_at_wvl 550 {st.aod550:.8f}",
        f"mol_modify H2O {h2o_mm:.8f} MM",
        f"mol_modify O3 {st.o3_cm * 1000.0:.8f} DU",
        f"altitude {st.elev_km:.8f}",
        f"albedo {albedo:.8f}",
        "output_user lambda eglo",
    ]) + "\n"

def get_uu_toa(st: RtState, wvl_nm: float, albedo: float) -> float:
    inp = mk_common_block(st, wvl_nm) + f"albedo {albedo:.8f}\n"
    txt = run_uvspec(inp)
    rows = parse_two_col_table(txt)
    if not rows:
        raise RuntimeError("No uu output parsed")
    return rows[-1][1]

def get_eglo_surf(
    st: RtState,
    wvl_nm: float,
    albedo: float,
    debug_log: Optional[List[Dict]] = None,
) -> float:
    inp = mk_surface_flux_block(st, wvl_nm, albedo)
    if debug_log is not None:
        stdout_txt, stderr_txt, returncode = run_uvspec_raw(inp)
        debug_log.append({
            "label": f"eglo_albedo_{albedo}",
            "inp": inp,
            "stdout": stdout_txt,
            "stderr": stderr_txt,
            "returncode": returncode,
        })
        if returncode != 0:
            raise RuntimeError(f"uvspec failed (eglo albedo={albedo}): {stderr_txt[:500]}")
        txt = stdout_txt
    else:
        txt = run_uvspec(inp)
    rows = parse_two_col_table(txt)
    if not rows:
        preview = "\n".join(txt.splitlines()[:40])
        raise RuntimeError(f"No eglo output parsed. uvspec stdout (first lines):\n{preview}")
    return rows[-1][1]


def get_eglo_surf_spectrum(
    st: RtState,
    wvl_range,
    albedo: float,
    debug_log: Optional[List[Dict]] = None,
) -> Tuple[List[float], List[float]]:
    inp = mk_surface_flux_block(st, wvl_range, albedo)
    if debug_log is not None:
        stdout_txt, stderr_txt, returncode = run_uvspec_raw(inp)
        debug_log.append({
            "label": f"eglo_spectrum_albedo_{albedo}",
            "inp": inp,
            "stdout": stdout_txt,
            "stderr": stderr_txt,
            "returncode": returncode,
        })
        if returncode != 0:
            raise RuntimeError(f"uvspec failed (eglo spectrum albedo={albedo}): {stderr_txt[:500]}")
        txt = stdout_txt
    else:
        txt = run_uvspec(inp)
    rows = parse_two_col_table(txt)
    if not rows:
        preview = "\n".join(txt.splitlines()[:40])
        raise RuntimeError(f"No eglo spectrum output parsed. uvspec stdout (first lines):\n{preview}")
    return [x for x, _ in rows], [y for _, y in rows]


def get_rho_toa(st: RtState, wvl_nm: float, albedo: float) -> float:
    """Backward-compatible single-wavelength TOA reflectance (not SRF-averaged)."""
    wvls, rhos = get_rho_toa_spectrum(st, (wvl_nm, wvl_nm), albedo)
    if not rhos:
        raise RuntimeError("No uu/edir output parsed")
    return rhos[-1]


def get_rho_toa_spectrum(
    st: RtState,
    wvl_range,
    albedo: float,
    debug_log: Optional[List[Dict]] = None,
):
    """
    Compute TOA reflectance spectrum over a wavelength range.

    Returns (wavelengths_nm, rho_toa(lambda)).
    """
    inp = mk_common_block(st, wvl_range) + f"albedo {albedo:.8f}\n"
    if debug_log is not None:
        stdout_txt, stderr_txt, returncode = run_uvspec_raw(inp)
        debug_log.append({
            "label": f"rho_toa_albedo_{albedo}",
            "inp": inp,
            "stdout": stdout_txt,
            "stderr": stderr_txt,
            "returncode": returncode,
        })
        if returncode != 0:
            raise RuntimeError(f"uvspec failed (rho_toa albedo={albedo}): {stderr_txt[:500]}")
        txt = stdout_txt
    else:
        txt = run_uvspec(inp)
    rows = parse_three_col_table(txt)
    if not rows:
        raise RuntimeError("No uu/edir output parsed")
    wvls: List[float] = []
    rhos: List[float] = []
    for lam, L_toa, edir in rows:
        # libRadtran manual (Table 3.1): edir is direct beam irradiance w.r.t. horizontal plane.
        # Do not multiply by cos(SZA) again; that would double-apply projection.
        rho = math.pi * L_toa / (edir + 1e-30)
        wvls.append(lam)
        rhos.append(rho)
    return wvls, rhos

def solve_params_reflectance(
    r0: float, r1: float, r2: float,
    rho1: float, rho2: float
) -> Tuple[float, float, float]:
    """Solve for path reflectance, total transmittance, and spherical albedo in reflectance space."""
    y1 = r1 - r0
    y2 = r2 - r0
    s = (y1 / rho1 - y2 / rho2) / (y1 - y2 + 1e-30)
    T = y1 * (1 - s * rho1) / (rho1 + 1e-30)
    rho_path = r0
    return rho_path, T, s

DEBUG_CASES_DIR = PROJECT_ROOT / "data" / "generated" / "debug_cases"


def _write_debug_case(
    st: RtState,
    band: str,
    debug_log: List[Dict],
    r0: float,
    r1: float,
    r2: float,
    row: Optional[Dict] = None,
    reason: str = "validation",
) -> Path:
    """Write debug_cases/<timestamp>_<reason>/ and return the path."""
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    debug_dir = DEBUG_CASES_DIR / f"{timestamp}_{reason}"
    debug_dir.mkdir(parents=True, exist_ok=True)

    with (debug_dir / "rt_state.json").open("w") as f:
        json.dump(asdict(st), f, indent=2)

    (debug_dir / "band.txt").write_text(band, encoding="utf-8")

    intermediates = {"r0": r0, "r1": r1, "r2": r2}
    with (debug_dir / "intermediates_r0_r1_r2.json").open("w") as f:
        json.dump(intermediates, f, indent=2)

    if row is not None:
        with (debug_dir / "row_before_validate.json").open("w") as f:
            json.dump(row, f, indent=2)

    runs_dir = debug_dir / "uvspec_runs"
    runs_dir.mkdir(exist_ok=True)
    for i, rec in enumerate(debug_log):
        prefix = runs_dir / f"run_{i:02d}_{rec['label'].replace('.', '_')}"
        (prefix.with_suffix(".inp")).write_text(rec["inp"], encoding="utf-8")
        (prefix.with_name(prefix.name + "_stdout.txt")).write_text(
            rec["stdout"], encoding="utf-8"
        )
        (prefix.with_name(prefix.name + "_stderr.txt")).write_text(
            rec["stderr"], encoding="utf-8"
        )
        (prefix.with_name(prefix.name + "_returncode.txt")).write_text(
            str(rec["returncode"]), encoding="utf-8"
        )

    return debug_dir


def generate_lut_row(
    st: RtState,
    band: str,
    rho1: float = 0.2,
    rho2: float = 0.6,
    capture_debug: bool = True,
) -> Dict:
    wvl = S2_BANDS_NM[band]
    debug_log: List[Dict] = [] if capture_debug else None

    srf_wavelengths, srf_response = load_srf(band)
    srf_support = [lam for lam, rsp in zip(srf_wavelengths, srf_response) if rsp > 0.0]
    if not srf_support:
        raise ValueError(f"No positive SRF support for band {band}")
    wvl_range = (min(srf_support), max(srf_support))

    eglo_w0, eglo_spec0 = get_eglo_surf_spectrum(st, wvl_range, 0.0, debug_log=debug_log)
    eglo_w1, eglo_spec1 = get_eglo_surf_spectrum(st, wvl_range, rho1, debug_log=debug_log)
    eglo_w2, eglo_spec2 = get_eglo_surf_spectrum(st, wvl_range, rho2, debug_log=debug_log)
    Eg0 = convolve_to_band(eglo_w0, eglo_spec0, band)
    Eg1 = convolve_to_band(eglo_w1, eglo_spec1, band)
    Eg2 = convolve_to_band(eglo_w2, eglo_spec2, band)

    w0, rho_spec0 = get_rho_toa_spectrum(st, wvl_range, 0.0, debug_log=debug_log)
    w1, rho_spec1 = get_rho_toa_spectrum(st, wvl_range, rho1, debug_log=debug_log)
    w2, rho_spec2 = get_rho_toa_spectrum(st, wvl_range, rho2, debug_log=debug_log)
    r0 = convolve_to_band(w0, rho_spec0, band)
    r1 = convolve_to_band(w1, rho_spec1, band)
    r2 = convolve_to_band(w2, rho_spec2, band)

    rho_path, T_total, s = solve_params_reflectance(r0, r1, r2, rho1, rho2)

    # Monotonicity check (soft): expect r0 < r1 < r2
    monotonicity_ok = (r0 < r1 < r2)
    if not monotonicity_ok and debug_log:
        _write_debug_case(st, band, debug_log, r0, r1, r2, row=None, reason="monotonicity")

    row: Dict = {
        "band": band,
        "wvl_nm": wvl,
        "sza_deg": st.sza_deg,
        "vza_deg": st.vza_deg,
        "raa_deg": st.raa_deg,
        "aod550": st.aod550,
        "cwv_cm": st.cwv_cm,
        "o3_cm": st.o3_cm,
        "elev_km": st.elev_km,
        "aerosol_type": st.aerosol_type,
        "atm_profile": st.atm_profile,
        "solver": st.solver,
        "nstr": st.nstr,
        "rho1": rho1,
        "rho2": rho2,
        "Eg0": Eg0,
        "Eg1": Eg1,
        "Eg2": Eg2,
        "r0": r0,
        "r1": r1,
        "r2": r2,
        "rho_path": rho_path,
        "T_total": T_total,
        "spher_alb": s,
        "T_total_raw": T_total,
    }
    row = postprocess_row(row)

    try:
        validate_row(row)
    except (ValueError, TypeError) as e:
        if debug_log:
            _write_debug_case(st, band, debug_log, r0, r1, r2, row=row)
        raise

    return row


def _is_finite(x: float) -> bool:
    return math.isfinite(x)


def validate_row(row: Dict) -> None:
    """Hard sanity checks; only reject non-finite values."""
    for key in ("rho_path", "T_total", "spher_alb", "Eg0", "Eg1", "Eg2"):
        if not _is_finite(row[key]):
            raise ValueError(f"Non-finite value for {key} in row: {row}")


def postprocess_row(row: Dict) -> Dict:
    """Keep raw outputs unchanged; store only diagnostics (no clamped-tag columns)."""
    band = row.get("band", "UNKNOWN")
    stats = CLAMP_STATS.setdefault(
        band,
        {
            "rho_path_clamped": 0,
            "T_total_clamped": 0,
            "spher_alb_clamped": 0,
            "T_total_over_1_count": 0,
            "T_total_over_1_max": 0.0,
        },
    )

    rho_path = row["rho_path"]
    T_total = row["T_total"]
    spher_alb = row["spher_alb"]

    # Keep raw for output (T_total_raw already set in row)
    row["rho_path_raw"] = rho_path
    row["spher_alb_raw"] = spher_alb

    if T_total > 1.0:
        stats["T_total_over_1_count"] += 1
        if T_total > stats["T_total_over_1_max"]:
            stats["T_total_over_1_max"] = T_total

    rho_out_of_range = (rho_path < 0.0) or (rho_path > 1.0)
    T_out_of_range = (T_total < 0.0) or (T_total > 1.0)
    s_out_of_range = (spher_alb < 0.0) or (spher_alb > 1.0)

    if rho_out_of_range:
        stats["rho_path_clamped"] += 1
    if T_out_of_range:
        stats["T_total_clamped"] += 1
    if s_out_of_range:
        stats["spher_alb_clamped"] += 1
    return row


# ----- Resumability: stable key for (state, band) -----

STATE_KEY_FIELDS = (
    "sza_deg", "vza_deg", "raa_deg", "aod550", "cwv_cm", "o3_cm", "elev_km",
    "aerosol_type", "atm_profile", "solver", "nstr",
)


def state_to_key(st: RtState) -> str:
    """Stable hash key for an RtState (for resumability)."""
    d = asdict(st)
    subset = {k: d[k] for k in STATE_KEY_FIELDS if k in d}
    return hashlib.sha256(json.dumps(subset, sort_keys=True).encode()).hexdigest()[:16]


def row_to_key(row: Dict) -> str:
    """Stable key for a row (state + band) from existing shard line."""
    subset = {k: row[k] for k in STATE_KEY_FIELDS if k in row}
    state_part = hashlib.sha256(json.dumps(subset, sort_keys=True).encode()).hexdigest()[:16]
    return f"{state_part}_{row['band']}"


def load_done_keys(out_dir: Path) -> Set[str]:
    """Load set of row keys already present in existing JSONL shards under out_dir."""
    done: Set[str] = set()
    if not out_dir.exists():
        return done
    for path in sorted(out_dir.glob("shard_*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    done.add(row_to_key(row))
                except Exception:
                    continue
    return done


def compute_band_coeffs(
    state_dict: Dict,
    band: str,
    rho1: float = 0.2,
    rho2: float = 0.6,
) -> Dict:
    """
    Compute one LUT row for a given state and band. Used by parallel workers.
    state_dict must be the result of asdict(RtState).
    """
    st = RtState(**{k: v for k, v in state_dict.items() if k in RtState.__dataclass_fields__})
    st = normalize_and_validate_state_labels(st)
    return generate_lut_row(st, band, rho1=rho1, rho2=rho2, capture_debug=False)


OUT_DIR = PROJECT_ROOT / "data" / "generated" / "lut"
SHARD_SIZE = 10_000
RHO1, RHO2 = 0.2, 0.6


def build_state_grid(
    sza_deg: Optional[List[float]] = None,
    vza_deg: Optional[List[float]] = None,
    raa_deg: Optional[List[float]] = None,
    aod550: Optional[List[float]] = None,
    cwv_cm: Optional[List[float]] = None,
    o3_cm: Optional[List[float]] = None,
    elev_km: Optional[List[float]] = None,
    aerosol_type: Optional[List[str]] = None,
    atm_profile: Optional[List[str]] = None,
    **defaults,
) -> List[RtState]:
    """
    Build a list of RtState by iterating over given lists; missing keys use a single default.
    Example: build_state_grid(sza_deg=[20,40], aod550=[0.1,0.3]) → 4 states.
    """
    from itertools import product

    # Single-value defaults
    opts = {
        "sza_deg": sza_deg or [defaults.get("sza_deg", 30.0)],
        "vza_deg": vza_deg or [defaults.get("vza_deg", 5.0)],
        "raa_deg": raa_deg or [defaults.get("raa_deg", 90.0)],
        "aod550": aod550 or [defaults.get("aod550", 0.2)],
        "cwv_cm": cwv_cm or [defaults.get("cwv_cm", 2.0)],
        "o3_cm": o3_cm or [defaults.get("o3_cm", 0.32)],
        "elev_km": elev_km or [defaults.get("elev_km", 0.5)],
        "aerosol_type": aerosol_type or [defaults.get("aerosol_type", "continental")],
        "atm_profile": atm_profile or [defaults.get("atm_profile", "midlatitude_summer")],
    }
    solver = defaults.get("solver", "disort")
    nstr = defaults.get("nstr", 8)

    states = []
    for v in product(
        opts["sza_deg"],
        opts["vza_deg"],
        opts["raa_deg"],
        opts["aod550"],
        opts["cwv_cm"],
        opts["o3_cm"],
        opts["elev_km"],
        opts["aerosol_type"],
        opts["atm_profile"],
    ):
        states.append(
            normalize_and_validate_state_labels(
                RtState(
                    sza_deg=v[0],
                    vza_deg=v[1],
                    raa_deg=v[2],
                    aod550=v[3],
                    cwv_cm=v[4],
                    o3_cm=v[5],
                    elev_km=v[6],
                    aerosol_type=v[7],
                    atm_profile=v[8],
                    solver=solver,
                    nstr=nstr,
                )
            )
        )
    return states


def _task_key(state_dict: Dict, band: str) -> str:
    st = RtState(**{k: v for k, v in state_dict.items() if k in RtState.__dataclass_fields__})
    return f"{state_to_key(st)}_{band}"


def main_legacy():
    # ----- How to get more data -----
    # Option A: single state (6 rows = 1 state × 6 bands)
    # states = [RtState(sza_deg=30.0, vza_deg=5.0, raa_deg=90.0, aod550=0.2, cwv_cm=2.0, o3_cm=0.32, elev_km=0.5)]

    # Small grid: 2×2×2 states × 6 bands = 48 rows (quick run)
    states = build_state_grid(
        sza_deg=[30.0, 45.0],
        vza_deg=[5.0, 15.0],
        aod550=[0.15, 0.3],
        # raa_deg, cwv_cm, o3_cm, elev_km use defaults
    )
    # Total tasks = len(states) × len(bands); resumable (skips already in data/generated/lut/shard_*.jsonl)

    bands = ["B2", "B3", "B4", "B8", "B11", "B12"]

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    done_keys = load_done_keys(out_dir)

    tasks = [(asdict(st), band) for st in states for band in bands]
    todo = [(sd, b) for (sd, b) in tasks if _task_key(sd, b) not in done_keys]

    if not todo:
        print("All tasks already present in existing shards. Nothing to do.")
        return

    # Next shard index from existing shards
    existing = list(out_dir.glob("shard_*.jsonl"))
    shard_index = max(
        (int(p.stem.split("_")[1]) for p in existing if p.stem.split("_")[1].isdigit()),
        default=0,
    )
    buffer: List[Dict] = []
    all_new_rows: List[Dict] = []
    max_workers = min(4, len(todo)) or 1

    print(f"Running {len(todo)} tasks (skipped {len(tasks) - len(todo)} already done), max_workers={max_workers}")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(compute_band_coeffs, sd, b, RHO1, RHO2): (sd, b)
            for (sd, b) in todo
        }
        for future in as_completed(future_to_task):
            sd, band = future_to_task[future]
            try:
                row = future.result()
                buffer.append(row)
                all_new_rows.append(row)
                done_keys.add(row_to_key(row))
                if len(buffer) >= SHARD_SIZE:
                    shard_index += 1
                    shard_path = out_dir / f"shard_{shard_index:05d}.jsonl"
                    with shard_path.open("w", encoding="utf-8") as f:
                        for r in buffer:
                            f.write(json.dumps(r) + "\n")
                    buffer.clear()
                    print(f"Wrote {shard_path}")
            except Exception as e:
                print(f"Failed ({band}): {e}")
                raise

    if buffer:
        shard_index += 1
        shard_path = out_dir / f"shard_{shard_index:05d}.jsonl"
        with shard_path.open("w", encoding="utf-8") as f:
            for r in buffer:
                f.write(json.dumps(r) + "\n")
        print(f"Wrote {shard_path}")

    # Monotonicity: fail run if > 1% of rows have r0 < r1 < r2 violated
    n_total = len(all_new_rows)
    mono_failures = sum(
        1
        for r in all_new_rows
        if not (r.get("r0", float("nan")) < r.get("r1", float("nan")) < r.get("r2", float("nan")))
    )
    if n_total > 0 and (mono_failures / n_total) > 0.01:
        raise RuntimeError(
            f"Monotonicity failures {mono_failures}/{n_total} ({100 * mono_failures / n_total:.2f}%) exceed 1%"
        )
    if mono_failures > 0:
        print(f"Monotonicity: {mono_failures}/{n_total} rows had r0 < r1 < r2 violated (soft warning)")

    # End-of-run out-of-range diagnostics (from rows, works with multiprocessing)
    if all_new_rows:
        band_rows: Dict[str, List[Dict]] = defaultdict(list)
        for r in all_new_rows:
            band_rows[r["band"]].append(r)

        print("Per-band out-of-range diagnostics (this run):")
        for band in sorted(band_rows.keys()):
            rows_b = band_rows[band]
            n_b = len(rows_b)

            def _stats(raw_key: str, name: str) -> None:
                raw_vals = [r[raw_key] for r in rows_b if raw_key in r]
                if not raw_vals:
                    return
                clamp_count = sum(1 for v in raw_vals if (v < 0.0) or (v > 1.0))
                clamp_rate = clamp_count / n_b if n_b else 0
                overshoot = [r[raw_key] - 1.0 for r in rows_b if raw_key in r and r[raw_key] > 1.0]
                undershoot = [r[raw_key] for r in rows_b if raw_key in r and r[raw_key] < 0.0]
                mean_raw = sum(raw_vals) / len(raw_vals)
                max_over = max(overshoot) if overshoot else None
                min_under = min(undershoot) if undershoot else None
                print(f"  {band} {name}: clamp_count={clamp_count}, clamp_rate={clamp_rate:.4f}, mean_raw={mean_raw:.6f}", end="")
                if max_over is not None:
                    print(f", max_overshoot={max_over:.6f}", end="")
                if min_under is not None:
                    print(f", min_undershoot={min_under:.6f}", end="")
                print()

            _stats("T_total_raw", "T_total")
            _stats("rho_path_raw", "rho_path")
            _stats("spher_alb_raw", "spher_alb")

    # Backward compat: single-state run → also write pretty JSON in project root
    if len(states) == 1 and len(todo) == len(bands):
        all_rows = []
        for path in sorted(out_dir.glob("shard_*.jsonl")):
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        all_rows.append(json.loads(line))
        st = states[0]
        sk = state_to_key(st)
        single_state_rows = sorted(
            [r for r in all_rows if row_to_key(r).startswith(sk + "_")],
            key=lambda r: r.get("band", ""),
        )
        if single_state_rows:
            with (OUT_DIR / "libradtran_s2_lut_rows.json").open("w", encoding="utf-8") as f:
                json.dump(single_state_rows, f, indent=2)
            print(f"Wrote {OUT_DIR / 'libradtran_s2_lut_rows.json'}")

    if CLAMP_STATS:
        print("Per-band in-process diagnostics (main process only):")
        for band in sorted(CLAMP_STATS.keys()):
            stats = CLAMP_STATS[band]
            print(
                f"  {band}: "
                f"rho_path_clamped={int(stats['rho_path_clamped'])}, "
                f"T_total_clamped={int(stats['T_total_clamped'])}, "
                f"spher_alb_clamped={int(stats['spher_alb_clamped'])}, "
                f"T_total_over_1_count={int(stats['T_total_over_1_count'])}, "
                f"T_total_over_1_max={stats['T_total_over_1_max']:.6f}"
            )


def parse_csv_list(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_int_list(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_range(value: str) -> Tuple[float, float]:
    a, b = value.split(",")
    lo = float(a.strip())
    hi = float(b.strip())
    if hi < lo:
        raise ValueError(f"Invalid range {value}: max < min")
    return lo, hi


def print_dry_run_mapping(state: RtState) -> None:
    st = normalize_and_validate_state_labels(state)
    print("Dry-run mapping check (libRadtran + 6S):")
    print(
        json.dumps(
            {
                "normalized_state": {
                    "aerosol_type": st.aerosol_type,
                    "atm_profile": st.atm_profile,
                    "cwv_cm": st.cwv_cm,
                    "o3_cm": st.o3_cm,
                },
                "mapping_libradtran": {
                    "atmosphere_file": map_atm_profile_libradtran(st.atm_profile),
                    "aerosol_line": map_aerosol_libradtran(st.aerosol_type),
                },
                "mapping_6s": {
                    "atm_profile_mapped": SIXS_ATM_PROFILE_MAP[st.atm_profile],
                    "aerosol_mapped": SIXS_AEROSOL_MAP[st.aerosol_type],
                },
            },
            indent=2,
        )
    )


def lhs_unit(n: int, dims: int, rng: random.Random) -> List[List[float]]:
    cols: List[List[float]] = []
    for _ in range(dims):
        perm = list(range(n))
        rng.shuffle(perm)
        col = [((perm[i] + rng.random()) / n) for i in range(n)]
        cols.append(col)
    return [[cols[d][i] for d in range(dims)] for i in range(n)]


def scale_unit(x: float, lo: float, hi: float) -> float:
    return lo + x * (hi - lo)


def state_id_from_state(st: RtState, include_solver: bool, include_nstr: bool) -> str:
    payload: Dict[str, object] = {
        "sza_deg": st.sza_deg,
        "vza_deg": st.vza_deg,
        "raa_deg": st.raa_deg,
        "aod550": st.aod550,
        "cwv_cm": st.cwv_cm,
        "o3_cm": st.o3_cm,
        "elev_km": st.elev_km,
        "aerosol_type": st.aerosol_type,
        "atm_profile": st.atm_profile,
    }
    if include_solver:
        payload["solver"] = st.solver
    if include_nstr:
        payload["nstr"] = st.nstr
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def assign_splits(state_ids: List[str], seed: int) -> Dict[str, str]:
    rng = random.Random(seed + 1)
    ids = state_ids[:]
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    split_map: Dict[str, str] = {}
    for sid in ids[:n_train]:
        split_map[sid] = "train"
    for sid in ids[n_train:n_train + n_val]:
        split_map[sid] = "val"
    for sid in ids[n_train + n_val:]:
        split_map[sid] = "test"
    return split_map


def sample_states_lhs(
    n_states: int,
    seed: int,
    sza_range: Tuple[float, float],
    vza_range: Tuple[float, float],
    raa_range: Tuple[float, float],
    aod_range: Tuple[float, float],
    cwv_range: Tuple[float, float],
    o3_range: Tuple[float, float],
    elev_range: Tuple[float, float],
    aerosol_types: List[str],
    atm_profiles: List[str],
    solvers: List[str],
    nstr_values: List[int],
) -> Tuple[List[RtState], List[str], bool, bool]:
    rng = random.Random(seed)
    states: List[RtState] = []
    state_ids: List[str] = []
    seen = set()
    include_solver = len(set(solvers)) > 1
    include_nstr = len(set(nstr_values)) > 1

    while len(states) < n_states:
        need = n_states - len(states)
        design = lhs_unit(need, 7, rng)
        for row in design:
            st = normalize_and_validate_state_labels(RtState(
                sza_deg=scale_unit(row[0], *sza_range),
                vza_deg=scale_unit(row[1], *vza_range),
                raa_deg=scale_unit(row[2], *raa_range),
                aod550=scale_unit(row[3], *aod_range),
                cwv_cm=scale_unit(row[4], *cwv_range),
                o3_cm=scale_unit(row[5], *o3_range),
                elev_km=scale_unit(row[6], *elev_range),
                aerosol_type=rng.choice(aerosol_types),
                atm_profile=rng.choice(atm_profiles),
                solver=rng.choice(solvers),
                nstr=rng.choice(nstr_values),
            ))
            sid = state_id_from_state(st, include_solver=include_solver, include_nstr=include_nstr)
            if sid in seen:
                continue
            seen.add(sid)
            states.append(st)
            state_ids.append(sid)
            if len(states) >= n_states:
                break

    return states, state_ids, include_solver, include_nstr


def load_state_manifest(path: Path) -> Tuple[List[RtState], List[str], Dict[str, str], bool, bool]:
    states: List[RtState] = []
    state_ids: List[str] = []
    split_map: Dict[str, str] = {}
    include_solver = False
    include_nstr = False
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = str(row["state_id"])
            split = str(row.get("split", "train"))
            try:
                st = normalize_and_validate_state_labels(RtState(
                    sza_deg=float(row["sza_deg"]),
                    vza_deg=float(row["vza_deg"]),
                    raa_deg=float(row["raa_deg"]),
                    aod550=float(row["aod550"]),
                    cwv_cm=float(row["cwv_cm"]),
                    o3_cm=float(row["o3_cm"]),
                    elev_km=float(row["elev_km"]),
                    aerosol_type=str(row.get("aerosol_type", "")),
                    atm_profile=str(row.get("atm_profile", "")),
                    solver=str(row.get("solver", "disort")),
                    nstr=int(row.get("nstr", 8)),
                ))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid normalized labels in {path} line {line_no} (state_id={sid}): {exc}"
                ) from exc
            states.append(st)
            state_ids.append(sid)
            split_map[sid] = split
            include_solver = include_solver or bool(row.get("state_id_includes_solver", False))
            include_nstr = include_nstr or bool(row.get("state_id_includes_nstr", False))
    if not states:
        raise ValueError(f"No states found in manifest: {path}")
    unique_ids = len(set(state_ids))
    if unique_ids != len(state_ids):
        raise ValueError(f"Duplicate state_id values found in manifest: {path}")
    return states, state_ids, split_map, include_solver, include_nstr


def load_excluded_state_ids(path: Path) -> Set[str]:
    excluded: Set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = row.get("state_id")
            if sid is not None:
                excluded.add(str(sid))
    return excluded


def sample_states_lhs_excluding(
    n_states: int,
    seed: int,
    excluded_ids: Set[str],
    sza_range: Tuple[float, float],
    vza_range: Tuple[float, float],
    raa_range: Tuple[float, float],
    aod_range: Tuple[float, float],
    cwv_range: Tuple[float, float],
    o3_range: Tuple[float, float],
    elev_range: Tuple[float, float],
    aerosol_types: List[str],
    atm_profiles: List[str],
    solvers: List[str],
    nstr_values: List[int],
) -> Tuple[List[RtState], List[str], bool, bool]:
    states: List[RtState] = []
    state_ids: List[str] = []
    seen: Set[str] = set()
    include_solver = len(set(solvers)) > 1
    include_nstr = len(set(nstr_values)) > 1
    attempt = 0
    while len(states) < n_states:
        need = n_states - len(states)
        batch_size = max(need * 2, 2000)
        batch_states, batch_ids, _, _ = sample_states_lhs(
            n_states=batch_size,
            seed=seed + attempt,
            sza_range=sza_range,
            vza_range=vza_range,
            raa_range=raa_range,
            aod_range=aod_range,
            cwv_range=cwv_range,
            o3_range=o3_range,
            elev_range=elev_range,
            aerosol_types=aerosol_types,
            atm_profiles=atm_profiles,
            solvers=solvers,
            nstr_values=nstr_values,
        )
        for st, sid in zip(batch_states, batch_ids):
            if sid in excluded_ids or sid in seen:
                continue
            states.append(st)
            state_ids.append(sid)
            seen.add(sid)
            if len(states) >= n_states:
                break
        attempt += 1
    return states, state_ids, include_solver, include_nstr


def qa_flags_from_row(row: Dict) -> Tuple[bool, bool, bool, bool, bool]:
    monotonicity_failed = not (row["r0"] < row["r1"] < row["r2"])
    T_total_was_clamped = (row["T_total_raw"] < 0.0) or (row["T_total_raw"] > 1.0)
    rho_path_was_clamped = (row["rho_path_raw"] < 0.0) or (row["rho_path_raw"] > 1.0)
    spher_alb_was_clamped = (row["spher_alb_raw"] < 0.0) or (row["spher_alb_raw"] > 1.0)
    finite = all(
        math.isfinite(row[k])
        for k in ("Eg0", "Eg1", "Eg2", "r0", "r1", "r2", "rho_path", "T_total", "spher_alb")
    )
    qa_valid = finite and not monotonicity_failed and not (
        T_total_was_clamped or rho_path_was_clamped or spher_alb_was_clamped
    )
    return qa_valid, monotonicity_failed, T_total_was_clamped, rho_path_was_clamped, spher_alb_was_clamped


def failure_row_for_dataset(
    state_dict: Dict,
    band: str,
    state_id: str,
    split: str,
    rho1: float,
    rho2: float,
    error_message: str,
    error_type: str,
    mapped_atmosphere_file: Optional[str] = None,
) -> Dict:
    wvl_nm = S2_BANDS_NM.get(band)
    return {
        "band": band,
        "wvl_nm": wvl_nm,
        "sza_deg": state_dict["sza_deg"],
        "vza_deg": state_dict["vza_deg"],
        "raa_deg": state_dict["raa_deg"],
        "aod550": state_dict["aod550"],
        "cwv_cm": state_dict["cwv_cm"],
        "o3_cm": state_dict["o3_cm"],
        "elev_km": state_dict["elev_km"],
        "aerosol_type": state_dict["aerosol_type"],
        "atm_profile": state_dict["atm_profile"],
        "solver": state_dict["solver"],
        "nstr": state_dict["nstr"],
        "rho1": rho1,
        "rho2": rho2,
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
        "state_id": state_id,
        "split": split,
        "qa_valid": False,
        "monotonicity_failed": True,
        "T_total_was_clamped": False,
        "rho_path_was_clamped": False,
        "spher_alb_was_clamped": False,
        "error_message": error_message,
        "error_type": error_type,
        "mapped_atmosphere_file": mapped_atmosphere_file,
    }


def dataset_worker(task):
    state_dict, band, state_id, split, rho1, rho2, debug_first_failure_input = task
    try:
        row = compute_band_coeffs(state_dict, band, rho1=rho1, rho2=rho2)
        qa_valid, monotonicity_failed, t_flag, rp_flag, s_flag = qa_flags_from_row(row)
        row["state_id"] = state_id
        row["split"] = split
        row["qa_valid"] = qa_valid
        row["monotonicity_failed"] = monotonicity_failed
        row["T_total_was_clamped"] = t_flag
        row["rho_path_was_clamped"] = rp_flag
        row["spher_alb_was_clamped"] = s_flag
        return row
    except Exception as e:
        mapped_atmosphere_file = None
        debug_input = None
        try:
            mapped_atmosphere_file = map_atm_profile_libradtran(str(state_dict.get("atm_profile", "")))
            if debug_first_failure_input:
                st = normalize_and_validate_state_labels(
                    RtState(**{k: v for k, v in state_dict.items() if k in RtState.__dataclass_fields__})
                )
                wvl_nm = float(S2_BANDS_NM[band])
                debug_input = "\n\n".join([
                    "# Common block (TOA radiance/edir path)",
                    mk_common_block(st, wvl_nm) + f"albedo {rho1:.8f}\n",
                    "# Surface flux block (eglo path)",
                    mk_surface_flux_block(st, wvl_nm, rho1),
                ])
        except Exception:
            pass

        row = failure_row_for_dataset(
            state_dict=state_dict,
            band=band,
            state_id=state_id,
            split=split,
            rho1=rho1,
            rho2=rho2,
            error_message=str(e),
            error_type=type(e).__name__,
            mapped_atmosphere_file=mapped_atmosphere_file,
        )
        if debug_input is not None:
            row["__debug_uvspec_input"] = debug_input
        return row


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def make_dataset_summary(rows: List[Dict], n_states: int, bands: List[str]) -> Dict:
    rows_per_band = {b: 0 for b in bands}
    qa_false = 0
    mono_true = 0
    t_clamp_true = 0
    rp_clamp_true = 0
    s_clamp_true = 0
    for r in rows:
        if r["band"] in rows_per_band:
            rows_per_band[r["band"]] += 1
        if not r["qa_valid"]:
            qa_false += 1
        if r["monotonicity_failed"]:
            mono_true += 1
        if r["T_total_was_clamped"]:
            t_clamp_true += 1
        if r["rho_path_was_clamped"]:
            rp_clamp_true += 1
        if r["spher_alb_was_clamped"]:
            s_clamp_true += 1
    return {
        "unique_states": n_states,
        "total_rows": len(rows),
        "rows_per_band": rows_per_band,
        "qa_valid_false_count": qa_false,
        "monotonicity_failed_true_count": mono_true,
        "T_total_was_clamped_true_count": t_clamp_true,
        "rho_path_was_clamped_true_count": rp_clamp_true,
        "spher_alb_was_clamped_true_count": s_clamp_true,
    }


def build_dataset_parser() -> argparse.ArgumentParser:
    description = (
        "Run libRadtran rows from a shared normalized state manifest.\n\n"
        "Intended workflow:\n"
        "  1) python data_generator/generate_shared_states.py ...\n"
        "  2) python data_generator/generate_libradtran_dataset.py --states_manifest <manifest> ...\n"
        "  3) python data_generator/generate_6s_dataset.py --states_manifest <manifest> ..."
    )
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--states_manifest", type=str, required=True, help="Path to shared state_manifest.jsonl")
    p.add_argument("--output_dir", type=str, default="data/generated_libradtran_from_manifest")
    p.add_argument("--bands", type=str, default="B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B10,B11,B12")
    p.add_argument("--rho1", type=float, default=0.2)
    p.add_argument("--rho2", type=float, default=0.6)
    p.add_argument("--max_workers", type=int, default=min(8, (os.cpu_count() or 1)))
    p.add_argument("--max_states", type=int, default=None, help="Optional cap for smoke tests: use first N states.")
    p.add_argument("--copy_manifest", action="store_true", help="Copy input manifest into output_dir/state_manifest.jsonl.")
    p.add_argument(
        "--debug_first_failure_input",
        action="store_true",
        help="Write generated uvspec input text for the first failed state-band pair.",
    )
    p.add_argument(
        "--dry_run_mapping",
        action="store_true",
        help="Print normalized state and mapped libRadtran/6S atmosphere+aerosol values, then exit.",
    )
    return p


def main_dataset(argv: Optional[List[str]] = None) -> None:
    args = build_dataset_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bands = parse_csv_list(args.bands)
    manifest_path = Path(args.states_manifest)
    states, state_ids, split_map, include_solver, include_nstr = load_state_manifest(manifest_path)
    original_state_count = len(states)
    if args.max_states is not None:
        if args.max_states <= 0:
            raise ValueError("--max_states must be a positive integer when provided.")
        states = states[: args.max_states]
        state_ids = state_ids[: args.max_states]
        split_map = {sid: split_map[sid] for sid in state_ids}

    if args.dry_run_mapping:
        if not states:
            raise ValueError("No states available for --dry_run_mapping.")
        print_dry_run_mapping(states[0])
        return

    rows_path = output_dir / "dataset_rows.jsonl"
    summary_path = output_dir / "summary.json"
    total_tasks = len(states) * len(bands)
    progress_every = 10
    started_at = datetime.datetime.utcnow()
    max_in_flight = max(1, args.max_workers * 4)

    rows_per_band = {b: 0 for b in bands}
    qa_false = 0
    mono_true = 0
    t_clamp_true = 0
    rp_clamp_true = 0
    s_clamp_true = 0

    def iter_tasks():
        for st, sid in zip(states, state_ids):
            st_dict = asdict(st)
            split = split_map[sid]
            for band in bands:
                yield (st_dict, band, sid, split, args.rho1, args.rho2, bool(args.debug_first_failure_input))

    completed = 0
    pending = set()
    task_iter = iter(iter_tasks())
    failure_debug_printed = 0
    first_failure_input_written = False

    with rows_path.open("w", encoding="utf-8") as rf, ProcessPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
        while len(pending) < max_in_flight:
            try:
                pending.add(ex.submit(dataset_worker, next(task_iter)))
            except StopIteration:
                break

        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                row = fut.result()
                debug_input_text = row.pop("__debug_uvspec_input", None)
                if row.get("error_type") is not None:
                    if failure_debug_printed < 5:
                        print(
                            "Failure row:"
                            f" state_id={row.get('state_id')},"
                            f" band={row.get('band')},"
                            f" atm_profile={row.get('atm_profile')},"
                            f" aerosol_type={row.get('aerosol_type')},"
                            f" mapped_atmosphere_file={row.get('mapped_atmosphere_file')},"
                            f" error_message={row.get('error_message')}"
                        )
                        failure_debug_printed += 1
                    if args.debug_first_failure_input and (not first_failure_input_written) and debug_input_text:
                        debug_path = output_dir / "first_failure_uvspec_input.inp"
                        debug_path.write_text(debug_input_text, encoding="utf-8")
                        print(f"Wrote {debug_path}")
                        first_failure_input_written = True
                rf.write(json.dumps(row) + "\n")
                completed += 1

                band = row.get("band")
                if band in rows_per_band:
                    rows_per_band[band] += 1
                if not row.get("qa_valid", False):
                    qa_false += 1
                if row.get("monotonicity_failed", False):
                    mono_true += 1
                if row.get("T_total_was_clamped", False):
                    t_clamp_true += 1
                if row.get("rho_path_was_clamped", False):
                    rp_clamp_true += 1
                if row.get("spher_alb_was_clamped", False):
                    s_clamp_true += 1

                if (completed % progress_every == 0) or (completed == total_tasks):
                    rf.flush()
                    pct = (100.0 * completed / total_tasks) if total_tasks else 100.0
                    elapsed = (datetime.datetime.utcnow() - started_at).total_seconds()
                    print(f"Progress: {completed}/{total_tasks} rows ({pct:.1f}%), elapsed={elapsed:.1f}s")

            while len(pending) < max_in_flight:
                try:
                    pending.add(ex.submit(dataset_worker, next(task_iter)))
                except StopIteration:
                    break

    summary = {
        "source_model": "libradtran",
        "states_manifest": str(manifest_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "original_unique_states_in_manifest": original_state_count,
        "unique_states": len(states),
        "state_id_includes_solver": bool(include_solver),
        "state_id_includes_nstr": bool(include_nstr),
        "bands": bands,
        "total_rows": completed,
        "rows_per_band": rows_per_band,
        "qa_valid_false_count": qa_false,
        "monotonicity_failed_true_count": mono_true,
        "T_total_was_clamped_true_count": t_clamp_true,
        "rho_path_was_clamped_true_count": rp_clamp_true,
        "spher_alb_was_clamped_true_count": s_clamp_true,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if args.copy_manifest:
        copied_manifest_path = output_dir / "state_manifest.jsonl"
        with manifest_path.open("r", encoding="utf-8") as src, copied_manifest_path.open("w", encoding="utf-8") as dst:
            for line in src:
                dst.write(line)

    print(f"Wrote {rows_path}")
    if args.copy_manifest:
        print(f"Wrote {output_dir / 'state_manifest.jsonl'}")
    print(f"Wrote {summary_path}")


def main():
    main_dataset()


if __name__ == "__main__":
    main()