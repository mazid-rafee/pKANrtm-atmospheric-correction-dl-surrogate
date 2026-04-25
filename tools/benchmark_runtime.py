import argparse
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch

from benchmarks.runtime.runtime_utils import (
    benchmark_callable,
    choose_torch_device,
    collect_machine_info,
    markdown_table,
    save_json,
    synchronize_if_needed,
)
from data_generator.generate_6s_dataset import (
    S2_BANDS_NM,
    _compute_row as compute_6s_row,  # pylint: disable=protected-access
    _normalize_and_validate_state_labels as normalize_6s_state,  # pylint: disable=protected-access
    _run_6s_reflectance_and_eglo_at_wavelength as run_6s_point,  # pylint: disable=protected-access
)
from data_generator.generate_libradtran_dataset import (
    RtState,
    generate_lut_row,
    get_rho_toa_spectrum,
    normalize_and_validate_state_labels,
)
from data_generator.s2_srf import load_srf
from surrogate_pipeline.data_utils import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMNS, resolve_column_names
from surrogate_pipeline.models import build_model


NEURAL_MODELS = {"mlp", "kan", "pmlp", "pkan"}
FIDELITY_MODES = [
    "single_fidelity_libradtran",
    "single_fidelity_6s",
    "oracle_residual",
    "deployable_residual",
    "hybrid_runtime_6s_residual",
]


_SIXS_FEATURE_CACHE: Dict[Tuple, Tuple[float, float, float]] = {}


@dataclass
class PipelineRunner:
    method_name: str
    model_family: str
    fidelity_mode: str
    architecture_tag: str
    split_mode: str
    target_device: str
    parameter_count: Optional[int]
    model_size_mb: Optional[float]
    run_full: Callable[[int], None]


def _git_commit_hash() -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _parse_mode_run(values: Sequence[str]) -> Dict[str, List[Path]]:
    out: Dict[str, List[Path]] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid --mode_run item: {item}. Expected mode=/path/to/run")
        mode, path = item.split("=", 1)
        mode = mode.strip()
        if mode not in FIDELITY_MODES:
            raise ValueError(f"Unsupported mode '{mode}'. Supported: {FIDELITY_MODES}")
        resolved = Path(path).expanduser().resolve()
        out.setdefault(mode, [])
        if resolved not in out[mode]:
            out[mode].append(resolved)
    return out


def _parse_mode_run_root(values: Sequence[str]) -> Dict[str, List[Path]]:
    out: Dict[str, List[Path]] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid --mode_run_root item: {item}. Expected mode=/path/to/root")
        mode, root = item.split("=", 1)
        mode = mode.strip()
        if mode not in FIDELITY_MODES:
            raise ValueError(f"Unsupported mode '{mode}'. Supported: {FIDELITY_MODES}")
        root_path = Path(root).expanduser().resolve()
        if not root_path.exists():
            continue
        discovered = sorted({p.parent for p in root_path.rglob("run_config.json") if p.parent.exists()})
        out.setdefault(mode, [])
        for run_dir in discovered:
            if run_dir not in out[mode]:
                out[mode].append(run_dir)
    return out


def _merge_mode_runs(primary: Dict[str, List[Path]], secondary: Dict[str, List[Path]]) -> Dict[str, List[Path]]:
    out: Dict[str, List[Path]] = {}
    for source in [primary, secondary]:
        for mode, paths in source.items():
            out.setdefault(mode, [])
            for path in paths:
                if path not in out[mode]:
                    out[mode].append(path)
    return out


def _safe_tag(text: str) -> str:
    return "".join(c if c.isalnum() or c in {"-", "_"} else "-" for c in text)


def _progress(message: str) -> None:
    print(f"[progress] {message}", flush=True)


def _locate_split_dir(run_dir: Path, split_mode: str) -> Path:
    # `split_mode` for benchmarking data can be test/val/train/all, while
    # training artifacts are stored under split folders like standard/ood_aod_cwv.
    requested = str(split_mode).strip()
    logical_splits = {"test", "val", "train", "all"}
    artifact_split_candidates = [requested]
    if requested in logical_splits:
        artifact_split_candidates.extend(["standard", "ood_aod_cwv"])

    candidates = []
    for artifact_split in artifact_split_candidates:
        candidates.append(run_dir / "splits" / artifact_split)
        candidates.append(run_dir / run_dir.name / "splits" / artifact_split)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if (run_dir / "models").exists() and (run_dir / "preprocess").exists():
        return run_dir
    discovered = sorted([p for p in run_dir.glob("splits/*") if p.is_dir()])
    if discovered:
        return discovered[0]
    discovered_nested = sorted([p for p in run_dir.glob("*/splits/*") if p.is_dir()])
    if discovered_nested:
        return discovered_nested[0]
    raise FileNotFoundError(f"Could not locate split artifacts under {run_dir} for split_mode={split_mode}")


def _state_from_row(row: pd.Series) -> RtState:
    st = RtState(
        sza_deg=float(row["sza_deg"]),
        vza_deg=float(row["vza_deg"]),
        raa_deg=float(row["raa_deg"]),
        aod550=float(row["aod550"]),
        cwv_cm=float(row["cwv_cm"]),
        o3_cm=float(row["o3_cm"]),
        elev_km=float(row["elev_km"]),
        aerosol_type=str(row.get("aerosol_type", "continental")),
        atm_profile=str(row.get("atm_profile", "midlatitude_summer")),
        solver=str(row.get("solver", "disort")),
        nstr=int(row.get("nstr", 8)),
    )
    return normalize_and_validate_state_labels(st)


def _state_dict_for_6s(row: pd.Series) -> Dict:
    out = {
        "state_id": str(row.get("state_id", "")),
        "sza_deg": float(row["sza_deg"]),
        "vza_deg": float(row["vza_deg"]),
        "raa_deg": float(row["raa_deg"]),
        "aod550": float(row["aod550"]),
        "cwv_cm": float(row["cwv_cm"]),
        "o3_cm": float(row["o3_cm"]),
        "elev_km": float(row["elev_km"]),
        "aerosol_type": str(row.get("aerosol_type", "continental")),
        "atm_profile": str(row.get("atm_profile", "midlatitude_summer")),
        "solver": str(row.get("solver", "6s")),
        "nstr": int(row.get("nstr", 0)),
    }
    return normalize_6s_state(out)


def _sixs_cache_key(row: pd.Series) -> Tuple:
    return (
        str(row.get("band", "B6")),
        float(row["sza_deg"]),
        float(row["vza_deg"]),
        float(row["raa_deg"]),
        float(row["aod550"]),
        float(row["cwv_cm"]),
        float(row["o3_cm"]),
        float(row["elev_km"]),
        str(row.get("aerosol_type", "continental")),
        str(row.get("atm_profile", "midlatitude_summer")),
    )


def _compute_6s_features_for_row(row: pd.Series, rho1: float, rho2: float, use_predefined_atm_profile: bool) -> Tuple[float, float, float]:
    key = _sixs_cache_key(row)
    if key in _SIXS_FEATURE_CACHE:
        return _SIXS_FEATURE_CACHE[key]
    state = _state_dict_for_6s(row)
    band = str(row["band"])
    out = compute_6s_row((state, band, rho1, rho2, use_predefined_atm_profile))
    values = (float(out["rho_path"]), float(out["T_total"]), float(out["spher_alb"]))
    _SIXS_FEATURE_CACHE[key] = values
    return values


def _build_feature_frame(
    infer_df: pd.DataFrame,
    feature_names: Sequence[str],
    rho1: float,
    rho2: float,
    use_predefined_atm_profile: bool,
) -> pd.DataFrame:
    frame = infer_df.copy()
    needed = list(feature_names)

    base_from_suffix = {}
    for name in needed:
        if name.endswith("_lrt"):
            base_from_suffix[name] = name[: -len("_lrt")]
        elif name.endswith("_6s"):
            base_from_suffix[name] = name[: -len("_6s")]

    # Copy compatible unsuffixed columns into suffixed feature slots.
    for suffixed, base in base_from_suffix.items():
        if suffixed in frame.columns:
            continue
        if base in frame.columns:
            frame[suffixed] = frame[base]

    # If 6S low-fidelity target columns are required but missing, compute them on demand.
    needed_6s_targets = [c for c in ["rho_path_6s", "T_total_6s", "spher_alb_6s"] if c in needed and c not in frame.columns]
    if needed_6s_targets:
        rho_vals = []
        t_vals = []
        s_vals = []
        for _, row in frame.iterrows():
            rho_p, t_total, sph = _compute_6s_features_for_row(
                row=row,
                rho1=rho1,
                rho2=rho2,
                use_predefined_atm_profile=use_predefined_atm_profile,
            )
            rho_vals.append(rho_p)
            t_vals.append(t_total)
            s_vals.append(sph)
        if "rho_path_6s" in needed_6s_targets:
            frame["rho_path_6s"] = np.asarray(rho_vals, dtype=np.float32)
        if "T_total_6s" in needed_6s_targets:
            frame["T_total_6s"] = np.asarray(t_vals, dtype=np.float32)
        if "spher_alb_6s" in needed_6s_targets:
            frame["spher_alb_6s"] = np.asarray(s_vals, dtype=np.float32)

    missing = [c for c in needed if c not in frame.columns]
    if missing:
        raise KeyError(f"Missing required preprocessor features: {missing}")
    return frame[needed]


def _sample_benchmark_rows(df: pd.DataFrame, n_samples: int, seed: int, split_mode: str) -> pd.DataFrame:
    xdf = df.copy()
    if "qa_valid" in xdf.columns:
        xdf = xdf[xdf["qa_valid"].astype(bool)].copy()
    if split_mode != "all" and "split" in xdf.columns:
        if split_mode == "test":
            xdf = xdf[xdf["split"].astype(str).str.lower().eq("test")].copy()
        elif split_mode == "val":
            xdf = xdf[xdf["split"].astype(str).str.lower().eq("val")].copy()
        elif split_mode == "train":
            xdf = xdf[xdf["split"].astype(str).str.lower().eq("train")].copy()
    if xdf.empty:
        raise RuntimeError("No rows available after split/QA filtering.")
    if len(xdf) <= n_samples:
        return xdf.reset_index(drop=True)
    return xdf.sample(n=n_samples, random_state=seed).reset_index(drop=True)


def _ensure_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "band" not in out.columns:
        out["band"] = "B6"
    if "wvl_nm" not in out.columns:
        out["wvl_nm"] = out["band"].astype(str).map(lambda b: float(S2_BANDS_NM.get(b, S2_BANDS_NM["B6"])))
    return out


def _stats_from_samples(sample_latencies_ms: List[float]) -> Dict[str, float]:
    arr = np.asarray(sample_latencies_ms, dtype=float)
    mean_ms = float(arr.mean())
    return {
        "mean_latency_ms_per_sample": mean_ms,
        "std_latency_ms_per_sample": float(arr.std(ddof=0)),
        "median_latency_ms_per_sample": float(np.median(arr)),
        "p95_latency_ms_per_sample": float(np.percentile(arr, 95)),
        "throughput_samples_per_sec": float(1000.0 / max(mean_ms, 1e-12)),
    }


def _benchmark_rtm_calls(
    rows: pd.DataFrame,
    repeats: int,
    call_fn: Callable[[pd.Series], None],
    label: str,
) -> Tuple[Dict[str, float], Dict]:
    sample_latencies_ms: List[float] = []
    repeat_totals: List[float] = []
    total = max(1, repeats)
    for rep_idx in range(total):
        _progress(f"{label}: repeat {rep_idx + 1}/{total} started")
        t0_repeat = time.perf_counter()
        for _, row in rows.iterrows():
            t0 = time.perf_counter()
            call_fn(row)
            sample_latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        repeat_totals.append(time.perf_counter() - t0_repeat)
        _progress(f"{label}: repeat {rep_idx + 1}/{total} finished")
    stats = _stats_from_samples(sample_latencies_ms)
    raw = {
        "sample_latencies_ms": [float(x) for x in sample_latencies_ms],
        "repeat_total_seconds": [float(x) for x in repeat_totals],
    }
    return stats, raw


def _libradtran_call(row: pd.Series, rho1: float, rho2: float) -> None:
    st = _state_from_row(row)
    band = str(row["band"])
    _ = generate_lut_row(st, band=band, rho1=rho1, rho2=rho2, capture_debug=False)


def _sixs_call(row: pd.Series, rho1: float, rho2: float, use_predefined_atm_profile: bool) -> None:
    state = _state_dict_for_6s(row)
    band = str(row["band"])
    _ = compute_6s_row((state, band, rho1, rho2, use_predefined_atm_profile))


def _estimate_model_size_mb(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    return float(path.stat().st_size / (1024.0 * 1024.0))


def _load_model_build_kwargs(split_dir: Path) -> Dict:
    defaults = {
        "mlp_arch": "baseline",
        "kan_arch": "baseline",
        "activation": "relu",
        "use_layernorm": False,
        "deployable_stage_sizing": False,
    }
    candidates = [split_dir]
    candidates.extend(list(split_dir.parents)[:6])
    for base in candidates:
        cfg = base / "run_config.json"
        if not cfg.exists():
            continue
        try:
            payload = json.loads(cfg.read_text(encoding="utf-8"))
        except Exception:
            continue
        out = defaults.copy()
        out["mlp_arch"] = payload.get("mlp_arch", defaults["mlp_arch"])
        out["kan_arch"] = payload.get("kan_arch", defaults["kan_arch"])
        out["activation"] = payload.get("activation", defaults["activation"])
        out["use_layernorm"] = bool(payload.get("use_layernorm", defaults["use_layernorm"]))
        out["deployable_stage_sizing"] = bool(payload.get("deployable_stage_sizing", defaults["deployable_stage_sizing"]))
        return out
    return defaults


def _predict_neural_in_batches(model: torch.nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    out = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            xb = torch.from_numpy(x[i : i + batch_size]).float().to(device, non_blocking=True)
            out.append(model(xb).detach().cpu().numpy())
    return np.vstack(out).astype(np.float32)


def _load_single_stage_components(
    split_dir: Path,
    model_name: str,
    infer_df: pd.DataFrame,
    device_name: str,
    rho1: float,
    rho2: float,
    use_predefined_atm_profile: bool,
) -> Optional[Tuple[np.ndarray, Callable[[np.ndarray, int, int], np.ndarray], Optional[int], Optional[float], Optional[torch.device]]]:
    preproc_path = split_dir / "preprocess" / "preprocessor.joblib"
    scaler_path = split_dir / "preprocess" / "target_scaler.joblib"
    model_dir = split_dir / "models" / model_name
    if not (preproc_path.exists() and scaler_path.exists() and model_dir.exists()):
        return None

    preprocessor = joblib.load(preproc_path)
    scaler = joblib.load(scaler_path)
    feature_names = list(preprocessor.feature_names_in_)
    feature_frame = _build_feature_frame(
        infer_df=infer_df,
        feature_names=feature_names,
        rho1=rho1,
        rho2=rho2,
        use_predefined_atm_profile=use_predefined_atm_profile,
    )
    x = preprocessor.transform(feature_frame)
    if hasattr(x, "toarray"):
        x = x.toarray()
    x = np.asarray(x, dtype=np.float32)

    if model_name in NEURAL_MODELS:
        ckpt_path = model_dir / "best.pt"
        if not ckpt_path.exists():
            return None
        device = choose_torch_device(device_name)
        model_kwargs = _load_model_build_kwargs(split_dir)
        model, _ = build_model(
            model_name,
            in_dim=x.shape[1],
            out_dim=len(TARGET_COLUMNS),
            stage_name="single_stage",
            mlp_arch=model_kwargs["mlp_arch"],
            kan_arch=model_kwargs["kan_arch"],
            activation=model_kwargs["activation"],
            use_layernorm=model_kwargs["use_layernorm"],
            deployable_stage_sizing=model_kwargs["deployable_stage_sizing"],
        )
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device).eval()
        param_count = int(sum(p.numel() for p in model.parameters()))
        model_size_mb = _estimate_model_size_mb(ckpt_path)

        def runner(arr: np.ndarray, batch_size: int, start_idx: int = 0) -> np.ndarray:
            _ = start_idx
            pred_scaled = _predict_neural_in_batches(model, arr, batch_size=batch_size, device=device)
            return scaler.inverse_transform(pred_scaled).astype(np.float32)

        return x, runner, param_count, model_size_mb, device

    model_path = model_dir / "model.joblib"
    if not model_path.exists():
        return None
    models = joblib.load(model_path)
    model_size_mb = _estimate_model_size_mb(model_path)

    def runner(arr: np.ndarray, batch_size: int, start_idx: int = 0) -> np.ndarray:
        _ = start_idx
        _ = batch_size
        pred_scaled = np.column_stack([m.predict(arr) for m in models]).astype(np.float32)
        return scaler.inverse_transform(pred_scaled).astype(np.float32)

    return x, runner, None, model_size_mb, None


def _load_deployable_components(
    split_dir: Path,
    model_name: str,
    infer_df: pd.DataFrame,
    device_name: str,
    rho1: float,
    rho2: float,
    use_predefined_atm_profile: bool,
) -> Optional[Tuple[np.ndarray, Callable[[np.ndarray, int, int], np.ndarray], Optional[int], Optional[float], Optional[torch.device]]]:
    model_root = split_dir / "models" / model_name
    stage1 = model_root / "stage1_6s"
    stage2 = model_root / "stage2_residual"
    if not (stage1.exists() and stage2.exists()):
        return None

    pre1 = joblib.load(stage1 / "preprocessor.joblib")
    sc1 = joblib.load(stage1 / "target_scaler.joblib")
    pre2 = joblib.load(stage2 / "preprocessor.joblib")
    sc2 = joblib.load(stage2 / "target_scaler.joblib")

    stage1_frame = _build_feature_frame(
        infer_df=infer_df,
        feature_names=list(pre1.feature_names_in_),
        rho1=rho1,
        rho2=rho2,
        use_predefined_atm_profile=use_predefined_atm_profile,
    )
    x1 = pre1.transform(stage1_frame)
    if hasattr(x1, "toarray"):
        x1 = x1.toarray()
    x1 = np.asarray(x1, dtype=np.float32)

    if model_name in NEURAL_MODELS:
        device = choose_torch_device(device_name)
        model_kwargs = _load_model_build_kwargs(split_dir)
        m1, _ = build_model(
            model_name,
            in_dim=x1.shape[1],
            out_dim=len(TARGET_COLUMNS),
            stage_name="stage1_6s",
            mlp_arch=model_kwargs["mlp_arch"],
            kan_arch=model_kwargs["kan_arch"],
            activation=model_kwargs["activation"],
            use_layernorm=model_kwargs["use_layernorm"],
            deployable_stage_sizing=model_kwargs["deployable_stage_sizing"],
        )
        dummy = infer_df.copy()
        for target in TARGET_COLUMNS:
            dummy[f"pred_{target}_6s"] = 0.0
        stage2_dummy_frame = _build_feature_frame(
            infer_df=dummy,
            feature_names=list(pre2.feature_names_in_),
            rho1=rho1,
            rho2=rho2,
            use_predefined_atm_profile=use_predefined_atm_profile,
        )
        x2_dummy = pre2.transform(stage2_dummy_frame)
        if hasattr(x2_dummy, "toarray"):
            x2_dummy = x2_dummy.toarray()
        x2_dummy = np.asarray(x2_dummy, dtype=np.float32)
        m2, _ = build_model(
            model_name,
            in_dim=x2_dummy.shape[1],
            out_dim=len(TARGET_COLUMNS),
            stage_name="stage2_residual",
            mlp_arch=model_kwargs["mlp_arch"],
            kan_arch=model_kwargs["kan_arch"],
            activation=model_kwargs["activation"],
            use_layernorm=model_kwargs["use_layernorm"],
            deployable_stage_sizing=model_kwargs["deployable_stage_sizing"],
        )
        ck1 = torch.load(stage1 / "best.pt", map_location=device)
        ck2 = torch.load(stage2 / "best.pt", map_location=device)
        m1.load_state_dict(ck1["model_state_dict"])
        m2.load_state_dict(ck2["model_state_dict"])
        m1 = m1.to(device).eval()
        m2 = m2.to(device).eval()
        param_count = int(sum(p.numel() for p in m1.parameters()) + sum(p.numel() for p in m2.parameters()))
        model_size_mb = (_estimate_model_size_mb(stage1 / "best.pt") or 0.0) + (_estimate_model_size_mb(stage2 / "best.pt") or 0.0)

        def runner(arr: np.ndarray, batch_size: int, start_idx: int = 0) -> np.ndarray:
            p1s = _predict_neural_in_batches(m1, arr, batch_size=batch_size, device=device)
            p1 = sc1.inverse_transform(p1s).astype(np.float32)
            chunk_df = infer_df.iloc[start_idx : start_idx + arr.shape[0]].copy()
            for idx, target in enumerate(TARGET_COLUMNS):
                chunk_df[f"pred_{target}_6s"] = p1[:, idx]
            stage2_chunk_frame = _build_feature_frame(
                infer_df=chunk_df,
                feature_names=list(pre2.feature_names_in_),
                rho1=rho1,
                rho2=rho2,
                use_predefined_atm_profile=use_predefined_atm_profile,
            )
            x2 = pre2.transform(stage2_chunk_frame)
            if hasattr(x2, "toarray"):
                x2 = x2.toarray()
            x2 = np.asarray(x2, dtype=np.float32)
            p2s = _predict_neural_in_batches(m2, x2, batch_size=batch_size, device=device)
            p2 = sc2.inverse_transform(p2s).astype(np.float32)
            return (p1 + p2).astype(np.float32)

        return x1, runner, param_count, model_size_mb, device

    mdl1 = joblib.load(stage1 / "model.joblib")
    mdl2 = joblib.load(stage2 / "model.joblib")
    model_size_mb = (_estimate_model_size_mb(stage1 / "model.joblib") or 0.0) + (_estimate_model_size_mb(stage2 / "model.joblib") or 0.0)

    def runner(arr: np.ndarray, batch_size: int, start_idx: int = 0) -> np.ndarray:
        _ = batch_size
        p1s = np.column_stack([m.predict(arr) for m in mdl1]).astype(np.float32)
        p1 = sc1.inverse_transform(p1s).astype(np.float32)
        chunk_df = infer_df.iloc[start_idx : start_idx + arr.shape[0]].copy()
        for idx, target in enumerate(TARGET_COLUMNS):
            chunk_df[f"pred_{target}_6s"] = p1[:, idx]
        stage2_chunk_frame = _build_feature_frame(
            infer_df=chunk_df,
            feature_names=list(pre2.feature_names_in_),
            rho1=rho1,
            rho2=rho2,
            use_predefined_atm_profile=use_predefined_atm_profile,
        )
        x2 = pre2.transform(stage2_chunk_frame)
        if hasattr(x2, "toarray"):
            x2 = x2.toarray()
        x2 = np.asarray(x2, dtype=np.float32)
        p2s = np.column_stack([m.predict(x2) for m in mdl2]).astype(np.float32)
        p2 = sc2.inverse_transform(p2s).astype(np.float32)
        return (p1 + p2).astype(np.float32)

    return x1, runner, None, model_size_mb, None


def _build_process_full_fn(
    x: np.ndarray,
    runner: Callable[[np.ndarray, int, int], np.ndarray],
    batch_size: int,
    sync_device: Optional[torch.device],
) -> Callable[[], None]:
    def run_once() -> None:
        for i in range(0, x.shape[0], batch_size):
            _ = runner(x[i : i + batch_size], batch_size, i)
        synchronize_if_needed(sync_device)

    return run_once


def _build_runtime6s_residual_runner(
    split_dir: Path,
    model_name: str,
    infer_df: pd.DataFrame,
    device_name: str,
    rho1: float,
    rho2: float,
    use_predefined_atm_profile: bool,
) -> Optional[Tuple[Callable[[int], None], Optional[int], Optional[float], Optional[torch.device], str]]:
    bundle = _load_single_stage_components(
        split_dir=split_dir,
        model_name=model_name,
        infer_df=infer_df,
        device_name=device_name,
        rho1=rho1,
        rho2=rho2,
        use_predefined_atm_profile=use_predefined_atm_profile,
    )
    if bundle is None:
        return None
    _, runner_model, param_count, model_size_mb, sync_device = bundle

    preproc = joblib.load(split_dir / "preprocess" / "preprocessor.joblib")
    feature_names = list(preproc.feature_names_in_)
    if not all(col in feature_names for col in ["rho_path_6s", "T_total_6s", "spher_alb_6s"]):
        return None

    surrogate_device_label = "cpu+gpu" if (sync_device is not None and sync_device.type == "cuda") else "cpu"

    def run_full(batch_size: int) -> None:
        for i in range(0, len(infer_df), batch_size):
            chunk = infer_df.iloc[i : i + batch_size].copy()
            sixs_features = []
            for _, row in chunk.iterrows():
                state = _state_dict_for_6s(row)
                band = str(row["band"])
                out = compute_6s_row((state, band, rho1, rho2, use_predefined_atm_profile))
                sixs_features.append([out["rho_path"], out["T_total"], out["spher_alb"]])
            arr = np.asarray(sixs_features, dtype=np.float32)
            chunk["rho_path_6s"] = arr[:, 0]
            chunk["T_total_6s"] = arr[:, 1]
            chunk["spher_alb_6s"] = arr[:, 2]
            x = preproc.transform(chunk[feature_names])
            if hasattr(x, "toarray"):
                x = x.toarray()
            x = np.asarray(x, dtype=np.float32)
            _ = runner_model(x, batch_size)
        synchronize_if_needed(sync_device)

    return run_full, param_count, model_size_mb, sync_device, surrogate_device_label


def _build_pipelines(
    mode_runs: Dict[str, List[Path]],
    infer_df: pd.DataFrame,
    models: Sequence[str],
    split_mode: str,
    devices: Sequence[str],
    rho1: float,
    rho2: float,
    use_predefined_atm_profile: bool,
    oracle_runtime_behavior: str,
) -> List[PipelineRunner]:
    runners: List[PipelineRunner] = []
    for fidelity_mode, run_dirs in mode_runs.items():
        _progress(f"pipeline discovery: mode={fidelity_mode}, run_dirs={len(run_dirs)}")
        for run_dir in run_dirs:
            try:
                split_dir = _locate_split_dir(run_dir, split_mode=split_mode)
            except FileNotFoundError:
                _progress(f"skipping run without compatible split artifacts: {run_dir}")
                continue
            architecture_tag = run_dir.parent.name if run_dir.parent != run_dir else run_dir.name
            arch_token = _safe_tag(architecture_tag)
            _progress(f"pipeline discovery: using split_dir={split_dir} (variant={architecture_tag})")
            for model_name in models:
                if fidelity_mode == "deployable_residual":
                    builder = _load_deployable_components
                else:
                    builder = _load_single_stage_components

                for device_name in devices:
                    _progress(
                        f"pipeline discovery: loading runner mode={fidelity_mode} "
                        f"variant={architecture_tag} model={model_name} device={device_name}"
                    )
                    if model_name not in NEURAL_MODELS and str(device_name).lower() != "cpu":
                        _progress(
                            f"pipeline discovery: skipped mode={fidelity_mode} variant={architecture_tag} "
                            f"model={model_name} device={device_name} (non-neural supports cpu only)"
                        )
                        continue
                    if str(device_name).startswith("cuda") and not torch.cuda.is_available():
                        _progress(
                            f"pipeline discovery: skipped mode={fidelity_mode} variant={architecture_tag} "
                            f"model={model_name} device={device_name} (cuda unavailable)"
                        )
                        continue
                    bundle = builder(
                        split_dir=split_dir,
                        model_name=model_name,
                        infer_df=infer_df,
                        device_name=device_name,
                        rho1=rho1,
                        rho2=rho2,
                        use_predefined_atm_profile=use_predefined_atm_profile,
                    )
                    if bundle is None:
                        _progress(
                            f"pipeline discovery: skipped mode={fidelity_mode} variant={architecture_tag} "
                            f"model={model_name} device={device_name} (artifacts missing/incompatible)"
                        )
                        continue
                    x, runner, param_count, model_size_mb, sync_device = bundle
                    target_device = str(sync_device) if sync_device is not None else "cpu"
                    method_name = f"{fidelity_mode}_{arch_token}_{model_name}_{target_device.replace(':', '-')}"
                    run_full = lambda bs, x=x, runner=runner, sd=sync_device: _build_process_full_fn(x=x, runner=runner, batch_size=bs, sync_device=sd)()
                    runners.append(
                        PipelineRunner(
                            method_name=method_name,
                            model_family=model_name,
                            fidelity_mode=fidelity_mode,
                            architecture_tag=architecture_tag,
                            split_mode=split_mode,
                            target_device=target_device,
                            parameter_count=param_count,
                            model_size_mb=model_size_mb,
                            run_full=run_full,
                        )
                    )
                    _progress(
                        f"pipeline discovery: loaded runner mode={fidelity_mode} variant={architecture_tag} "
                        f"model={model_name} target_device={target_device}"
                    )

                needs_runtime6s = (
                    fidelity_mode == "hybrid_runtime_6s_residual"
                    or (fidelity_mode == "oracle_residual" and oracle_runtime_behavior in {"runtime6s", "both"})
                )
                if needs_runtime6s:
                    for device_name in devices:
                        _progress(
                            f"pipeline discovery: loading runtime6s runner mode={fidelity_mode} "
                            f"variant={architecture_tag} model={model_name} device={device_name}"
                        )
                        if model_name not in NEURAL_MODELS and str(device_name).lower() != "cpu":
                            _progress(
                                f"pipeline discovery: skipped runtime6s mode={fidelity_mode} variant={architecture_tag} "
                                f"model={model_name} device={device_name} (non-neural supports cpu only)"
                            )
                            continue
                        rt6_bundle = _build_runtime6s_residual_runner(
                            split_dir=split_dir,
                            model_name=model_name,
                            infer_df=infer_df,
                            device_name=device_name,
                            rho1=rho1,
                            rho2=rho2,
                            use_predefined_atm_profile=use_predefined_atm_profile,
                        )
                        if rt6_bundle is None:
                            _progress(
                                f"pipeline discovery: skipped runtime6s mode={fidelity_mode} variant={architecture_tag} "
                                f"model={model_name} device={device_name} (artifacts/features unavailable)"
                            )
                            continue
                        run_full, param_count, model_size_mb, _, target_device = rt6_bundle
                        method_name = f"{fidelity_mode}_runtime6s_{arch_token}_{model_name}_{target_device.replace(':', '-')}"
                        runners.append(
                            PipelineRunner(
                                method_name=method_name,
                                model_family=model_name,
                                fidelity_mode=fidelity_mode,
                                architecture_tag=architecture_tag,
                                split_mode=split_mode,
                                target_device=target_device,
                                parameter_count=param_count,
                                model_size_mb=model_size_mb,
                                run_full=run_full,
                            )
                        )
                        _progress(
                            f"pipeline discovery: loaded runtime6s runner mode={fidelity_mode} variant={architecture_tag} "
                            f"model={model_name} target_device={target_device}"
                        )
    _progress(f"pipeline discovery complete: total_runners={len(runners)}")
    return runners


def _write_plots(results_df: pd.DataFrame, batch1_df: pd.DataFrame, throughput_df: pd.DataFrame, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    non_rtm = batch1_df[batch1_df["model_family"] != "rtm"].copy()
    if non_rtm.empty:
        return
    view = non_rtm.copy()
    view = view.sort_values("mean_latency_ms_per_sample")

    plt.figure(figsize=(12, 5))
    plt.bar(view["method_name"], view["mean_latency_ms_per_sample"])
    plt.ylabel("Mean latency (ms/sample)")
    plt.title("Mean latency by method")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "latency_by_method.png", dpi=160)
    plt.close()

    spd = view.copy()
    plt.figure(figsize=(12, 5))
    plt.bar(spd["method_name"], spd["speedup_vs_libradtran"].fillna(0.0))
    plt.ylabel("Speedup vs libRadtran")
    plt.title("Speedup vs libRadtran")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "speedup_vs_libradtran.png", dpi=160)
    plt.close()

    throughput_view = throughput_df.copy()
    if not throughput_view.empty:
        throughput_view = throughput_view.sort_values("throughput_samples_per_sec", ascending=False)
        plt.figure(figsize=(12, 5))
        plt.bar(throughput_view["method_name"], throughput_view["throughput_samples_per_sec"])
        plt.ylabel("Throughput (samples/s)")
        plt.title("Throughput by method (best batch size)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(out_dir / "throughput_by_method.png", dpi=160)
        plt.close()

    cpu_view = view[view["target_device"] == "cpu"]
    gpu_view = view[view["target_device"].str.contains("cuda|gpu", case=False, regex=True) | view["target_device"].str.contains("cpu\\+gpu", regex=True)]

    if not cpu_view.empty:
        plt.figure(figsize=(12, 5))
        plt.bar(cpu_view["method_name"], cpu_view["mean_latency_ms_per_sample"])
        plt.ylabel("Latency (ms/sample)")
        plt.title("CPU-only practical comparison")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(out_dir / "cpu_only_comparison.png", dpi=160)
        plt.close()

    if not gpu_view.empty:
        plt.figure(figsize=(12, 5))
        plt.bar(gpu_view["method_name"], gpu_view["mean_latency_ms_per_sample"])
        plt.ylabel("Latency (ms/sample)")
        plt.title("Practical deployment comparison")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(out_dir / "practical_deployment_comparison.png", dpi=160)
        plt.close()


def _build_markdown_report(
    machine_info: Dict,
    results_all_df: pd.DataFrame,
    batch1_df: pd.DataFrame,
    throughput_df: pd.DataFrame,
    rtm_srf_df: Optional[pd.DataFrame],
    warnings: List[str],
) -> str:
    lines = ["# Runtime Benchmark Results", ""]
    if warnings:
        lines.extend(["## Sanity Warnings", ""])
        for item in warnings:
            lines.append(f"- {item}")
        lines.append("")

    gpu_names = ", ".join([g["name"] for g in machine_info.get("gpu", {}).get("gpus", [])]) or "N/A"
    table_a = markdown_table(
        headers=["host", "CPU", "RAM (GB)", "GPU", "OS", "Python", "PyTorch", "CUDA"],
        rows=[
            [
                machine_info.get("hostname", ""),
                machine_info.get("cpu", {}).get("model_name", ""),
                f"{machine_info.get('memory', {}).get('total_ram_gb', ''):.2f}" if machine_info.get("memory", {}).get("total_ram_gb") else "",
                gpu_names,
                machine_info.get("os", {}).get("platform", ""),
                machine_info.get("python_version", ""),
                machine_info.get("torch_version", ""),
                machine_info.get("cuda_version", ""),
            ]
        ],
    )
    lines.extend(["## Table A: Machine details", "", table_a, ""])

    table_b_df = results_all_df[results_all_df["model_family"] == "rtm"].copy()
    table_b = markdown_table(
        headers=["method", "device", "batch_size", "latency_ms", "throughput_sps"],
        rows=[
            [
                r["method_name"],
                r["target_device"],
                int(r["batch_size"]),
                f"{r['mean_latency_ms_per_sample']:.4f}",
                f"{r['throughput_samples_per_sec']:.4f}",
            ]
            for _, r in table_b_df.iterrows()
        ],
    )
    lines.extend(["## Table B: RTM baseline timings", "", table_b, ""])

    cpu_df = batch1_df[(batch1_df["target_device"] == "cpu") & (batch1_df["model_family"] != "rtm")].copy()
    cpu_df = cpu_df.sort_values("mean_latency_ms_per_sample")
    table_c = markdown_table(
        headers=["method", "fidelity_mode", "model", "batch_size", "latency_ms", "speedup_vs_libradtran"],
        rows=[
            [
                r["method_name"],
                r["fidelity_mode"],
                r["model_family"],
                int(r["batch_size"]),
                f"{r['mean_latency_ms_per_sample']:.4f}",
                f"{r['speedup_vs_libradtran']:.4f}",
            ]
            for _, r in cpu_df.iterrows()
        ],
    )
    lines.extend(["## Table C: CPU latency comparison (batch_size=1 only)", "", table_c, ""])

    gpu_df = batch1_df[
        (batch1_df["target_device"].str.contains("cuda|cpu\\+gpu", case=False, regex=True)) & (batch1_df["model_family"] != "rtm")
    ].copy()
    gpu_df = gpu_df.sort_values("mean_latency_ms_per_sample")
    table_d = markdown_table(
        headers=["method", "fidelity_mode", "model", "device", "batch_size", "latency_ms", "speedup_vs_libradtran"],
        rows=[
            [
                r["method_name"],
                r["fidelity_mode"],
                r["model_family"],
                r["target_device"],
                int(r["batch_size"]),
                f"{r['mean_latency_ms_per_sample']:.4f}",
                f"{r['speedup_vs_libradtran']:.4f}",
            ]
            for _, r in gpu_df.iterrows()
        ],
    )
    lines.extend(["## Table D: GPU latency comparison (batch_size=1 only)", "", table_d, ""])

    table_e = markdown_table(
        headers=["method", "fidelity_mode", "model", "device", "batch_size", "total_elapsed_sec", "latency_ms", "samples_per_sec"],
        rows=[
            [
                r["method_name"],
                r["fidelity_mode"],
                r["model_family"],
                r["target_device"],
                int(r["batch_size"]),
                f"{r['mean_total_elapsed_sec']:.6f}",
                f"{r['mean_latency_ms_per_sample']:.4f}",
                f"{r['throughput_samples_per_sec']:.2f}",
            ]
            for _, r in throughput_df.iterrows()
        ],
    )
    lines.extend(
        [
            "## Table E: Throughput comparison (best batch size per method; amortized, not batch_size=1 latency)",
            "",
            table_e,
            "",
        ]
    )

    if rtm_srf_df is not None and not rtm_srf_df.empty:
        table_f = markdown_table(
            headers=["method", "no_srf_ms", "srf_ms", "increase_pct"],
            rows=[
                [r["method"], f"{r['no_srf_ms']:.4f}", f"{r['srf_ms']:.4f}", f"{r['increase_pct']:.2f}%"]
                for _, r in rtm_srf_df.iterrows()
            ],
        )
        lines.extend(["## Table F: RTM timing sensitivity to SRF-aware band integration", "", table_f, ""])

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Full runtime benchmarking pipeline for atmospheric correction surrogates.")
    parser.add_argument("--data", required=True, help="Main benchmark dataset JSONL (central wavelength / non-SRF).")
    parser.add_argument(
        "--mode_run",
        nargs="+",
        default=[],
        help="Mode to run-dir mapping (repeat mode for multiple variants), e.g. deployable_residual=/path/to/run",
    )
    parser.add_argument(
        "--mode_run_root",
        nargs="+",
        default=[],
        help="Auto-discover variants under roots via run_config.json, e.g. deployable_residual=/path/to/root",
    )
    parser.add_argument("--models", nargs="+", default=["mlp", "kan"], help="Shortlisted model families to benchmark.")
    parser.add_argument("--split_mode", default="test", choices=["test", "val", "train", "all"])
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda:0"], help="Devices for surrogate/pipeline benchmark.")
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[1, 256], help="Batch sizes for surrogate/pipeline benchmarks.")
    parser.add_argument("--n_samples", type=int, default=256, help="Number of representative rows.")
    parser.add_argument("--repeats", type=int, default=5, help="Number of timing repeats.")
    parser.add_argument("--warmup_runs", type=int, default=2, help="Warmup runs per benchmark entry.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rho1", type=float, default=0.2)
    parser.add_argument("--rho2", type=float, default=0.6)
    parser.add_argument("--output_dir", default="benchmarks/runtime")
    parser.add_argument("--use_predefined_atm_profile", action="store_true", help="Use predefined atmosphere profile in 6S calls.")
    parser.add_argument(
        "--oracle_runtime_behavior",
        default="actual",
        choices=["actual", "runtime6s", "both"],
        help="How to benchmark oracle_residual: actual implementation, runtime 6S correction, or both.",
    )
    parser.add_argument("--include_srf_sensitivity", action="store_true")
    parser.add_argument("--srf_samples", type=int, default=16)
    args = parser.parse_args()

    _progress("benchmark initialization started")
    out_dir = Path(args.output_dir).resolve()
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    mode_runs = _merge_mode_runs(_parse_mode_run(args.mode_run), _parse_mode_run_root(args.mode_run_root))
    if not mode_runs:
        raise RuntimeError("No mode runs provided. Use --mode_run and/or --mode_run_root.")
    total_run_dirs = sum(len(v) for v in mode_runs.values())
    _progress(f"discovered {total_run_dirs} run directories across {len(mode_runs)} fidelity modes")
    used_gpu_benchmark = any(str(d).startswith("cuda") for d in args.devices) and torch.cuda.is_available()
    machine_info = collect_machine_info(used_gpu_benchmark=used_gpu_benchmark)
    machine_info["git_commit_hash"] = _git_commit_hash()
    _progress("machine info collected")

    df = pd.read_json(args.data, lines=True)
    resolved = resolve_column_names(
        df=df,
        required=CATEGORICAL_FEATURES + [n for n in NUMERIC_FEATURES if n != "wvl_nm"] + TARGET_COLUMNS,
        optional=["state_id", "solver", "nstr", "qa_valid", "split", "wvl_nm"],
    )
    rename_map = {v: k for k, v in resolved.items() if v is not None}
    df = df.rename(columns=rename_map)
    sampled = _sample_benchmark_rows(df=df, n_samples=args.n_samples, seed=args.seed, split_mode=args.split_mode)
    sampled = _ensure_feature_columns(sampled)
    sampled_states_csv = out_dir / "sampled_inputs.csv"
    sampled.to_csv(sampled_states_csv, index=False)
    _progress(f"sampled {len(sampled)} benchmark rows from input dataset")

    results: List[Dict] = []
    raw_timings: Dict[str, Dict] = {}

    lrt_stats, lrt_raw = _benchmark_rtm_calls(
        rows=sampled,
        repeats=args.repeats,
        call_fn=lambda row: _libradtran_call(row, rho1=args.rho1, rho2=args.rho2),
        label="libRadtran CPU baseline",
    )
    raw_timings["libRadtran_cpu"] = lrt_raw
    lrt_mean_total_sec = float(np.mean(lrt_raw.get("repeat_total_seconds", [0.0])))
    results.append(
        {
            "method_name": "libRadtran_cpu",
            "model_family": "rtm",
            "fidelity_mode": "rtm_baseline",
            "architecture_tag": "rtm",
            "split_mode": args.split_mode,
            "target_device": "cpu",
            "batch_size": 1,
            "num_samples": int(len(sampled)),
            "num_repeats": int(args.repeats),
            "warmup_runs": 0,
            "parameter_count": None,
            "model_size_mb": None,
            "mean_total_elapsed_sec": lrt_mean_total_sec,
            **lrt_stats,
        }
    )

    sixs_baseline_available = True
    try:
        sixs_stats, sixs_raw = _benchmark_rtm_calls(
            rows=sampled,
            repeats=args.repeats,
            call_fn=lambda row: _sixs_call(
                row,
                rho1=args.rho1,
                rho2=args.rho2,
                use_predefined_atm_profile=bool(args.use_predefined_atm_profile),
            ),
            label="6S CPU baseline",
        )
        raw_timings["6S_cpu"] = sixs_raw
        sixs_mean_total_sec = float(np.mean(sixs_raw.get("repeat_total_seconds", [0.0])))
        results.append(
            {
                "method_name": "6S_cpu",
                "model_family": "rtm",
                "fidelity_mode": "rtm_baseline",
                "architecture_tag": "rtm",
                "split_mode": args.split_mode,
                "target_device": "cpu",
                "batch_size": 1,
                "num_samples": int(len(sampled)),
                "num_repeats": int(args.repeats),
                "warmup_runs": 0,
                "parameter_count": None,
                "model_size_mb": None,
                "mean_total_elapsed_sec": sixs_mean_total_sec,
                **sixs_stats,
            }
        )
    except Exception as exc:
        sixs_baseline_available = False
        raw_timings["6S_cpu_error"] = {"error": str(exc)}
        _progress(f"6S baseline skipped due to error: {exc}")

    pipeline_runners = _build_pipelines(
        mode_runs=mode_runs,
        infer_df=sampled,
        models=args.models,
        split_mode=args.split_mode,
        devices=args.devices,
        rho1=args.rho1,
        rho2=args.rho2,
        use_predefined_atm_profile=bool(args.use_predefined_atm_profile),
        oracle_runtime_behavior=args.oracle_runtime_behavior,
    )
    _progress(f"pipeline runner build finished ({len(pipeline_runners)} runnable combinations)")

    total_jobs = max(1, len(pipeline_runners) * max(1, len(args.batch_sizes)))
    job_idx = 0
    for runner in pipeline_runners:
        for bs in args.batch_sizes:
            job_idx += 1
            _progress(
                f"pipeline timing {job_idx}/{total_jobs} started: "
                f"{runner.fidelity_mode} | {runner.architecture_tag} | {runner.model_family} | "
                f"{runner.target_device} | batch={bs}"
            )
            fn = lambda runner=runner, bs=bs: runner.run_full(bs)
            sync_device = None
            if runner.target_device.startswith("cuda"):
                sync_device = choose_torch_device(runner.target_device)
            timing = benchmark_callable(
                fn=fn,
                n_samples=len(sampled),
                repeats=args.repeats,
                warmup_runs=args.warmup_runs,
                sync_device=sync_device,
            )
            key = f"{runner.method_name}__bs{bs}"
            raw_timings[key] = {
                "repeat_total_seconds": timing.repeat_total_seconds,
                "sample_latencies_ms": timing.sample_latencies_ms,
            }
            results.append(
                {
                    "method_name": runner.method_name,
                    "model_family": runner.model_family,
                    "fidelity_mode": runner.fidelity_mode,
                    "architecture_tag": runner.architecture_tag,
                    "split_mode": runner.split_mode,
                    "target_device": runner.target_device,
                    "batch_size": int(bs),
                    "num_samples": int(len(sampled)),
                    "num_repeats": int(args.repeats),
                    "warmup_runs": int(args.warmup_runs),
                    "mean_latency_ms_per_sample": timing.mean_latency_ms_per_sample,
                    "std_latency_ms_per_sample": timing.std_latency_ms_per_sample,
                    "median_latency_ms_per_sample": timing.median_latency_ms_per_sample,
                    "p95_latency_ms_per_sample": timing.p95_latency_ms_per_sample,
                    "throughput_samples_per_sec": timing.throughput_samples_per_sec,
                    "mean_total_elapsed_sec": timing.mean_total_elapsed_sec,
                    "parameter_count": runner.parameter_count,
                    "model_size_mb": runner.model_size_mb,
                }
            )
            _progress(
                f"pipeline timing {job_idx}/{total_jobs} finished: "
                f"{runner.method_name} (batch={bs})"
            )

    results_df = pd.DataFrame(results)
    baseline_lrt_ms = float(results_df.loc[results_df["method_name"] == "libRadtran_cpu", "mean_latency_ms_per_sample"].iloc[0])
    baseline_6s_ms = None
    if sixs_baseline_available and (results_df["method_name"] == "6S_cpu").any():
        baseline_6s_ms = float(results_df.loc[results_df["method_name"] == "6S_cpu", "mean_latency_ms_per_sample"].iloc[0])

    results_df["baseline_libradtran_latency_ms"] = baseline_lrt_ms
    results_df["baseline_6s_latency_ms"] = baseline_6s_ms
    results_df["speedup_vs_libradtran"] = baseline_lrt_ms / results_df["mean_latency_ms_per_sample"].clip(lower=1e-12)
    if baseline_6s_ms is not None:
        results_df["speedup_vs_6s"] = baseline_6s_ms / results_df["mean_latency_ms_per_sample"].clip(lower=1e-12)
    else:
        results_df["speedup_vs_6s"] = np.nan

    srf_df = None
    if args.include_srf_sensitivity:
        _progress("SRF timing sensitivity benchmark started")
        subset = sampled.head(min(args.srf_samples, len(sampled))).copy()
        if not subset.empty:
            no_srf_lrt = baseline_lrt_ms
            no_srf_6s = baseline_6s_ms if baseline_6s_ms is not None else float("nan")

            lrt_srf_lat = []
            sixs_srf_lat = []
            for _, row in subset.iterrows():
                band = str(row["band"])
                wvl, rsp = load_srf(band)
                support = [(lam, r) for lam, r in zip(wvl, rsp) if r > 0.0]
                if not support:
                    continue
                st = _state_from_row(row)

                t0 = time.perf_counter()
                wvls, rho = get_rho_toa_spectrum(st, (float(min(l for l, _ in support)), float(max(l for l, _ in support))), 0.0, debug_log=None)
                _ = np.interp(np.asarray([l for l, _ in support], dtype=float), np.asarray(wvls, dtype=float), np.asarray(rho, dtype=float))
                lrt_srf_lat.append((time.perf_counter() - t0) * 1000.0)

                try:
                    state6 = _state_dict_for_6s(row)
                    t1 = time.perf_counter()
                    out_r = []
                    out_w = []
                    for lam, r in support:
                        ref, _ = run_6s_point(
                            state=state6,
                            wavelength_nm=float(lam),
                            albedo=0.0,
                            use_predefined_atm_profile=bool(args.use_predefined_atm_profile),
                        )
                        out_w.append(float(lam))
                        out_r.append(ref)
                    _ = np.interp(np.asarray([l for l, _ in support], dtype=float), np.asarray(out_w, dtype=float), np.asarray(out_r, dtype=float))
                    sixs_srf_lat.append((time.perf_counter() - t1) * 1000.0)
                except Exception:
                    pass

            if lrt_srf_lat:
                lrt_srf_ms = float(np.mean(lrt_srf_lat))
                rows = [
                    {
                        "method": "libRadtran",
                        "no_srf_ms": no_srf_lrt,
                        "srf_ms": lrt_srf_ms,
                        "increase_pct": ((lrt_srf_ms - no_srf_lrt) / max(no_srf_lrt, 1e-12)) * 100.0,
                    }
                ]
                if sixs_srf_lat and baseline_6s_ms is not None and math.isfinite(no_srf_6s):
                    sixs_srf_ms = float(np.mean(sixs_srf_lat))
                    rows.append(
                        {
                            "method": "6S",
                            "no_srf_ms": no_srf_6s,
                            "srf_ms": sixs_srf_ms,
                            "increase_pct": ((sixs_srf_ms - no_srf_6s) / max(no_srf_6s, 1e-12)) * 100.0,
                        }
                    )
                srf_df = pd.DataFrame(rows)
                srf_df.to_csv(out_dir / "rtm_srf_sensitivity.csv", index=False)
        _progress("SRF timing sensitivity benchmark finished")

    results_df = results_df.sort_values(["model_family", "method_name", "batch_size", "mean_latency_ms_per_sample"]).reset_index(drop=True)
    results_all_df = results_df.copy()
    results_batch1_df = results_all_df[results_all_df["batch_size"] == 1].copy()
    throughput_source = results_all_df[(results_all_df["model_family"] != "rtm") & (results_all_df["batch_size"] > 1)].copy()
    warnings: List[str] = []
    if throughput_source.empty:
        throughput_source = results_all_df[results_all_df["model_family"] != "rtm"].copy()
        warnings.append("No batch_size>1 rows found; throughput table falls back to available rows.")
    if not throughput_source.empty:
        idx = throughput_source.groupby(["method_name"])["throughput_samples_per_sec"].idxmax()
        results_throughput_df = throughput_source.loc[idx].sort_values("throughput_samples_per_sec", ascending=False).reset_index(drop=True)
    else:
        results_throughput_df = pd.DataFrame(columns=list(results_all_df.columns))

    cpu_batch1_non_rtm = results_batch1_df[(results_batch1_df["target_device"] == "cpu") & (results_batch1_df["model_family"] != "rtm")]
    if not cpu_batch1_non_rtm.empty and float(cpu_batch1_non_rtm["mean_latency_ms_per_sample"].min()) < 0.05:
        warnings.append(
            "CPU batch_size=1 latency below 0.05 ms detected; values may be unrealistic due to timer/amortization artifacts."
        )
        _progress("warning: CPU batch_size=1 latency below 0.05 ms detected; interpret with caution")

    results_all_csv = out_dir / "benchmark_results_all.csv"
    results_batch1_csv = out_dir / "benchmark_results_batch1.csv"
    results_throughput_csv = out_dir / "benchmark_results_throughput.csv"
    # Backward-compatible path (full results)
    results_csv = out_dir / "benchmark_results.csv"
    results_all_df.to_csv(results_all_csv, index=False)
    results_batch1_df.to_csv(results_batch1_csv, index=False)
    results_throughput_df.to_csv(results_throughput_csv, index=False)
    results_all_df.to_csv(results_csv, index=False)

    report_md = _build_markdown_report(
        machine_info=machine_info,
        results_all_df=results_all_df,
        batch1_df=results_batch1_df,
        throughput_df=results_throughput_df,
        rtm_srf_df=srf_df,
        warnings=warnings,
    )
    (out_dir / "benchmark_results.md").write_text(report_md, encoding="utf-8")
    save_json(out_dir / "machine_info.json", machine_info)
    (out_dir / "machine_info.md").write_text(
        "# Machine Info\n\n```json\n" + json.dumps(machine_info, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    save_json(out_dir / "raw_timings.json", raw_timings)

    _write_plots(results_df=results_all_df, batch1_df=results_batch1_df, throughput_df=results_throughput_df, out_dir=plots_dir)
    _progress("plots generation finished")

    print("Runtime benchmark complete.")
    print(f"Samples: {len(sampled)}, repeats: {args.repeats}, warmup: {args.warmup_runs}")
    print(f"Saved all results CSV: {results_all_csv}")
    print(f"Saved batch_size=1 latency CSV: {results_batch1_csv}")
    print(f"Saved throughput CSV: {results_throughput_csv}")
    print(f"Saved markdown report: {out_dir / 'benchmark_results.md'}")
    print(f"Saved machine info: {out_dir / 'machine_info.json'}")
    print(f"Saved raw timings: {out_dir / 'raw_timings.json'}")


if __name__ == "__main__":
    main()
