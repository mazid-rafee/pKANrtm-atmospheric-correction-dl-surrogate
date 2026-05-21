#!/usr/bin/env python3
"""
Generate paper-ready diagnostic figures for atmospheric correction surrogate modeling.

This script is robust to multiple dataset layouts:
1) Single table with explicit 6S/libRadtran suffix columns.
2) Separate 6S and libRadtran tables merged on shared keys.
3) Optional prediction tables (or auto-detected run outputs).

Outputs are saved under:
    figures/paper_diagnostics/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


COEFS = ["rho_path", "t_total", "spher_alb"]
COEF_LABEL = {"rho_path": "rho_path", "t_total": "T_total", "spher_alb": "spher_alb"}
PRED_COL_ALIASES = {
    "rho_path": ["pred_rho_path"],
    "t_total": ["pred_t_total", "pred_T_total"],
    "spher_alb": ["pred_spher_alb"],
}
STATE_HISTOGRAM_VARS = ["aod550", "cwv_cm", "o3_cm", "sza_deg", "vza_deg", "raa_deg", "elev_km"]
FEATURE_CANDIDATES = [
    "band",
    "wvl_nm",
    "sza_deg",
    "vza_deg",
    "raa_deg",
    "aod550",
    "cwv_cm",
    "o3_cm",
    "elev_km",
    "aerosol_type",
    "atm_profile",
]
MERGE_KEYS = [
    "state_id",
    "band",
    "wvl_nm",
    "sza_deg",
    "vza_deg",
    "raa_deg",
    "aod550",
    "cwv_cm",
    "o3_cm",
    "elev_km",
    "aerosol_type",
    "atm_profile",
]
BAND_ORDER = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".jsonl", ".json"}:
        if suffix == ".jsonl":
            return pd.read_json(path, lines=True)
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            return pd.DataFrame(obj)
        if isinstance(obj, dict):
            return pd.DataFrame(obj)
        raise ValueError(f"Unsupported JSON structure in {path}")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file extension: {path}")


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "wavelength_nm": "wvl_nm",
        "wavelength": "wvl_nm",
        "sza": "sza_deg",
        "vza": "vza_deg",
        "raa": "raa_deg",
        "aod_550": "aod550",
        "aod_550nm": "aod550",
        "cwv": "cwv_cm",
        "ozone_cm": "o3_cm",
        "elevation_km": "elev_km",
        "profile": "atm_profile",
        "state": "state_id",
        "sample_id": "state_id",
    }
    cols = {}
    for c in df.columns:
        low = str(c).strip().lower()
        cols[c] = aliases.get(low, low)
    return df.rename(columns=cols)


def find_coefficient_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    cols = set(df.columns)
    for c in COEFS:
        label = COEF_LABEL[c]
        if f"{c}_6s" in cols and f"{c}_lrt" in cols:
            out[label] = f"{c}_6s,{c}_lrt"
        elif c in cols:
            out[label] = c
        elif f"{c}_raw" in cols:
            out[label] = f"{c}_raw"
        else:
            out[label] = None
    return out


def _resolve_project_root() -> Path:
    """Return pKANrtm root when script lives under Atmospheric Correction/scripts/."""
    root = Path(__file__).resolve().parents[1]
    pkan = root / "pKANrtm"
    if pkan.is_dir() and not (root / "data").is_dir() and (pkan / "data").is_dir():
        return pkan
    return root


def detect_default_paths(root: Path) -> Dict[str, Optional[Path]]:
    cand_6s = [
        root / "data/qavalid_intersection_libradtran_6s_50k_13b/dataset_rows_6s.jsonl",
        root / "data/generated_6s_50k_13b/dataset_rows.jsonl",
    ]
    cand_lrt = [
        root / "data/qavalid_intersection_libradtran_6s_50k_13b/dataset_rows_libradtran.jsonl",
        root / "data/generated_libradtran_50k_13b/dataset_rows.jsonl",
    ]
    cand_pred = [
        root / "benchmarks/inference_error_benchmark/predictions.csv",
        root / "runs/mf_parallel_2gpu_allbands/final_comparison.csv",
    ]
    cand_run_roots = [
        root / "runs/fig05_surrogate_pkan",
        root / "runs/mf_parallel_4gpu_oracle_kan_pkan",
        root / "runs/mf_parallel_2gpu_allbands",
        root / "runs/mf_oracle_kan_pkan",
    ]
    run_root = next((p for p in cand_run_roots if p.is_dir()), cand_run_roots[0])
    final_comp = run_root / "final_comparison.csv"
    return {
        "data_6s": next((p for p in cand_6s if p.exists()), None),
        "data_lrt": next((p for p in cand_lrt if p.exists()), None),
        "predictions": next((p for p in cand_pred if p.exists()), None),
        "run_root": run_root,
        "final_comparison": final_comp if final_comp.exists() else None,
        "accuracy_md": root / "results/inference_accuracy_benchmark.md",
    }


def _pick_merge_keys(df_6s: pd.DataFrame, df_lrt: pd.DataFrame) -> List[str]:
    keys = [k for k in MERGE_KEYS if k in df_6s.columns and k in df_lrt.columns]
    if "state_id" in keys and "band" in keys:
        return ["state_id", "band"]
    if not keys:
        raise ValueError("Could not infer merge keys between 6S and libRadtran tables.")
    return keys


def prepare_paired_coefficients(df_data: pd.DataFrame, df_pred: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    # For compatibility with requested helper signature; returns normalized dataframe.
    _ = df_pred
    return normalize_column_names(df_data.copy())


def savefig(fig: plt.Figure, output_dir: Path, filename_stem: str) -> List[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{filename_stem}.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [str(png)]


def build_paired_dataframe(
    data_path: Optional[Path],
    data_6s_path: Optional[Path],
    data_lrt_path: Optional[Path],
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Returns a row-level paired dataframe with:
      coef_6s, coef_lrt columns for each coefficient.
    """
    meta: Dict[str, str] = {}

    if data_path is not None:
        df = normalize_column_names(load_table(data_path))
        cols = set(df.columns)
        needed = [f"{c}_6s" for c in COEFS] + [f"{c}_lrt" for c in COEFS]
        if all(c in cols for c in needed):
            meta["paired_mode"] = "single_table_suffixes"
            return df, meta
        raise ValueError(
            "Single --data file was provided but does not contain explicit *_6s and *_lrt columns. "
            "Use --data_6s and --data_lrt instead."
        )

    if data_6s_path is None or data_lrt_path is None:
        raise ValueError("Need either --data (with suffix cols) OR both --data_6s and --data_lrt.")

    df_6s = normalize_column_names(load_table(data_6s_path))
    df_lrt = normalize_column_names(load_table(data_lrt_path))
    if "qa_valid" in df_6s.columns:
        df_6s = df_6s[df_6s["qa_valid"].astype(bool)].copy()
    if "qa_valid" in df_lrt.columns:
        df_lrt = df_lrt[df_lrt["qa_valid"].astype(bool)].copy()

    # Keep only necessary feature/coef columns for stable merge.
    keep_6s = [c for c in FEATURE_CANDIDATES + ["state_id"] + COEFS if c in df_6s.columns]
    keep_lrt = [c for c in FEATURE_CANDIDATES + ["state_id"] + COEFS if c in df_lrt.columns]
    df_6s = df_6s[keep_6s].copy()
    df_lrt = df_lrt[keep_lrt].copy()

    keys = _pick_merge_keys(df_6s, df_lrt)
    merged = df_6s.merge(df_lrt, on=keys, how="inner", suffixes=("_6s", "_lrt")).reset_index(drop=True)
    if merged.empty:
        raise ValueError("Merging 6S and libRadtran produced empty dataframe.")
    # Restore unsuffixed feature columns from libRadtran side first (fallback to 6S).
    for feat in FEATURE_CANDIDATES + ["state_id"]:
        if feat in merged.columns:
            continue
        if f"{feat}_lrt" in merged.columns:
            merged[feat] = merged[f"{feat}_lrt"]
        elif f"{feat}_6s" in merged.columns:
            merged[feat] = merged[f"{feat}_6s"]
    meta["paired_mode"] = "merged_6s_lrt"
    meta["merge_keys"] = ",".join(keys)
    return merged, meta


def _band_sorter(df: pd.DataFrame) -> List[str]:
    if "band" not in df.columns:
        return []
    present = sorted(df["band"].dropna().astype(str).unique().tolist(), key=lambda b: BAND_ORDER.index(b) if b in BAND_ORDER else 999)
    return present


def _safe_qcut(values: pd.Series, n_bins: int = 6) -> pd.Series:
    v = pd.to_numeric(values, errors="coerce")
    v = v[~v.isna()]
    if v.nunique() < 2:
        return pd.Series(["all"] * len(values), index=values.index)
    try:
        bins = pd.qcut(values, q=min(n_bins, int(v.nunique())), duplicates="drop")
        return bins
    except Exception:
        return pd.cut(values, bins=min(n_bins, int(v.nunique())), duplicates="drop")


def _fmt_float_short(x: float) -> str:
    if not np.isfinite(x):
        return "nan"
    ax = abs(float(x))
    if ax == 0:
        return "0"
    if ax < 1e-3 or ax >= 1e4:
        return f"{x:.2e}"
    return f"{x:.4g}"


def _format_bin_label(v) -> str:
    if isinstance(v, pd.Interval):
        left = _fmt_float_short(float(v.left))
        right = _fmt_float_short(float(v.right))
        return f"[{left}, {right}]"
    return str(v)


def _parse_md_tables(md_path: Path) -> List[pd.DataFrame]:
    lines = md_path.read_text(encoding="utf-8").splitlines()
    tables: List[pd.DataFrame] = []
    i = 0
    while i < len(lines) - 1:
        line = lines[i].strip()
        line2 = lines[i + 1].strip()
        if line.startswith("|") and line.endswith("|") and line2.startswith("|") and ("---" in line2 or ":---" in line2):
            headers = [c.strip() for c in line.strip("|").split("|")]
            rows: List[List[str]] = []
            j = i + 2
            while j < len(lines):
                row = lines[j].strip()
                if not (row.startswith("|") and row.endswith("|")):
                    break
                vals = [c.strip() for c in row.strip("|").split("|")]
                if len(vals) == len(headers):
                    rows.append(vals)
                j += 1
            tables.append(pd.DataFrame(rows, columns=headers))
            i = j
        else:
            i += 1
    return tables


def _load_overall_accuracy_md(md_path: Path) -> pd.DataFrame:
    want = {"fidelity_mode", "split_mode", "model", "rmse", "variant_tag", "run_subfolder"}
    chosen = None
    for table in _parse_md_tables(md_path):
        if not want.issubset(set(table.columns)) or "band" in table.columns:
            continue
        if chosen is None or abs(len(table) - 20) < abs(len(chosen) - 20):
            chosen = table
    if chosen is None:
        raise ValueError(f"Could not find overall Accuracy table in {md_path}")
    out = chosen.copy()
    out["rmse"] = pd.to_numeric(out["rmse"], errors="coerce")
    return out


def _select_best_from_accuracy_md(md_path: Path, split_mode: str) -> Optional[dict]:
    if not md_path.exists():
        return None
    sub = _load_overall_accuracy_md(md_path)
    sub = sub[(sub["fidelity_mode"] == "oracle_residual") & (sub["split_mode"] == split_mode)].copy()
    sub = sub.dropna(subset=["rmse"])
    if sub.empty:
        return None
    row = sub.sort_values("rmse").iloc[0]
    return row.to_dict()


def _select_best_models(final_comp_path: Optional[Path], split_mode: str) -> Dict[str, Optional[dict]]:
    out = {"best_kan": None, "best_mlp": None, "best_overall": None}
    if final_comp_path is None or not final_comp_path.exists():
        return out
    df = pd.read_csv(final_comp_path)
    df = normalize_column_names(df)
    needed = {"fidelity_mode", "split_mode", "model", "rmse"}
    if not needed.issubset(df.columns):
        return out
    sub = df[(df["fidelity_mode"] == "oracle_residual") & (df["split_mode"] == split_mode)].copy()
    if sub.empty:
        return out
    sub["rmse"] = pd.to_numeric(sub["rmse"], errors="coerce")
    sub = sub.dropna(subset=["rmse"])
    if sub.empty:
        return out
    out["best_overall"] = sub.sort_values("rmse").iloc[0].to_dict()
    kan_rows = sub[sub["model"].astype(str).str.lower() == "kan"]
    if not kan_rows.empty:
        out["best_kan"] = kan_rows.sort_values("rmse").iloc[0].to_dict()
    mlp_rows = sub[sub["model"].astype(str).str.lower() == "mlp"]
    if not mlp_rows.empty:
        out["best_mlp"] = mlp_rows.sort_values("rmse").iloc[0].to_dict()
    return out


def _pred_column(df: pd.DataFrame, coef: str) -> Optional[str]:
    for name in PRED_COL_ALIASES.get(coef, [f"pred_{coef}"]):
        if name in df.columns:
            return name
    return None


def _find_prediction_csv(
    run_root: Path,
    variant_tag: str,
    split_mode: str,
    model_name: str,
    run_subfolder: Optional[str] = None,
) -> Optional[Path]:
    split_folder = "standard" if split_mode == "standard" else "ood_aod_cwv"
    if run_subfolder is None:
        run_subfolder = "oracle_standard_gpu3" if split_mode == "standard" else "oracle_ood_gpu5"
    p = (
        run_root
        / run_subfolder
        / variant_tag
        / "oracle_residual"
        / "splits"
        / split_folder
        / "models"
        / model_name
        / "test_predictions.csv"
    )
    return p if p.exists() else None


def _subset_test_rows_for_prediction(paired: pd.DataFrame, pred_csv: Path) -> Optional[pd.DataFrame]:
    if "state_id" not in paired.columns or "band" not in paired.columns:
        return None
    split_assign = pred_csv.parents[2] / "split_assignment.csv"
    if not split_assign.exists():
        return None
    sa = pd.read_csv(split_assign)
    if not {"state_id", "split"}.issubset(sa.columns):
        return None
    test_states = set(sa.loc[sa["split"] == "test", "state_id"].astype(str))
    sub = paired[paired["state_id"].astype(str).isin(test_states)].copy()
    pred = pd.read_csv(pred_csv)
    if len(pred) != len(sub):
        return None
    for c in COEFS:
        pc = _pred_column(pred, c)
        if pc is None:
            return None
        sub[f"pred_{c}"] = pd.to_numeric(pred[pc], errors="coerce")
    return sub.reset_index(drop=True)


def _resolve_model_split_dir(run_root: Path, model_row: Optional[dict], split_mode: str) -> Optional[Path]:
    split_folder = "standard" if split_mode == "standard" else "ood_aod_cwv"
    candidates: List[Path] = []
    if model_row is not None:
        rs = model_row.get("run_subfolder")
        vt = model_row.get("variant_tag")
        if rs is not None and vt is not None and str(rs) not in {"", "nan"} and str(vt) not in {"", "nan"}:
            candidates.append(
                run_root / str(rs) / str(vt) / "oracle_residual" / "splits" / split_folder
            )
    candidates.append(run_root / "oracle_residual" / "splits" / split_folder)
    for path in candidates:
        if (path / "preprocess" / "preprocessor.joblib").exists():
            return path
    return None


def _load_model_build_kwargs(split_dir: Path) -> Dict:
    defaults = {
        "mlp_arch": "baseline",
        "kan_arch": "baseline",
        "activation": "relu",
        "use_layernorm": False,
        "deployable_stage_sizing": False,
    }
    for base in [split_dir, *list(split_dir.parents)[:5]]:
        cfg = base / "run_config.json"
        if not cfg.exists():
            continue
        payload = json.loads(cfg.read_text(encoding="utf-8"))
        defaults["mlp_arch"] = payload.get("mlp_arch", defaults["mlp_arch"])
        defaults["kan_arch"] = payload.get("kan_arch", defaults["kan_arch"])
        defaults["activation"] = payload.get("activation", defaults["activation"])
        defaults["use_layernorm"] = bool(payload.get("use_layernorm", defaults["use_layernorm"]))
        defaults["deployable_stage_sizing"] = bool(
            payload.get("deployable_stage_sizing", defaults["deployable_stage_sizing"])
        )
        break
    return defaults


def _infer_surrogate_predictions(
    paired: pd.DataFrame,
    split_dir: Path,
    model_name: str,
    batch_size: int = 8192,
    device_name: str = "cpu",
) -> pd.DataFrame:
    import joblib
    import torch
    from surrogate_pipeline.models import build_model

    preproc_path = split_dir / "preprocess" / "preprocessor.joblib"
    scaler_path = split_dir / "preprocess" / "target_scaler.joblib"
    model_dir = split_dir / "models" / model_name
    ckpt_path = model_dir / "best.pt"
    if not (preproc_path.exists() and scaler_path.exists() and ckpt_path.exists()):
        raise FileNotFoundError(
            f"Missing inference artifacts under {split_dir} for model={model_name}. "
            f"Need preprocessor.joblib, target_scaler.joblib, and {ckpt_path}."
        )

    preprocessor = joblib.load(preproc_path)
    y_scaler = joblib.load(scaler_path)
    feature_names = list(preprocessor.feature_names_in_)
    infer_df = paired.copy()
    required = list(preprocessor.feature_names_in_)
    missing = [c for c in required if c not in infer_df.columns]
    if missing:
        raise ValueError(f"Inference dataframe missing required feature columns: {missing}")
    x = preprocessor.transform(infer_df[required])
    if hasattr(x, "toarray"):
        x = x.toarray()
    x = np.asarray(x, dtype=np.float32)

    device = torch.device("cuda:0" if device_name != "cpu" and torch.cuda.is_available() else "cpu")
    kwargs = _load_model_build_kwargs(split_dir)
    model, _ = build_model(
        model_name,
        in_dim=x.shape[1],
        out_dim=len(COEFS),
        stage_name="single_stage",
        mlp_arch=kwargs["mlp_arch"],
        kan_arch=kwargs["kan_arch"],
        activation=kwargs["activation"],
        use_layernorm=kwargs["use_layernorm"],
        deployable_stage_sizing=kwargs["deployable_stage_sizing"],
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    chunks = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            xb = torch.from_numpy(x[i : i + batch_size]).float().to(device)
            chunks.append(model(xb).detach().cpu().numpy())
    pred_scaled = np.vstack(chunks).astype(np.float32)
    pred_raw = y_scaler.inverse_transform(pred_scaled).astype(np.float32)

    out = infer_df[["state_id", "band"] + [f"{c}_lrt" for c in COEFS]].copy()
    for i, c in enumerate(COEFS):
        out[f"pred_{c}"] = pred_raw[:, i]
    return out


def _load_surrogate_predictions(
    paired: pd.DataFrame,
    run_root: Path,
    model_row: dict,
    split_mode: str,
    infer_device: str = "cpu",
) -> Tuple[Optional[pd.DataFrame], str]:
    model_name = str(model_row.get("model", "surrogate"))
    label = f"Surrogate ({model_name})"
    variant_tag = str(model_row.get("variant_tag", ""))
    run_subfolder = model_row.get("run_subfolder")
    run_subfolder = str(run_subfolder) if run_subfolder is not None and str(run_subfolder) not in {"", "nan"} else None

    pred_csv = _find_prediction_csv(run_root, variant_tag, split_mode, model_name, run_subfolder=run_subfolder)
    if pred_csv is not None:
        merged = _subset_test_rows_for_prediction(paired, pred_csv)
        if merged is not None:
            return merged, label

    split_dir = _resolve_model_split_dir(run_root, model_row, split_mode)
    if split_dir is None:
        return None, label
    try:
        return _infer_surrogate_predictions(paired, split_dir, model_name, device_name=infer_device), label
    except Exception:
        return None, label


def fig_01_state_space_coverage(df: pd.DataFrame, out_dir: Path) -> List[str]:
    numeric = STATE_HISTOGRAM_VARS + ["wvl_nm"]
    present_num = [c for c in numeric if c in df.columns]
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), constrained_layout=True)
    axes = axes.flatten()

    for i, c in enumerate(present_num[:8]):
        ax = axes[i]
        vals = pd.to_numeric(df[c], errors="coerce").dropna()
        use_density = c in STATE_HISTOGRAM_VARS
        _, _, patches = ax.hist(vals, bins=50, alpha=0.85, color="#1f77b4", density=use_density)
        ax.set_title(COEF_LABEL.get(c, c))
        if use_density:
            ax.set_ylabel("Normalized density")
            heights = [p.get_height() for p in patches if p.get_height() > 0]
            if heights:
                h_min, h_max = min(heights), max(heights)
                span = h_max - h_min
                if span > 0:
                    pad = max(span * 0.12, h_max * 0.02)
                    ax.set_ylim(h_min - pad, h_max + pad)
                else:
                    ax.set_ylim(0, h_max * 1.08)
        ax.grid(alpha=0.25)
    if "band" in df.columns:
        ax = axes[8]
        counts = df["band"].astype(str).value_counts().reindex(_band_sorter(df)).dropna()
        ax.bar(counts.index, counts.values, color="#2ca02c", alpha=0.85)
        ax.set_title("band")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.25)
    return savefig(fig, out_dir, "figure_01_state_space_coverage")


def fig_02_discrepancy(df: pd.DataFrame, out_dir: Path) -> List[str]:
    if "band" not in df.columns:
        raise ValueError("Need 'band' column for discrepancy plot.")
    bands = _band_sorter(df)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for i, c in enumerate(COEFS):
        c6, cl = f"{c}_6s", f"{c}_lrt"
        if c6 not in df.columns or cl not in df.columns:
            raise ValueError(f"Missing columns {c6}/{cl}")
        tmp = df[["band", c6, cl]].copy()
        tmp["abs_diff"] = (pd.to_numeric(tmp[cl], errors="coerce") - pd.to_numeric(tmp[c6], errors="coerce")).abs()
        grp = tmp.groupby("band")["abs_diff"]
        mean_s = grp.mean().reindex(bands)
        med_s = grp.median().reindex(bands)
        q25 = grp.quantile(0.25).reindex(bands)
        q75 = grp.quantile(0.75).reindex(bands)
        x = np.arange(len(bands))
        ax = axes[i]
        ax.plot(x, mean_s.values, marker="o", linewidth=1.8, label="mean |lrt-6s|")
        ax.plot(x, med_s.values, marker="s", linewidth=1.5, label="median |lrt-6s|")
        ax.fill_between(x, q25.values, q75.values, alpha=0.2, label="IQR")
        ax.set_xticks(x)
        ax.set_xticklabels(bands, rotation=45)
        ax.set_title(c)
        ax.set_xlabel("Band")
        ax.set_ylabel("|libRadtran - 6S|")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    return savefig(fig, out_dir, "figure_02_6s_libradtran_discrepancy")


def _band_stats(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce").groupby(df["band"]).median().reindex(_band_sorter(df))


def _band_abs_err(df: pd.DataFrame, pred_col: str, true_col: str, q: Optional[float] = None) -> pd.Series:
    err = (pd.to_numeric(df[pred_col], errors="coerce") - pd.to_numeric(df[true_col], errors="coerce")).abs()
    grp = err.groupby(df["band"])
    if q is None:
        out = grp.median()
    else:
        out = grp.quantile(q)
    return out.reindex(_band_sorter(df))


def fig_03_srtmnet_panel(
    df: pd.DataFrame,
    out_dir: Path,
    pred_best_kan: Optional[pd.DataFrame],
    pred_best_mlp: Optional[pd.DataFrame],
) -> List[str]:
    if "band" not in df.columns:
        raise ValueError("Need band column.")
    bands = _band_sorter(df)
    x = np.arange(len(bands))
    fig, axes = plt.subplots(3, 3, figsize=(16, 11), constrained_layout=False)
    fig.subplots_adjust(top=0.86, bottom=0.10, wspace=0.25, hspace=0.30)

    for j, c in enumerate(["t_total", "rho_path", "spher_alb"]):
        c6, cl = f"{c}_6s", f"{c}_lrt"
        # Row 1: median modeled output by band
        ax = axes[0, j]
        ax.plot(x, _band_stats(df, cl).values, color="black", linewidth=2.2, label="libRadtran true")
        ax.plot(x, _band_stats(df, c6).values, color="gray", linewidth=1.8, linestyle="--", label="6S")
        if pred_best_kan is not None and f"pred_{c}" in pred_best_kan.columns:
            ax.plot(x, _band_stats(pred_best_kan, f"pred_{c}").values, linewidth=1.8, label="Best KAN pred")
        if pred_best_mlp is not None and f"pred_{c}" in pred_best_mlp.columns:
            ax.plot(x, _band_stats(pred_best_mlp, f"pred_{c}").values, linewidth=1.8, label="Best MLP pred")
        ax.set_title(c)
        ax.set_xticks(x)
        ax.set_xticklabels(bands, rotation=45)
        ax.set_ylabel("Median output")
        ax.grid(alpha=0.25)

        # Row 2: median absolute residual
        ax = axes[1, j]
        ax.plot(x, _band_abs_err(df, c6, cl).values, color="gray", linestyle="--", linewidth=1.8, label="|6S-lrt| median")
        if pred_best_kan is not None and f"pred_{c}" in pred_best_kan.columns:
            ax.plot(x, _band_abs_err(pred_best_kan, f"pred_{c}", cl).values, linewidth=1.8, label="|KAN-lrt| median")
        if pred_best_mlp is not None and f"pred_{c}" in pred_best_mlp.columns:
            ax.plot(x, _band_abs_err(pred_best_mlp, f"pred_{c}", cl).values, linewidth=1.8, label="|MLP-lrt| median")
        ax.set_xticks(x)
        ax.set_xticklabels(bands, rotation=45)
        ax.set_ylabel("Median |residual|")
        ax.grid(alpha=0.25)

        # Row 3: 95th percentile absolute residual
        ax = axes[2, j]
        ax.plot(x, _band_abs_err(df, c6, cl, q=0.95).values, color="gray", linestyle="--", linewidth=1.8, label="|6S-lrt| p95")
        if pred_best_kan is not None and f"pred_{c}" in pred_best_kan.columns:
            ax.plot(x, _band_abs_err(pred_best_kan, f"pred_{c}", cl, q=0.95).values, linewidth=1.8, label="|KAN-lrt| p95")
        if pred_best_mlp is not None and f"pred_{c}" in pred_best_mlp.columns:
            ax.plot(x, _band_abs_err(pred_best_mlp, f"pred_{c}", cl, q=0.95).values, linewidth=1.8, label="|MLP-lrt| p95")
        ax.set_xticks(x)
        ax.set_xticklabels(bands, rotation=45)
        ax.set_ylabel("P95 |residual|")
        ax.grid(alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=4, frameon=True)
    return savefig(fig, out_dir, "figure_03_srtmnet_style_coefficients_residuals")


def _heatmap_panel(df_err: pd.DataFrame, coeff: str, cond: str, ax: plt.Axes) -> None:
    d = df_err[[cond, "band", coeff]].dropna().copy()
    d["cond_bin"] = _safe_qcut(d[cond], n_bins=6)
    piv = d.pivot_table(index="cond_bin", columns="band", values=coeff, aggfunc="median")
    piv = piv.reindex(columns=[b for b in BAND_ORDER if b in piv.columns])
    im = ax.imshow(piv.values, aspect="auto")
    ax.set_xticks(np.arange(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=45)
    ax.set_yticks(np.arange(len(piv.index)))
    ax.set_yticklabels([_format_bin_label(v) for v in piv.index])
    ax.set_title(f"{coeff} | {cond}")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def fig_04_heatmaps(
    df: pd.DataFrame,
    out_dir: Path,
    pred_best: Optional[pd.DataFrame],
) -> List[str]:
    generated: List[str] = []
    cond_vars = [c for c in ["aod550", "cwv_cm", "sza_deg", "elev_km"] if c in df.columns]
    if not cond_vars:
        raise ValueError("No conditioning variables available.")

    # 6S baseline
    base = df.copy()
    for c in COEFS:
        base[c] = (pd.to_numeric(base[f"{c}_6s"], errors="coerce") - pd.to_numeric(base[f"{c}_lrt"], errors="coerce")).abs()
    fig, axes = plt.subplots(len(cond_vars), len(COEFS), figsize=(15, 3.2 * len(cond_vars)), constrained_layout=True)
    if len(cond_vars) == 1:
        axes = np.array([axes])
    for i, cond in enumerate(cond_vars):
        for j, c in enumerate(COEFS):
            _heatmap_panel(base, c, cond, axes[i, j])
    generated += savefig(fig, out_dir, "figure_04b_conditional_error_heatmap_6s_baseline")

    if pred_best is not None and all(f"pred_{c}" in pred_best.columns for c in COEFS):
        best = pred_best.copy()
        for c in COEFS:
            best[c] = (pd.to_numeric(best[f"pred_{c}"], errors="coerce") - pd.to_numeric(best[f"{c}_lrt"], errors="coerce")).abs()
        fig, axes = plt.subplots(len(cond_vars), len(COEFS), figsize=(15, 3.2 * len(cond_vars)), constrained_layout=True)
        if len(cond_vars) == 1:
            axes = np.array([axes])
        for i, cond in enumerate(cond_vars):
            for j, c in enumerate(COEFS):
                _heatmap_panel(best, c, cond, axes[i, j])
        generated += savefig(fig, out_dir, "figure_04a_conditional_error_heatmap_best_model")
    return generated


def _select_representative_states(df: pd.DataFrame, pred_best: Optional[pd.DataFrame]) -> List[Tuple[str, str]]:
    if "state_id" not in df.columns:
        return []
    out: List[Tuple[str, str]] = []
    state_df = df.groupby("state_id", as_index=False).agg(
        aod550=("aod550", "median") if "aod550" in df.columns else ("state_id", "size"),
        cwv_cm=("cwv_cm", "median") if "cwv_cm" in df.columns else ("state_id", "size"),
        sza_deg=("sza_deg", "median") if "sza_deg" in df.columns else ("state_id", "size"),
    )
    if "aod550" in state_df.columns and "cwv_cm" in state_df.columns:
        low = state_df.sort_values(["aod550", "cwv_cm"]).iloc[0]["state_id"]
        out.append(("low AOD/CWV", str(low)))
    if "aod550" in state_df.columns:
        high_aod = state_df.sort_values("aod550", ascending=False).iloc[0]["state_id"]
        out.append(("high AOD", str(high_aod)))
    if "cwv_cm" in state_df.columns:
        high_cwv = state_df.sort_values("cwv_cm", ascending=False).iloc[0]["state_id"]
        out.append(("high CWV", str(high_cwv)))
    if "sza_deg" in state_df.columns:
        high_sza = state_df.sort_values("sza_deg", ascending=False).iloc[0]["state_id"]
        out.append(("high SZA", str(high_sza)))

    # high 6S-lrt discrepancy
    tmp = df.copy()
    tmp["disc"] = 0.0
    for c in COEFS:
        tmp["disc"] += (pd.to_numeric(tmp[f"{c}_6s"], errors="coerce") - pd.to_numeric(tmp[f"{c}_lrt"], errors="coerce")).abs()
    worst_disc = tmp.groupby("state_id")["disc"].mean().sort_values(ascending=False).index[0]
    out.append(("high 6S-lrt discrepancy", str(worst_disc)))

    if pred_best is not None and all(f"pred_{c}" in pred_best.columns for c in COEFS):
        pp = pred_best.copy()
        pp["pred_err"] = 0.0
        for c in COEFS:
            pp["pred_err"] += (pd.to_numeric(pp[f"pred_{c}"], errors="coerce") - pd.to_numeric(pp[f"{c}_lrt"], errors="coerce")).abs()
        worst_pred = pp.groupby("state_id")["pred_err"].mean().sort_values(ascending=False).index[0]
        out.append(("worst prediction error", str(worst_pred)))

    # unique by state_id preserve order
    seen = set()
    uniq = []
    for label, sid in out:
        if sid in seen:
            continue
        seen.add(sid)
        uniq.append((label, sid))
    return uniq[:5]


def fig_05_representative_cases(
    df: pd.DataFrame,
    out_dir: Path,
    pred_best: Optional[pd.DataFrame],
    pred_label: str = "Surrogate",
) -> List[str]:
    if "state_id" not in df.columns or "band" not in df.columns:
        raise ValueError("Need state_id and band for representative case plots.")
    if pred_best is None or not all(f"pred_{c}" in pred_best.columns for c in COEFS):
        raise ValueError(
            "Surrogate predictions are required for figure 05. "
            "Provide --run_root with test_predictions.csv or fix prediction merge."
        )
    cases = _select_representative_states(df, pred_best)
    if not cases:
        raise ValueError("Could not select representative states.")
    bands = [b for b in BAND_ORDER if b in df["band"].astype(str).unique()]
    x = np.arange(len(bands))
    fig, axes = plt.subplots(len(cases), 3, figsize=(15, 3.0 * len(cases)), constrained_layout=False)
    fig.subplots_adjust(top=0.90, bottom=0.11, wspace=0.28, hspace=0.35)
    if len(cases) == 1:
        axes = np.array([axes])
    for i, (label, sid) in enumerate(cases):
        d = df[df["state_id"].astype(str) == sid].copy()
        p = pred_best[pred_best["state_id"].astype(str) == sid].copy()
        for j, c in enumerate(COEFS):
            ax = axes[i, j]
            s_lrt = d.groupby("band")[f"{c}_lrt"].mean().reindex(bands)
            s_6s = d.groupby("band")[f"{c}_6s"].mean().reindex(bands)
            s_p = p.groupby("band")[f"pred_{c}"].mean().reindex(bands)
            ax.plot(x, s_lrt.values, color="black", linewidth=2.0, label="libRadtran")
            ax.plot(x, s_6s.values, color="gray", linestyle="--", linewidth=1.6, label="6S")
            ax.plot(
                x,
                s_p.values,
                color="#d62728",
                linewidth=1.8,
                linestyle="-.",
                label=pred_label,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(bands, rotation=45)
            ax.grid(alpha=0.25)
            if j == 0:
                ax.set_ylabel(label)
            ax.set_title(COEF_LABEL[c])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 0.99))
    return savefig(fig, out_dir, "figure_05_representative_cases")


def fig_06_residual_structure(df: pd.DataFrame, out_dir: Path) -> List[str]:
    if "state_id" not in df.columns or "band" not in df.columns:
        raise ValueError("Need state_id and band columns for residual PCA/correlation.")
    work = df.copy()
    for c in COEFS:
        work[f"res_{c}"] = pd.to_numeric(work[f"{c}_lrt"], errors="coerce") - pd.to_numeric(work[f"{c}_6s"], errors="coerce")
    # state-level residual vectors
    parts = []
    for c in COEFS:
        piv = work.pivot_table(index="state_id", columns="band", values=f"res_{c}", aggfunc="median")
        piv = piv.reindex(columns=[b for b in BAND_ORDER if b in piv.columns])
        piv.columns = [f"{c}_{b}" for b in piv.columns]
        parts.append(piv)
    X = pd.concat(parts, axis=1).dropna()
    if X.empty:
        raise ValueError("No complete residual vectors after pivot/dropna.")

    generated: List[str] = []
    corr = X.corr()
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    im = ax.imshow(corr.values, aspect="auto", vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_title("Residual correlation matrix (libRadtran - 6S)")
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    generated += savefig(fig, out_dir, "figure_06a_residual_correlation_matrix")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)
    pca = PCA(n_components=min(10, Xs.shape[1]))
    pca.fit(Xs)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes = axes.flatten()
    axes[0].bar(np.arange(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
    axes[0].set_title("Explained variance ratio")
    axes[0].set_xlabel("PC")
    axes[0].set_ylabel("Ratio")
    axes[0].grid(alpha=0.25)

    comp_df = pd.DataFrame(pca.components_[:3], columns=X.columns, index=["PC1", "PC2", "PC3"])
    for i, pc in enumerate(["PC1", "PC2", "PC3"], start=1):
        ax = axes[i]
        for c in COEFS:
            cols = [col for col in comp_df.columns if col.startswith(f"{c}_")]
            vals = comp_df.loc[pc, cols].values
            bands = [col.split("_")[-1] for col in cols]
            x = np.arange(len(bands))
            ax.plot(x, vals, marker="o", linewidth=1.5, label=c)
        ax.set_title(f"{pc} loadings by band")
        ax.set_xticks(np.arange(len(bands)))
        ax.set_xticklabels(bands, rotation=45)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    generated += savefig(fig, out_dir, "figure_06b_residual_pca_modes")
    return generated


def fig_07_feature_importance(df: pd.DataFrame, out_dir: Path) -> List[str]:
    req = [f"{c}_6s" for c in COEFS] + [f"{c}_lrt" for c in COEFS]
    if not all(c in df.columns for c in req):
        raise ValueError("Missing coefficient columns for residual target.")

    data = df.copy()
    for c in COEFS:
        data[f"res_{c}"] = pd.to_numeric(data[f"{c}_lrt"], errors="coerce") - pd.to_numeric(data[f"{c}_6s"], errors="coerce")

    feat_cols = [c for c in FEATURE_CANDIDATES if c in data.columns]
    feat_cols += [f"{c}_6s" for c in COEFS if f"{c}_6s" in data.columns]
    feat_cols = list(dict.fromkeys(feat_cols))
    if len(feat_cols) < 2:
        raise ValueError("Too few feature columns for diagnostic feature importance.")

    X = data[feat_cols].copy()
    y = data[[f"res_{c}" for c in COEFS]].copy()
    mask = X.notna().all(axis=1) & y.notna().all(axis=1)
    X, y = X[mask], y[mask]
    if len(X) < 100:
        raise ValueError("Too few complete rows for feature-importance training.")

    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    prep = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", ohe, cat_cols),
        ]
    )
    model = ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    pipe = Pipeline([("prep", prep), ("model", model)])
    pipe.fit(X, y)
    fitted = pipe.named_steps["model"]
    prep_f = pipe.named_steps["prep"]
    feat_names = prep_f.get_feature_names_out()
    imp = pd.Series(fitted.feature_importances_, index=feat_names).sort_values(ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.barh(np.arange(len(imp)), imp.values)
    ax.set_yticks(np.arange(len(imp)))
    ax.set_yticklabels(imp.index)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title("Diagnostic feature importance for residual structure (ExtraTrees)")
    ax.grid(alpha=0.25, axis="x")
    return savefig(fig, out_dir, "figure_07_feature_importance")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper diagnostic figures.")
    parser.add_argument("--data", type=Path, default=None, help="Single paired dataset with *_6s/*_lrt columns.")
    parser.add_argument("--data_6s", type=Path, default=None, help="6S dataset file.")
    parser.add_argument("--data_lrt", type=Path, default=None, help="libRadtran dataset file.")
    parser.add_argument("--predictions", type=Path, default=None, help="Optional prediction table.")
    parser.add_argument("--run_root", type=Path, default=None, help="Optional runs root for auto model/pred selection.")
    parser.add_argument("--split_mode", type=str, default="ood_aod_cwv", choices=["standard", "ood_aod_cwv"])
    parser.add_argument(
        "--accuracy_md",
        type=Path,
        default=None,
        help="Markdown benchmark (e.g. results/inference_accuracy_benchmark.md) for best-model selection.",
    )
    parser.add_argument(
        "--figures",
        type=str,
        default="all",
        help="Comma-separated figure ids to generate (01,05,all). Default: all.",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("figures/paper_diagnostics"))
    parser.add_argument(
        "--infer_device",
        type=str,
        default="cuda:0",
        help="Device for live surrogate inference when test_predictions.csv is absent.",
    )
    args = parser.parse_args()

    repo_root = _resolve_project_root()
    defaults = detect_default_paths(repo_root)
    figure_ids = {x.strip() for x in args.figures.split(",")} if args.figures != "all" else None
    data_path = args.data if args.data is not None else None
    data_6s_path = args.data_6s if args.data_6s is not None else defaults["data_6s"]
    data_lrt_path = args.data_lrt if args.data_lrt is not None else defaults["data_lrt"]
    pred_path = args.predictions if args.predictions is not None else defaults["predictions"]
    run_root = args.run_root if args.run_root is not None else defaults["run_root"]
    final_comp = defaults["final_comparison"]
    if final_comp is not None and not final_comp.exists():
        final_comp = None
    accuracy_md = args.accuracy_md if args.accuracy_md is not None else defaults.get("accuracy_md")
    if accuracy_md is not None and not accuracy_md.is_absolute():
        accuracy_md = repo_root / accuracy_md
    out_dir = args.output_dir if args.output_dir.is_absolute() else (repo_root / args.output_dir)

    summary = {
        "input_data_path": str(data_path) if data_path else None,
        "input_data_6s_path": str(data_6s_path) if data_6s_path else None,
        "input_data_lrt_path": str(data_lrt_path) if data_lrt_path else None,
        "prediction_path": str(pred_path) if pred_path else None,
        "split_mode": args.split_mode,
        "detected_coefficient_columns": {},
        "generated_figures": [],
        "skipped_figures": {},
        "rows_used": None,
        "number_of_bands": None,
        "available_models_modes": {},
        "selected_models": {},
    }

    paired, pair_meta = build_paired_dataframe(data_path, data_6s_path, data_lrt_path)
    summary["paired_mode"] = pair_meta.get("paired_mode")
    summary["merge_keys"] = pair_meta.get("merge_keys")
    summary["rows_used"] = int(len(paired))
    if "band" in paired.columns:
        summary["number_of_bands"] = int(paired["band"].nunique())
    summary["detected_coefficient_columns"] = find_coefficient_columns(paired)

    selected = _select_best_models(final_comp, args.split_mode)
    if accuracy_md is not None and accuracy_md.exists():
        md_best = _select_best_from_accuracy_md(accuracy_md, args.split_mode)
        if md_best is not None:
            selected["best_from_accuracy_md"] = md_best
    summary["selected_models"] = selected
    summary["accuracy_md_path"] = str(accuracy_md) if accuracy_md else None
    if final_comp and final_comp.exists():
        fc = pd.read_csv(final_comp)
        summary["available_models_modes"]["models"] = sorted(fc["model"].astype(str).unique().tolist()) if "model" in fc.columns else []
        summary["available_models_modes"]["split_modes"] = sorted(fc["split_mode"].astype(str).unique().tolist()) if "split_mode" in fc.columns else []
    elif accuracy_md is not None and accuracy_md.exists():
        ov = _load_overall_accuracy_md(accuracy_md)
        summary["available_models_modes"]["models"] = sorted(ov["model"].astype(str).unique().tolist())
        summary["available_models_modes"]["split_modes"] = sorted(ov["split_mode"].astype(str).unique().tolist())

    pred_best_kan = None
    pred_best_mlp = None
    pred_best_overall = None
    pred_label = "Surrogate"
    surrogate_row = selected.get("best_from_accuracy_md") or selected.get("best_overall")
    if run_root is not None and run_root.is_dir() and surrogate_row is not None:
        pred_best_overall, pred_label = _load_surrogate_predictions(
            paired, run_root, surrogate_row, args.split_mode, infer_device=args.infer_device
        )

    if run_root is not None and run_root.is_dir() and final_comp is not None:
        for key, model_label in [("best_kan", "kan"), ("best_mlp", "mlp")]:
            row = selected.get(key)
            if row is None:
                continue
            model_name = model_label
            variant_tag = str(row.get("variant_tag"))
            run_sub = row.get("run_subfolder")
            run_sub = str(run_sub) if run_sub is not None else None
            p_csv = _find_prediction_csv(
                run_root, variant_tag, args.split_mode, model_name, run_subfolder=run_sub
            )
            if p_csv is None:
                continue
            sub = _subset_test_rows_for_prediction(paired, p_csv)
            if sub is None:
                continue
            if key == "best_kan":
                pred_best_kan = sub
            elif key == "best_mlp":
                pred_best_mlp = sub

    def run_fig(fig_id: str, name: str, fn):
        if figure_ids is not None and fig_id not in figure_ids:
            return
        try:
            generated = fn()
            summary["generated_figures"].extend(generated)
        except Exception as exc:  # pylint: disable=broad-except
            summary["skipped_figures"][name] = str(exc)

    run_fig("01", "figure_01_state_space_coverage", lambda: fig_01_state_space_coverage(paired, out_dir))
    run_fig("02", "figure_02_6s_libradtran_discrepancy", lambda: fig_02_discrepancy(paired, out_dir))
    run_fig(
        "03",
        "figure_03_srtmnet_style_coefficients_residuals",
        lambda: fig_03_srtmnet_panel(paired, out_dir, pred_best_kan, pred_best_mlp),
    )
    run_fig("04", "figure_04_conditional_error_heatmap", lambda: fig_04_heatmaps(paired, out_dir, pred_best_overall))
    run_fig(
        "05",
        "figure_05_representative_cases",
        lambda: fig_05_representative_cases(paired, out_dir, pred_best_overall, pred_label=pred_label),
    )
    run_fig("06", "figure_06_residual_structure", lambda: fig_06_residual_structure(paired, out_dir))
    run_fig("07", "figure_07_feature_importance", lambda: fig_07_feature_importance(paired, out_dir))

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "figure_generation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Detected columns:")
    print(json.dumps(summary["detected_coefficient_columns"], indent=2))
    print(f"Rows used: {summary['rows_used']}, bands: {summary['number_of_bands']}")
    print("Selected models:")
    print(json.dumps(summary["selected_models"], indent=2, default=str))
    print("Generated figures:")
    for p in summary["generated_figures"]:
        print(f"- {p}")
    if summary["skipped_figures"]:
        print("Skipped figures:")
        for k, v in summary["skipped_figures"].items():
            print(f"- {k}: {v}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()