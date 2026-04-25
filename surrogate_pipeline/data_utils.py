import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "wvl_nm",
    "sza_deg",
    "vza_deg",
    "raa_deg",
    "aod550",
    "cwv_cm",
    "o3_cm",
    "elev_km",
]

CATEGORICAL_FEATURES = [
    "band",
    "aerosol_type",
    "atm_profile",
]

TARGET_COLUMNS = [
    "rho_path",
    "T_total",
    "spher_alb",
]

LOW_FIDELITY_TARGET_COLUMNS = [f"{c}_6s" for c in TARGET_COLUMNS]
HIGH_FIDELITY_TARGET_COLUMNS = [f"{c}_lrt" for c in TARGET_COLUMNS]
RESIDUAL_TARGET_COLUMNS = [f"delta_{c}" for c in TARGET_COLUMNS]

OPTIONAL_COLUMNS = ["split", "qa_valid", "state_id"]

ALIASES = {
    "band": ["band_id", "s2_band"],
    "wvl_nm": ["wavelength_nm", "wavelength"],
    "sza_deg": ["sza", "solar_zenith_deg", "solar_zenith"],
    "vza_deg": ["vza", "view_zenith_deg", "view_zenith"],
    "raa_deg": ["raa", "rel_azimuth_deg", "relative_azimuth_deg"],
    "aod550": ["aod", "aod_550", "aod_550nm"],
    "cwv_cm": ["cwv", "water_vapor_cm", "tcwv_cm"],
    "o3_cm": ["o3", "ozone_cm", "ozone"],
    "elev_km": ["elevation_km", "altitude_km", "elev"],
    "atm_profile": ["atmos_profile", "atmospheric_profile", "profile"],
    "rho_path": ["path_reflectance", "rho_path_raw_model"],
    "T_total": ["transmittance_total", "ttotal", "t_total"],
    "spher_alb": ["spherical_albedo", "sph_alb"],
    "qa_valid": ["qa", "is_valid", "valid"],
    "state_id": ["state", "sample_id"],
    "split": ["set", "partition"],
}

SPLIT_ALIASES = {
    "train": "train",
    "trn": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "dev": "val",
    "test": "test",
    "tst": "test",
    "testing": "test",
}


@dataclass
class SplitResult:
    split_source: str
    split_column: Optional[str]
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def load_jsonl(path: str) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def resolve_column_names(df: pd.DataFrame, required: List[str], optional: Optional[List[str]] = None) -> Dict[str, Optional[str]]:
    optional = optional or []
    normalized_to_actual = {_normalize_name(c): c for c in df.columns}
    resolved: Dict[str, Optional[str]] = {}
    for name in required + optional:
        candidates = [name] + ALIASES.get(name, [])
        selected = None
        for c in candidates:
            if c in df.columns:
                selected = c
                break
            n = _normalize_name(c)
            if n in normalized_to_actual:
                selected = normalized_to_actual[n]
                break
        if selected is None and name in required:
            raise ValueError(f"Missing required column for '{name}'. Available columns: {list(df.columns)}")
        resolved[name] = selected
    return resolved


def normalize_split_values(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().map(SPLIT_ALIASES)


def split_by_state(
    df: pd.DataFrame,
    state_col: str,
    split_col: Optional[str],
    seed: int,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> SplitResult:
    if split_col is not None:
        normalized = normalize_split_values(df[split_col])
        if normalized.notna().all():
            state_split = pd.DataFrame({state_col: df[state_col], "split": normalized}).drop_duplicates()
            state_counts = state_split.groupby(state_col)["split"].nunique()
            if (state_counts == 1).all():
                state_to_split = state_split.groupby(state_col)["split"].first()
                train_states = set(state_to_split[state_to_split == "train"].index)
                val_states = set(state_to_split[state_to_split == "val"].index)
                test_states = set(state_to_split[state_to_split == "test"].index)
                if train_states and val_states and test_states:
                    mask_train = df[state_col].isin(train_states).to_numpy()
                    mask_val = df[state_col].isin(val_states).to_numpy()
                    mask_test = df[state_col].isin(test_states).to_numpy()
                    return SplitResult(
                        split_source="existing_split_column",
                        split_column=split_col,
                        train_idx=np.where(mask_train)[0],
                        val_idx=np.where(mask_val)[0],
                        test_idx=np.where(mask_test)[0],
                    )

    unique_states = pd.Series(df[state_col].astype(str).unique())
    shuffled = unique_states.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_states = len(shuffled)
    n_train = int(n_states * train_frac)
    n_val = int(n_states * val_frac)
    train_states = set(shuffled.iloc[:n_train].tolist())
    val_states = set(shuffled.iloc[n_train : n_train + n_val].tolist())
    test_states = set(shuffled.iloc[n_train + n_val :].tolist())

    state_series = df[state_col].astype(str)
    mask_train = state_series.isin(train_states).to_numpy()
    mask_val = state_series.isin(val_states).to_numpy()
    mask_test = state_series.isin(test_states).to_numpy()

    return SplitResult(
        split_source="reconstructed_by_state_id",
        split_column=None,
        train_idx=np.where(mask_train)[0],
        val_idx=np.where(mask_val)[0],
        test_idx=np.where(mask_test)[0],
    )


def filter_excluded_band(df: pd.DataFrame, band_col: str, exclude_b10: bool) -> pd.DataFrame:
    if not exclude_b10:
        return df
    return df[df[band_col].astype(str).str.upper() != "B10"].copy()


def assert_no_duplicate_keys(df: pd.DataFrame, key_cols: List[str], name: str) -> None:
    dup = df.duplicated(subset=key_cols, keep=False)
    if bool(dup.any()):
        examples = df.loc[dup, key_cols].head(5).to_dict(orient="records")
        raise ValueError(f"Found duplicate {name} keys on {key_cols}. Example duplicates: {examples}")


def merge_multi_fidelity(
    df_6s: pd.DataFrame,
    df_lrt: pd.DataFrame,
    key_cols: List[str],
    target_cols: List[str],
) -> pd.DataFrame:
    missing_6s = [c for c in key_cols + target_cols if c not in df_6s.columns]
    missing_lrt = [c for c in key_cols + target_cols if c not in df_lrt.columns]
    if missing_6s:
        raise ValueError(f"6S dataset missing required columns: {missing_6s}")
    if missing_lrt:
        raise ValueError(f"libRadtran dataset missing required columns: {missing_lrt}")

    assert_no_duplicate_keys(df_6s, key_cols=key_cols, name="6S")
    assert_no_duplicate_keys(df_lrt, key_cols=key_cols, name="libRadtran")

    merged = df_6s.merge(df_lrt, on=key_cols, how="inner", suffixes=("_6s", "_lrt"), validate="one_to_one")
    if merged.empty:
        raise ValueError("Merged multi-fidelity dataset is empty. Check matching keys between 6S and libRadtran files.")

    for t in target_cols:
        c6 = f"{t}_6s"
        cl = f"{t}_lrt"
        merged[c6] = pd.to_numeric(merged[c6], errors="coerce")
        merged[cl] = pd.to_numeric(merged[cl], errors="coerce")
        merged[f"delta_{t}"] = merged[cl] - merged[c6]
    return merged


def validate_split_integrity(df: pd.DataFrame, state_col: str, split_idx: Dict[str, np.ndarray]) -> Dict[str, int]:
    state_sets = {
        split_name: set(df.iloc[idx][state_col].astype(str).tolist())
        for split_name, idx in split_idx.items()
    }
    leakage = {
        "train_val_overlap": len(state_sets["train"] & state_sets["val"]),
        "train_test_overlap": len(state_sets["train"] & state_sets["test"]),
        "val_test_overlap": len(state_sets["val"] & state_sets["test"]),
    }
    return leakage


def infer_expected_rows_per_state(df: pd.DataFrame, state_col: Optional[str], band_col: Optional[str]) -> Tuple[Optional[int], pd.Series]:
    if state_col is None:
        return None, pd.Series(dtype=int)
    state_counts = df.groupby(state_col).size()
    if state_counts.empty:
        return None, state_counts
    expected = int(state_counts.mode().iloc[0])
    if band_col is not None and band_col in df.columns:
        unique_band_count = int(df[band_col].nunique())
        if unique_band_count > 1:
            expected = unique_band_count
    return expected, state_counts


def summarize_numeric_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    stats = []
    for col in columns:
        numeric = pd.to_numeric(df[col], errors="coerce")
        stats.append(
            {
                "column": col,
                "count": int(numeric.notna().sum()),
                "min": float(numeric.min()),
                "max": float(numeric.max()),
                "mean": float(numeric.mean()),
                "std": float(numeric.std(ddof=0)),
            }
        )
    return pd.DataFrame(stats)


def _make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(train_df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", _make_ohe(), categorical_cols),
        ],
        remainder="drop",
    )
    transformer.fit(train_df[numeric_cols + categorical_cols])
    return transformer
