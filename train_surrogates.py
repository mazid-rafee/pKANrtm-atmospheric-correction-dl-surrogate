import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

from surrogate_pipeline.data_utils import (
    CATEGORICAL_FEATURES,
    HIGH_FIDELITY_TARGET_COLUMNS,
    LOW_FIDELITY_TARGET_COLUMNS,
    NUMERIC_FEATURES,
    RESIDUAL_TARGET_COLUMNS,
    TARGET_COLUMNS,
    assert_no_duplicate_keys,
    build_preprocessor,
    filter_excluded_band,
    load_jsonl,
    merge_multi_fidelity,
    resolve_column_names,
    split_by_state,
    validate_split_integrity,
)
from surrogate_pipeline.models import build_model
from surrogate_pipeline.train_utils import (
    PERCENT_METRIC_EPS,
    compute_error_metrics,
    compute_metrics,
    physics_penalty,
    save_accuracy_runtime_scatter,
    save_json,
    save_loss_curve,
    save_residual_histograms,
    save_rmse_bar,
    save_runtime_bar,
    save_scatter_plots,
    seed_everything,
    select_device,
)

NEURAL_MODELS = {"mlp", "kan", "pmlp", "pkan"}
FIDELITY_MODES = [
    "single_fidelity_6s",
    "single_fidelity_libradtran",
    "oracle_residual",
    "deployable_residual",
]
COMBINED_FIDELITY_MODES = ["both_residual"]
MANDATORY_NUMERIC_FEATURES = [c for c in NUMERIC_FEATURES if c != "wvl_nm"]


def _build_dataloader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).float())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=False)


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "toarray"):
        return x.toarray()
    return np.asarray(x)


def _progress(iterable, enabled: bool, **kwargs):
    if enabled and tqdm is not None:
        return tqdm(iterable, **kwargs)
    return iterable


def _predict_neural(model: torch.nn.Module, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    preds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = torch.from_numpy(x[start : start + batch_size]).float().to(device, non_blocking=True)
            preds.append(model(xb).detach().cpu().numpy())
    return np.vstack(preds).astype(np.float32)


def _benchmark_predict(predict_fn, n_samples: int, repeats: int, device: Optional[torch.device] = None, batch_size: Optional[int] = None) -> Dict[str, float]:
    _ = predict_fn()
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = predict_fn()
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(time.perf_counter() - t0)
    avg_total = float(np.mean(times))
    result = {
        "avg_test_inference_sec": avg_total,
        "total_test_inference_sec": float(times[-1]),
        "per_sample_inference_sec": float(avg_total / max(1, n_samples)),
        "runtime_repeats": int(repeats),
    }
    if batch_size is not None and batch_size > 0:
        result["per_batch_inference_sec"] = float(avg_total / max(1, int(np.ceil(n_samples / batch_size))))
    return result


def _resolve_rtm_row_time(args) -> Optional[float]:
    if args.rtm_row_time_sec is not None:
        return float(args.rtm_row_time_sec)
    if args.rtm_timing_json is None:
        return None
    p = Path(args.rtm_timing_json)
    if not p.exists():
        return None
    data = pd.read_json(p, typ="series")
    for key in ["avg_row_runtime_sec", "avg_runtime_sec", "mean_row_runtime_sec"]:
        if key in data.index:
            return float(data[key])
    return None


def _xgb_device_from_gpu_arg(gpu_arg: str) -> str:
    if str(gpu_arg).lower() == "cpu":
        return "cpu"
    try:
        gpu_index = int(gpu_arg)
    except ValueError as exc:
        raise ValueError("--gpu must be 'cpu' or an integer GPU index for xgboost/rf.") from exc
    return f"cuda:{gpu_index}"


def _canonicalize_input_df(df: pd.DataFrame, resolved: Dict[str, Optional[str]]) -> pd.DataFrame:
    rename_map = {v: k for k, v in resolved.items() if v is not None}
    return df.rename(columns=rename_map)


def _load_single_fidelity(path: str, exclude_b10: bool) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    df = load_jsonl(path)
    resolved = resolve_column_names(
        df=df,
        required=CATEGORICAL_FEATURES + MANDATORY_NUMERIC_FEATURES + TARGET_COLUMNS + ["state_id"],
        optional=["split", "qa_valid", "wvl_nm"],
    )
    qa_col = resolved["qa_valid"]
    if qa_col is not None:
        df = df[df[qa_col].astype(bool)].copy()
    df = _canonicalize_input_df(df, resolved)
    df = filter_excluded_band(df, band_col="band", exclude_b10=exclude_b10)
    assert_no_duplicate_keys(df, key_cols=["state_id", "band"], name="single-fidelity")
    return df.reset_index(drop=True), resolved


def _load_dual_fidelity(path_6s: str, path_lrt: str, exclude_b10: bool) -> Tuple[pd.DataFrame, Dict]:
    df_6s, resolved_6s = _load_single_fidelity(path_6s, exclude_b10=exclude_b10)
    df_lrt, resolved_lrt = _load_single_fidelity(path_lrt, exclude_b10=exclude_b10)
    merge_df = merge_multi_fidelity(df_6s=df_6s, df_lrt=df_lrt, key_cols=["state_id", "band"], target_cols=TARGET_COLUMNS)
    n_6s = len(df_6s)
    n_lrt = len(df_lrt)
    n_merged = len(merge_df)
    meta = {
        "rows_6s": int(n_6s),
        "rows_lrt": int(n_lrt),
        "rows_merged": int(n_merged),
        "unmatched_6s_rows": int(max(n_6s - n_merged, 0)),
        "unmatched_lrt_rows": int(max(n_lrt - n_merged, 0)),
        "resolved_6s": resolved_6s,
        "resolved_lrt": resolved_lrt,
    }
    return merge_df.reset_index(drop=True), meta


def _resolve_merged_feature_columns(df: pd.DataFrame) -> Dict[str, str]:
    resolved = {}
    for feature in CATEGORICAL_FEATURES + NUMERIC_FEATURES:
        if feature == "wvl_nm" and feature not in df.columns and f"{feature}_lrt" not in df.columns and f"{feature}_6s" not in df.columns:
            continue
        if feature in df.columns:
            resolved[feature] = feature
        elif f"{feature}_lrt" in df.columns:
            resolved[feature] = f"{feature}_lrt"
        elif f"{feature}_6s" in df.columns:
            resolved[feature] = f"{feature}_6s"
        else:
            raise ValueError(f"Missing required feature in merged dataframe: {feature}")
    return resolved


def _build_ood_state_split(df: pd.DataFrame, state_col: str, aod_col: str, cwv_col: str, seed: int, quantile: float) -> Tuple[Dict[str, np.ndarray], pd.DataFrame, Dict]:
    state_df = df[[state_col, aod_col, cwv_col]].drop_duplicates(subset=[state_col]).copy()
    state_df[state_col] = state_df[state_col].astype(str)
    q_aod = float(state_df[aod_col].quantile(quantile))
    q_cwv = float(state_df[cwv_col].quantile(quantile))
    test_mask = (state_df[aod_col] >= q_aod) & (state_df[cwv_col] >= q_cwv)
    min_test_states = max(1, int(0.10 * len(state_df)))
    if int(test_mask.sum()) < min_test_states:
        test_mask = (state_df[aod_col] >= q_aod) | (state_df[cwv_col] >= q_cwv)
    if int(test_mask.sum()) < min_test_states:
        z_aod = (state_df[aod_col] - state_df[aod_col].mean()) / (state_df[aod_col].std(ddof=0) + 1e-12)
        z_cwv = (state_df[cwv_col] - state_df[cwv_col].mean()) / (state_df[cwv_col].std(ddof=0) + 1e-12)
        top_k = max(1, int(0.15 * len(state_df)))
        test_mask = state_df.index.isin((z_aod + z_cwv).sort_values(ascending=False).head(top_k).index)

    test_states = set(state_df.loc[test_mask, state_col].astype(str).tolist())
    remaining_states = state_df.loc[~test_mask, state_col].astype(str).sample(frac=1.0, random_state=seed).tolist()
    if len(remaining_states) < 2:
        raise RuntimeError("OOD split failed due to insufficient remaining states.")
    n_val = max(1, int(0.15 * len(remaining_states)))
    n_val = min(n_val, len(remaining_states) - 1)
    val_states = set(remaining_states[:n_val])
    train_states = set(remaining_states[n_val:])

    state_series = df[state_col].astype(str)
    idx_map = {
        "train": np.where(state_series.isin(train_states).to_numpy())[0],
        "val": np.where(state_series.isin(val_states).to_numpy())[0],
        "test": np.where(state_series.isin(test_states).to_numpy())[0],
    }
    assignment = state_df[[state_col, aod_col, cwv_col]].copy()
    assignment["split"] = assignment[state_col].astype(str).map(lambda x: "test" if x in test_states else ("val" if x in val_states else "train"))
    meta = {
        "split_source": "ood_aod_cwv_holdout_by_state",
        "quantile": quantile,
        "aod_threshold": q_aod,
        "cwv_threshold": q_cwv,
        "n_train_states": int(len(train_states)),
        "n_val_states": int(len(val_states)),
        "n_test_states": int(len(test_states)),
    }
    return idx_map, assignment, meta


def _get_split(
    df: pd.DataFrame,
    split_mode: str,
    seed: int,
    ood_quantile: float,
    aod_col: str = "aod550",
    cwv_col: str = "cwv_cm",
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame, Dict]:
    if split_mode == "standard":
        split_result = split_by_state(df=df, state_col="state_id", split_col="split" if "split" in df.columns else None, seed=seed, train_frac=0.7, val_frac=0.15)
        idx_map = {"train": split_result.train_idx, "val": split_result.val_idx, "test": split_result.test_idx}
        assignment = pd.DataFrame({"state_id": df["state_id"].astype(str)})
        assignment["split"] = "train"
        assignment.loc[idx_map["val"], "split"] = "val"
        assignment.loc[idx_map["test"], "split"] = "test"
        assignment = assignment.drop_duplicates(subset=["state_id"]).reset_index(drop=True)
        meta = {"split_source": split_result.split_source, "split_column": split_result.split_column}
        return idx_map, assignment, meta
    if split_mode == "ood_aod_cwv":
        return _build_ood_state_split(df=df, state_col="state_id", aod_col=aod_col, cwv_col=cwv_col, seed=seed, quantile=ood_quantile)
    raise ValueError(f"Unknown split_mode: {split_mode}")


def _rows_from_metrics(model_name: str, split_mode: str, fidelity_mode: str, metrics: Dict) -> List[Dict]:
    rows = []
    for split_name, split_metrics in metrics.items():
        for target_name, vals in split_metrics.items():
            rows.append(
                {
                    "fidelity_mode": fidelity_mode,
                    "split_mode": split_mode,
                    "model": model_name,
                    "split": split_name,
                    "target": target_name,
                    "rmse": float(vals["rmse"]),
                    "mae": float(vals["mae"]),
                    "r2": float(vals["r2"]),
                    "mape": float(vals["mape"]),
                    "smape": float(vals["smape"]),
                }
            )
    return rows


def _canonical_metric_target_name(name: str) -> str:
    n = str(name).strip()
    lower = n.lower()
    if lower.startswith("delta_"):
        n = n[6:]
        lower = n.lower()
    for suffix in ("_6s", "_lrt"):
        if lower.endswith(suffix):
            n = n[: -len(suffix)]
            break
    return n


def _build_detailed_error_tables(
    split_frames: Dict[str, pd.DataFrame],
    split_preds: Dict[str, np.ndarray],
    split_truth: Dict[str, np.ndarray],
    targets: List[str],
    band_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    band_rows: List[Dict] = []
    coefficient_rows: List[Dict] = []
    band_coefficient_rows: List[Dict] = []

    for split_name in ["train", "val", "test"]:
        y_true = split_truth[split_name]
        y_pred = split_preds[split_name]
        split_df = split_frames[split_name]
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Prediction/target shape mismatch in {split_name}: {y_pred.shape} vs {y_true.shape}")
        if len(split_df) != y_true.shape[0]:
            raise ValueError(f"Split frame size mismatch in {split_name}: {len(split_df)} vs {y_true.shape[0]}")

        band_values = split_df[band_col].astype(str).to_numpy()

        for target_i, target_name in enumerate(targets):
            metric_vals = compute_error_metrics(y_true[:, target_i], y_pred[:, target_i], eps=PERCENT_METRIC_EPS)
            coefficient_rows.append(
                {
                    "split": split_name,
                    "target": target_name,
                    "coefficient": _canonical_metric_target_name(target_name),
                    "n_rows": int(y_true.shape[0]),
                    **metric_vals,
                }
            )

        for band_name in sorted(pd.unique(band_values)):
            mask = band_values == band_name
            if not np.any(mask):
                continue
            metric_vals = compute_error_metrics(y_true[mask], y_pred[mask], eps=PERCENT_METRIC_EPS)
            band_rows.append(
                {
                    "split": split_name,
                    "band": str(band_name),
                    "n_rows": int(np.sum(mask)),
                    "n_elements": int(np.sum(mask) * len(targets)),
                    **metric_vals,
                }
            )
            for target_i, target_name in enumerate(targets):
                metric_vals_bt = compute_error_metrics(y_true[mask, target_i], y_pred[mask, target_i], eps=PERCENT_METRIC_EPS)
                band_coefficient_rows.append(
                    {
                        "split": split_name,
                        "band": str(band_name),
                        "target": target_name,
                        "coefficient": _canonical_metric_target_name(target_name),
                        "n_rows": int(np.sum(mask)),
                        **metric_vals_bt,
                    }
                )

    band_df = pd.DataFrame(band_rows).sort_values(["split", "band"]).reset_index(drop=True)
    coefficient_df = pd.DataFrame(coefficient_rows).sort_values(["split", "coefficient"]).reset_index(drop=True)
    band_coefficient_df = pd.DataFrame(band_coefficient_rows).sort_values(["split", "band", "coefficient"]).reset_index(drop=True)
    return band_df, coefficient_df, band_coefficient_df


def _fit_target_scaler(y_train: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(y_train)
    return scaler


def _train_neural_stage(
    model_name: str,
    stage_name: str,
    out_dir: Path,
    split_data_scaled: Dict[str, Dict[str, np.ndarray]],
    split_data_raw_y: Dict[str, np.ndarray],
    metric_targets: List[str],
    y_scaler: StandardScaler,
    device: torch.device,
    lr: float,
    batch_size: int,
    epochs: int,
    patience: int,
    lambda_phys: float,
    num_workers: int,
    use_amp: bool,
    runtime_batch_size: int,
    benchmark_runtime: bool,
    runtime_repeats: int,
    rtm_row_time_sec: Optional[float],
    show_progress: bool,
    mlp_arch: str,
    kan_arch: str,
    activation: str,
    use_layernorm: bool,
    deployable_stage_sizing: bool,
) -> Tuple[Dict, Dict[str, np.ndarray], Optional[Dict], Dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    model, use_phys = build_model(
        model_name=model_name,
        in_dim=split_data_scaled["train"]["x"].shape[1],
        out_dim=split_data_scaled["train"]["y"].shape[1],
        stage_name=stage_name,
        mlp_arch=mlp_arch,
        kan_arch=kan_arch,
        activation=activation,
        use_layernorm=use_layernorm,
        deployable_stage_sizing=deployable_stage_sizing,
    )
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device="cuda", enabled=amp_enabled)
    phys_mean = torch.as_tensor(y_scaler.mean_, device=device, dtype=torch.float32)
    phys_scale = torch.as_tensor(y_scaler.scale_, device=device, dtype=torch.float32)
    train_loader = _build_dataloader(split_data_scaled["train"]["x"], split_data_scaled["train"]["y"], batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = _build_dataloader(split_data_scaled["val"]["x"], split_data_scaled["val"]["y"], batch_size=batch_size, shuffle=False, num_workers=num_workers)
    best_val = float("inf")
    wait = 0
    history = []
    checkpoint_path = out_dir / "best.pt"

    t0_train = time.perf_counter()
    epoch_iter = _progress(
        range(1, epochs + 1),
        enabled=show_progress,
        desc=f"{stage_name}:{model_name}",
        unit="epoch",
        leave=True,
    )
    for epoch in epoch_iter:
        model.train()
        train_losses = []
        train_iter = _progress(
            train_loader,
            enabled=show_progress,
            desc=f"{stage_name}:{model_name} train e{epoch}",
            unit="batch",
            leave=False,
        )
        for xb, yb in train_iter:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                pred = model(xb)
                loss_pred = criterion(pred, yb)
                phys = (
                    physics_penalty((pred.float() * phys_scale) + phys_mean, target_names=metric_targets)
                    if use_phys
                    else torch.zeros((), device=device)
                )
                loss = loss_pred + (lambda_phys * phys if use_phys else 0.0)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_loss = float(loss.detach().cpu().item())
            train_losses.append(batch_loss)
            if show_progress and hasattr(train_iter, "set_postfix"):
                train_iter.set_postfix(loss=f"{batch_loss:.4e}")

        model.eval()
        val_losses = []
        with torch.no_grad():
            val_iter = _progress(
                val_loader,
                enabled=show_progress,
                desc=f"{stage_name}:{model_name} val e{epoch}",
                unit="batch",
                leave=False,
            )
            for xb, yb in val_iter:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                    pred = model(xb)
                    loss_pred = criterion(pred, yb)
                    phys = (
                        physics_penalty((pred.float() * phys_scale) + phys_mean, target_names=metric_targets)
                        if use_phys
                        else torch.zeros((), device=device)
                    )
                    loss = loss_pred + (lambda_phys * phys if use_phys else 0.0)
                val_loss_batch = float(loss.detach().cpu().item())
                val_losses.append(val_loss_batch)
                if show_progress and hasattr(val_iter, "set_postfix"):
                    val_iter.set_postfix(loss=f"{val_loss_batch:.4e}")

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if show_progress and hasattr(epoch_iter, "set_postfix"):
            epoch_iter.set_postfix(train_loss=f"{train_loss:.4e}", val_loss=f"{val_loss:.4e}")
        print(f"[{stage_name}:{model_name}] epoch={epoch}/{epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "best_val_loss": best_val, "model_name": model_name}, checkpoint_path)
        else:
            wait += 1
            if wait >= patience:
                break
    train_seconds = float(time.perf_counter() - t0_train)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    history_df = pd.DataFrame(history)
    history_df.to_csv(out_dir / "history.csv", index=False)
    save_loss_curve(history_df, out_dir / "loss_curve.png")

    preds = {}
    for split_name in ["train", "val", "test"]:
        preds[split_name] = _predict_neural(model, split_data_scaled[split_name]["x"], device=device, batch_size=runtime_batch_size)

    runtime_row = None
    if benchmark_runtime:
        runtime = _benchmark_predict(
            predict_fn=lambda: _predict_neural(model, split_data_scaled["test"]["x"], device=device, batch_size=runtime_batch_size),
            n_samples=split_data_scaled["test"]["x"].shape[0],
            repeats=runtime_repeats,
            device=device,
            batch_size=runtime_batch_size,
        )
        runtime_row = {"model": model_name, "stage": stage_name, "train_time_sec": train_seconds, **runtime}
        if rtm_row_time_sec is not None:
            runtime_row["rtm_row_time_sec"] = float(rtm_row_time_sec)
            runtime_row["surrogate_speedup_vs_rtm"] = float(rtm_row_time_sec / max(runtime_row["per_sample_inference_sec"], 1e-12))
        save_json(runtime_row, out_dir / "runtime.json")
    return {}, preds, runtime_row, {"checkpoint_path": str(checkpoint_path)}


def _train_baseline_stage(
    model_name: str,
    stage_name: str,
    out_dir: Path,
    split_data_scaled: Dict[str, Dict[str, np.ndarray]],
    split_data_raw_y: Dict[str, np.ndarray],
    metric_targets: List[str],
    seed: int,
    num_workers: int,
    benchmark_runtime: bool,
    runtime_repeats: int,
    runtime_batch_size: int,
    rtm_row_time_sec: Optional[float],
    xgb_trees: int,
    rf_trees: int,
    gpu_arg: str,
    rf_progress_steps: int,
    show_progress: bool,
) -> Tuple[Dict, Dict[str, np.ndarray], Optional[Dict], Dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if model_name == "xgb":
        try:
            from xgboost import XGBRegressor
        except Exception as exc:
            raise ImportError("xgboost is not available. Use model 'rf' or install xgboost.") from exc
        estimator_device = _xgb_device_from_gpu_arg(gpu_arg)
        base = XGBRegressor(
            n_estimators=xgb_trees,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=num_workers if num_workers > 0 else None,
            tree_method="hist",
            device=estimator_device,
            verbosity=0,
        )
        models = []
        t0_train = time.perf_counter()
        target_iter = _progress(
            range(split_data_scaled["train"]["y"].shape[1]),
            enabled=show_progress,
            desc=f"{stage_name}:{model_name} targets",
            unit="target",
            leave=True,
        )
        for i in target_iter:
            model_i = XGBRegressor(**base.get_params())
            model_i.fit(split_data_scaled["train"]["x"], split_data_scaled["train"]["y"][:, i])
            models.append(model_i)
            print(f"[{stage_name}:{model_name}] epoch={i+1}/{split_data_scaled['train']['y'].shape[1]} done")
        train_seconds = float(time.perf_counter() - t0_train)
    else:
        try:
            from xgboost import XGBRFRegressor
        except Exception as exc:
            raise ImportError("xgboost is required for rf baseline. Install with pip install xgboost.") from exc
        estimator_device = _xgb_device_from_gpu_arg(gpu_arg)
        steps = max(1, min(rf_progress_steps, rf_trees))
        models = []
        t0_train = time.perf_counter()
        target_iter = _progress(
            range(split_data_scaled["train"]["y"].shape[1]),
            enabled=show_progress,
            desc=f"{stage_name}:{model_name} targets",
            unit="target",
            leave=True,
        )
        for target_i in target_iter:
            booster_ref = None
            prev_trees = 0
            step_iter = _progress(
                range(steps),
                enabled=show_progress,
                desc=f"{stage_name}:{model_name} t{target_i+1}",
                unit="step",
                leave=False,
            )
            for step_idx in step_iter:
                trees_done_next = max(prev_trees + 1, int(round(((step_idx + 1) * rf_trees) / steps)))
                trees_done_next = min(trees_done_next, rf_trees)
                model_i = XGBRFRegressor(
                    n_estimators=trees_done_next,
                    max_depth=12,
                    subsample=0.8,
                    colsample_bynode=0.8,
                    objective="reg:squarederror",
                    random_state=seed + target_i,
                    n_jobs=num_workers if num_workers > 0 else None,
                    tree_method="hist",
                    device=estimator_device,
                    verbosity=0,
                )
                if booster_ref is None:
                    model_i.fit(split_data_scaled["train"]["x"], split_data_scaled["train"]["y"][:, target_i])
                else:
                    model_i.fit(split_data_scaled["train"]["x"], split_data_scaled["train"]["y"][:, target_i], xgb_model=booster_ref)
                booster_ref = model_i
                prev_trees = trees_done_next
                if show_progress and hasattr(step_iter, "set_postfix"):
                    step_iter.set_postfix(trees=f"{trees_done_next}/{rf_trees}")
                print(f"[{stage_name}:{model_name}] epoch={step_idx+1}/{steps} target={target_i+1}/{split_data_scaled['train']['y'].shape[1]}")
                if trees_done_next >= rf_trees:
                    break
            models.append(booster_ref)
        train_seconds = float(time.perf_counter() - t0_train)

    joblib.dump(models, out_dir / "model.joblib")

    def _predict_fn(x):
        return np.column_stack([m.predict(x) for m in models]).astype(np.float32)

    preds = {}
    for split_name in ["train", "val", "test"]:
        p = _predict_fn(split_data_scaled[split_name]["x"])
        preds[split_name] = p

    runtime_row = None
    if benchmark_runtime:
        runtime = _benchmark_predict(
            predict_fn=lambda: _predict_fn(split_data_scaled["test"]["x"]),
            n_samples=split_data_scaled["test"]["x"].shape[0],
            repeats=runtime_repeats,
            device=None,
            batch_size=runtime_batch_size,
        )
        runtime_row = {"model": model_name, "stage": stage_name, "train_time_sec": train_seconds, **runtime}
        if rtm_row_time_sec is not None:
            runtime_row["rtm_row_time_sec"] = float(rtm_row_time_sec)
            runtime_row["surrogate_speedup_vs_rtm"] = float(rtm_row_time_sec / max(runtime_row["per_sample_inference_sec"], 1e-12))
        save_json(runtime_row, out_dir / "runtime.json")
    return {}, preds, runtime_row, {"model_path": str(out_dir / "model.joblib")}


def _build_best_worst_table(df: pd.DataFrame, entity_col: str) -> pd.DataFrame:
    rows: List[Dict] = []
    if df.empty:
        return pd.DataFrame(rows)
    for model_name, model_df in df.groupby("model"):
        for metric_name in ["rmse", "smape"]:
            sorted_df = model_df.sort_values(metric_name)
            best = sorted_df.iloc[0]
            worst = sorted_df.iloc[-1]
            rows.append(
                {
                    "model": model_name,
                    "metric": metric_name,
                    f"best_{entity_col}": best[entity_col],
                    "best_value": float(best[metric_name]),
                    f"worst_{entity_col}": worst[entity_col],
                    "worst_value": float(worst[metric_name]),
                }
            )
    return pd.DataFrame(rows)


def _build_mode_report(
    metrics_df: pd.DataFrame,
    runtime_df: pd.DataFrame,
    band_metrics_df: pd.DataFrame,
    coefficient_metrics_df: pd.DataFrame,
    band_coefficient_metrics_df: pd.DataFrame,
    out_path: Path,
    split_mode: str,
    fidelity_mode: str,
) -> None:
    overall_table = metrics_df[metrics_df["target"] == "overall"][["model", "split", "rmse", "mae", "r2", "mape", "smape"]].sort_values(["split", "rmse", "model"])
    test_band = band_metrics_df[band_metrics_df["split"] == "test"].sort_values(["model", "rmse", "band"])
    test_coeff = coefficient_metrics_df[coefficient_metrics_df["split"] == "test"].sort_values(["model", "rmse", "coefficient"])
    best_worst_band = _build_best_worst_table(test_band, entity_col="band")
    best_worst_coeff = _build_best_worst_table(test_coeff, entity_col="coefficient")

    lines = [f"# Model Comparison ({fidelity_mode}, {split_mode})", ""]
    lines.extend(
        [
            "MAPE/sMAPE denominator epsilon:",
            f"`eps = {PERCENT_METRIC_EPS:.1e}`",
            "",
            "For near-zero targets, MAPE can be unstable; treat sMAPE as the more robust percentage error metric.",
            "",
        ]
    )
    lines.extend(["## Overall Metrics (All Splits)", "", overall_table.to_markdown(index=False), ""])
    lines.extend(["## Band-wise Metrics (Test Split)", "", test_band.to_markdown(index=False), ""])
    lines.extend(["## Coefficient-wise Metrics (Test Split)", "", test_coeff.to_markdown(index=False), ""])
    if not best_worst_band.empty:
        lines.extend(["## Best/Worst Bands by RMSE and sMAPE (Test Split)", "", best_worst_band.to_markdown(index=False), ""])
    if not best_worst_coeff.empty:
        lines.extend(["## Best/Worst Coefficients by RMSE and sMAPE (Test Split)", "", best_worst_coeff.to_markdown(index=False), ""])
    lines.extend(["## Band x Coefficient Matrix (Test Split)", "", band_coefficient_metrics_df[band_coefficient_metrics_df["split"] == "test"].to_markdown(index=False), ""])
    lines.extend(
        [
            "## CSV Exports",
            "",
            "- overall metrics: `combined_metrics.csv`",
            "- band-wise metrics: `band_metrics_table.csv`",
            "- coefficient-wise metrics: `coefficient_metrics_table.csv`",
            "- band x coefficient matrix: `band_coefficient_metrics_table.csv`",
            "- best/worst summaries: `best_worst_band_metrics.csv`, `best_worst_coefficient_metrics.csv`",
            "",
        ]
    )
    if not runtime_df.empty:
        lines.extend(["## Runtime", "", runtime_df.sort_values("avg_test_inference_sec").to_markdown(index=False), ""])
    lines.extend(["## Full Per-Target Metrics (All Splits)", "", metrics_df.to_markdown(index=False)])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _build_global_report(accuracy_df: pd.DataFrame, runtime_df: pd.DataFrame, accuracy_runtime_df: pd.DataFrame, out_path: Path, fidelity_mode: str) -> None:
    lines = [f"# Surrogate Comparison Report ({fidelity_mode})", ""]
    lines.extend(
        [
            "MAPE/sMAPE denominator epsilon:",
            f"`eps = {PERCENT_METRIC_EPS:.1e}`",
            "",
            "For near-zero targets, MAPE can be unstable; treat sMAPE as the more robust percentage error metric.",
            "",
        ]
    )
    lines.extend(["## Overall Summary Table", "", accuracy_df.to_markdown(index=False), ""])
    if not runtime_df.empty:
        lines.extend(["## Runtime Table", "", runtime_df.to_markdown(index=False), ""])
    if not accuracy_runtime_df.empty:
        lines.extend(["## Accuracy vs Runtime", "", accuracy_runtime_df.to_markdown(index=False), ""])
    lines.extend(
        [
            "## CSV Exports",
            "",
            "- overall metrics: `combined_metrics.csv`, `accuracy_table.csv`",
            "- band-wise metrics: `band_metrics_table.csv`",
            "- coefficient-wise metrics: `coefficient_metrics_table.csv`",
            "- band x coefficient matrix: `band_coefficient_metrics_table.csv`",
            "- best/worst summaries: `best_worst_band_metrics.csv`, `best_worst_coefficient_metrics.csv`",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _build_stage_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    target_cols: List[str],
    preprocess_path: Path,
    scaler_path: Path,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, np.ndarray], StandardScaler]:
    preprocessor = build_preprocessor(train_df, numeric_cols=numeric_cols, categorical_cols=categorical_cols)
    preprocess_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, preprocess_path)

    x_train = _to_numpy(preprocessor.transform(train_df[categorical_cols + numeric_cols])).astype(np.float32)
    x_val = _to_numpy(preprocessor.transform(val_df[categorical_cols + numeric_cols])).astype(np.float32)
    x_test = _to_numpy(preprocessor.transform(test_df[categorical_cols + numeric_cols])).astype(np.float32)

    y_train_raw = train_df[target_cols].to_numpy(dtype=np.float32)
    y_val_raw = val_df[target_cols].to_numpy(dtype=np.float32)
    y_test_raw = test_df[target_cols].to_numpy(dtype=np.float32)
    y_scaler = _fit_target_scaler(y_train_raw)
    joblib.dump(y_scaler, scaler_path)

    split_data_scaled = {
        "train": {"x": x_train, "y": y_scaler.transform(y_train_raw).astype(np.float32)},
        "val": {"x": x_val, "y": y_scaler.transform(y_val_raw).astype(np.float32)},
        "test": {"x": x_test, "y": y_scaler.transform(y_test_raw).astype(np.float32)},
    }
    split_data_raw_y = {"train": y_train_raw, "val": y_val_raw, "test": y_test_raw}
    return split_data_scaled, split_data_raw_y, y_scaler


def _train_single_pipeline(
    model_name: str,
    stage_name: str,
    out_dir: Path,
    split_data_scaled: Dict[str, Dict[str, np.ndarray]],
    split_data_raw_y: Dict[str, np.ndarray],
    y_scaler: StandardScaler,
    metric_targets: List[str],
    args,
    device: torch.device,
    show_progress: bool,
) -> Tuple[Dict, Dict[str, np.ndarray], Optional[Dict], Dict]:
    if model_name in NEURAL_MODELS:
        metrics_scaled, preds_scaled, runtime_row, artifacts = _train_neural_stage(
            model_name=model_name,
            stage_name=stage_name,
            out_dir=out_dir,
            split_data_scaled=split_data_scaled,
            split_data_raw_y=split_data_raw_y,
            metric_targets=metric_targets,
            y_scaler=y_scaler,
            device=device,
            lr=args.lr,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            lambda_phys=args.lambda_phys,
            num_workers=args.num_workers,
            use_amp=args.amp,
            runtime_batch_size=args.runtime_batch_size,
            benchmark_runtime=args.benchmark_runtime,
            runtime_repeats=args.runtime_repeats,
            rtm_row_time_sec=_resolve_rtm_row_time(args),
            show_progress=show_progress,
            mlp_arch=args.mlp_arch,
            kan_arch=args.kan_arch,
            activation=args.activation,
            use_layernorm=bool(args.use_layernorm),
            deployable_stage_sizing=bool(args.deployable_stage_sizing),
        )
    else:
        metrics_scaled, preds_scaled, runtime_row, artifacts = _train_baseline_stage(
            model_name=model_name,
            stage_name=stage_name,
            out_dir=out_dir,
            split_data_scaled=split_data_scaled,
            split_data_raw_y=split_data_raw_y,
            metric_targets=metric_targets,
            seed=args.seed,
            num_workers=args.num_workers,
            benchmark_runtime=args.benchmark_runtime,
            runtime_repeats=args.runtime_repeats,
            runtime_batch_size=args.runtime_batch_size,
            rtm_row_time_sec=_resolve_rtm_row_time(args),
            xgb_trees=args.xgb_trees,
            rf_trees=args.rf_trees,
            gpu_arg=args.gpu,
            rf_progress_steps=args.rf_progress_steps,
            show_progress=show_progress,
        )
    preds_raw = {k: y_scaler.inverse_transform(v).astype(np.float32) for k, v in preds_scaled.items()}
    metrics = {k: compute_metrics(split_data_raw_y[k], preds_raw[k], targets=metric_targets) for k in ["train", "val", "test"]}
    save_json(metrics, out_dir / "metrics.json")
    pred_out = pd.concat(
        [
            pd.DataFrame(split_data_raw_y["test"], columns=[f"true_{t}" for t in metric_targets]),
            pd.DataFrame(preds_raw["test"], columns=[f"pred_{t}" for t in metric_targets]),
        ],
        axis=1,
    )
    pred_out.to_csv(out_dir / "predictions.csv", index=False)
    pred_out.to_csv(out_dir / "test_predictions.csv", index=False)
    save_scatter_plots(y_true=split_data_raw_y["test"], y_pred=preds_raw["test"], targets=metric_targets, out_dir=out_dir / "plots")
    save_residual_histograms(y_true=split_data_raw_y["test"], y_pred=preds_raw["test"], targets=metric_targets, out_dir=out_dir / "plots")
    if runtime_row is not None:
        runtime_row["stage"] = stage_name
    return metrics, preds_raw, runtime_row, artifacts


def main():
    parser = argparse.ArgumentParser(description="Train surrogate models from JSONL dataset.")
    parser.add_argument("--data", default=None, help="Single dataset JSONL path (compat mode).")
    parser.add_argument("--data_6s", default=None, help="6S dataset JSONL path.")
    parser.add_argument("--data_lrt", default=None, help="libRadtran dataset JSONL path.")
    parser.add_argument("--output_dir", required=True, help="Output run directory.")
    parser.add_argument("--fidelity_mode", default="single_fidelity_libradtran", choices=FIDELITY_MODES + COMBINED_FIDELITY_MODES)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", nargs="+", default=["mlp", "kan", "pmlp", "pkan"], choices=["mlp", "kan", "pmlp", "pkan", "rf", "xgb"])
    parser.add_argument(
        "--mlp_arch",
        default="baseline",
        choices=["baseline", "wider_gelu", "layernorm_gelu", "residual", "shared_trunk_multihead"],
        help="MLP architecture variant used by mlp/pmlp models.",
    )
    parser.add_argument(
        "--kan_arch",
        default="baseline",
        choices=["baseline", "small", "balanced_deep", "large_deep", "shared_trunk_multihead"],
        help="KAN architecture variant used by kan/pkan models.",
    )
    parser.add_argument("--activation", default="relu", choices=["relu", "gelu"], help="Activation used by MLP-based architectures.")
    parser.add_argument("--use_layernorm", action="store_true", help="Enable LayerNorm in MLP-based architectures.")
    parser.add_argument(
        "--deployable_stage_sizing",
        action="store_true",
        help="Use stronger stage1 and smaller stage2 neural architectures in deployable_residual mode.",
    )
    parser.add_argument("--lambda_phys", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu", default="cpu", help="GPU index like 0/1 or 'cpu'.")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--split_mode", default="standard", choices=["standard", "ood_aod_cwv", "both"])
    parser.add_argument("--ood_quantile", type=float, default=0.85)
    parser.add_argument("--benchmark_runtime", action="store_true")
    parser.add_argument("--runtime_repeats", type=int, default=5)
    parser.add_argument("--runtime_batch_size", type=int, default=8192)
    parser.add_argument("--rf_trees", type=int, default=150)
    parser.add_argument("--rf_progress_steps", type=int, default=None)
    parser.add_argument("--xgb_trees", type=int, default=200)
    parser.add_argument("--rtm_row_time_sec", type=float, default=None)
    parser.add_argument("--rtm_timing_json", type=str, default=None)
    parser.add_argument("--exclude_b10", action="store_true")
    parser.add_argument("--no_progress", action="store_true", help="Disable tqdm progress bars.")
    args = parser.parse_args()
    if args.benchmark_runtime or args.rtm_row_time_sec is not None or args.rtm_timing_json is not None:
        print(
            "[info] Runtime benchmarking during training is disabled. "
            "Use benchmark_runtime.py for all runtime/timing measurements."
        )
    # Force-disable in-training runtime timing to keep training and benchmarking separate.
    args.benchmark_runtime = False
    args.rtm_row_time_sec = None
    args.rtm_timing_json = None
    if args.rf_progress_steps is None:
        args.rf_progress_steps = args.epochs

    seed_everything(args.seed)
    device = select_device(args.gpu)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.fidelity_mode in {"single_fidelity_6s", "oracle_residual", "deployable_residual", "both_residual"} and args.data_6s is None and args.data is None:
        raise ValueError("This fidelity_mode requires --data_6s or --data.")
    if args.fidelity_mode in {"single_fidelity_libradtran"} and args.data_lrt is None and args.data is None:
        raise ValueError("This fidelity_mode requires --data_lrt or --data.")
    if args.fidelity_mode in {"oracle_residual", "deployable_residual", "both_residual"} and (args.data_6s is None or args.data_lrt is None):
        raise ValueError("Residual fidelity modes require both --data_6s and --data_lrt.")

    fidelity_modes_to_run = ["oracle_residual", "deployable_residual"] if args.fidelity_mode == "both_residual" else [args.fidelity_mode]
    split_modes = ["standard", "ood_aod_cwv"] if args.split_mode == "both" else [args.split_mode]
    mode_run_summaries = []

    for fidelity_mode in fidelity_modes_to_run:
        if fidelity_mode == "single_fidelity_6s":
            base_path = args.data_6s if args.data_6s is not None else args.data
            df, _ = _load_single_fidelity(base_path, exclude_b10=args.exclude_b10)
            feature_map = {c: c for c in CATEGORICAL_FEATURES + [n for n in NUMERIC_FEATURES if n in df.columns]}
            target_cols = TARGET_COLUMNS
            metric_targets = TARGET_COLUMNS
            merge_meta = {}
        elif fidelity_mode == "single_fidelity_libradtran":
            base_path = args.data_lrt if args.data_lrt is not None else args.data
            df, _ = _load_single_fidelity(base_path, exclude_b10=args.exclude_b10)
            feature_map = {c: c for c in CATEGORICAL_FEATURES + [n for n in NUMERIC_FEATURES if n in df.columns]}
            target_cols = TARGET_COLUMNS
            metric_targets = TARGET_COLUMNS
            merge_meta = {}
        else:
            df, merge_meta = _load_dual_fidelity(args.data_6s, args.data_lrt, exclude_b10=args.exclude_b10)
            feature_map = _resolve_merged_feature_columns(df)
            if fidelity_mode == "oracle_residual":
                for t in LOW_FIDELITY_TARGET_COLUMNS:
                    if t not in df.columns:
                        raise ValueError(f"Missing oracle feature column in merged data: {t}")
                # Oracle mode uses true 6S coefficients as additional inputs,
                # but trains/evaluates directly on full libRadtran targets.
                target_cols = HIGH_FIDELITY_TARGET_COLUMNS
                metric_targets = TARGET_COLUMNS
            else:
                target_cols = HIGH_FIDELITY_TARGET_COLUMNS
                metric_targets = TARGET_COLUMNS

        all_metric_rows: List[Dict] = []
        all_runtime_rows: List[Dict] = []
        all_band_metric_rows: List[Dict] = []
        all_coefficient_metric_rows: List[Dict] = []
        all_band_coefficient_metric_rows: List[Dict] = []
        split_summaries = []

        fidelity_root = out_dir / fidelity_mode
        fidelity_root.mkdir(parents=True, exist_ok=True)
        if merge_meta:
            save_json(merge_meta, fidelity_root / "merge_summary.json")

        for split_mode in split_modes:
            mode_dir = fidelity_root / "splits" / split_mode
            mode_dir.mkdir(parents=True, exist_ok=True)

            idx_map, split_assignment, split_meta = _get_split(
                df=df,
                split_mode=split_mode,
                seed=args.seed,
                ood_quantile=args.ood_quantile,
                aod_col=feature_map.get("aod550", "aod550"),
                cwv_col=feature_map.get("cwv_cm", "cwv_cm"),
            )
            leakage = validate_split_integrity(df=df, state_col="state_id", split_idx=idx_map)
            if any(v > 0 for v in leakage.values()):
                raise RuntimeError(f"State leakage detected for {split_mode}: {leakage}")
            split_assignment["split_mode"] = split_mode
            split_assignment.to_csv(mode_dir / "split_assignment.csv", index=False)
            split_summary = {"split_mode": split_mode, **split_meta, "rows_by_split": {k: int(len(v)) for k, v in idx_map.items()}, "states_by_split": split_assignment.groupby("split").size().to_dict(), "split_leakage": leakage}
            save_json(split_summary, mode_dir / "split_summary.json")
            split_summaries.append(split_summary)

            train_df = df.iloc[idx_map["train"]].copy()
            val_df = df.iloc[idx_map["val"]].copy()
            test_df = df.iloc[idx_map["test"]].copy()
            numeric_cols = [feature_map[c] for c in NUMERIC_FEATURES if c in feature_map]
            categorical_cols = [feature_map[c] for c in CATEGORICAL_FEATURES]

            if fidelity_mode == "oracle_residual":
                numeric_cols = numeric_cols + LOW_FIDELITY_TARGET_COLUMNS

            split_data_scaled, split_data_raw_y, y_scaler = _build_stage_data(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                target_cols=target_cols,
                preprocess_path=mode_dir / "preprocess" / "preprocessor.joblib",
                scaler_path=mode_dir / "preprocess" / "target_scaler.joblib",
            )

            model_iter = _progress(
                args.models,
                enabled=not args.no_progress,
                desc=f"{fidelity_mode}:{split_mode}",
                unit="model",
                leave=True,
            )
            for model_name in model_iter:
                model_dir = mode_dir / "models" / model_name
                if fidelity_mode != "deployable_residual":
                    metrics, preds, runtime_row, _ = _train_single_pipeline(
                        model_name=model_name,
                        stage_name="single_stage",
                        out_dir=model_dir,
                        split_data_scaled=split_data_scaled,
                        split_data_raw_y=split_data_raw_y,
                        y_scaler=y_scaler,
                        metric_targets=metric_targets,
                        args=args,
                        device=device,
                        show_progress=not args.no_progress,
                    )
                    split_truth_for_details = split_data_raw_y
                    targets_for_details = metric_targets
                else:
                    stage1_scaled, stage1_raw, stage1_scaler = _build_stage_data(
                        train_df=train_df,
                        val_df=val_df,
                        test_df=test_df,
                        numeric_cols=[feature_map[c] for c in NUMERIC_FEATURES if c in feature_map],
                        categorical_cols=categorical_cols,
                        target_cols=LOW_FIDELITY_TARGET_COLUMNS,
                        preprocess_path=model_dir / "stage1_6s" / "preprocessor.joblib",
                        scaler_path=model_dir / "stage1_6s" / "target_scaler.joblib",
                    )
                    stage1_metrics, stage1_preds, _, _ = _train_single_pipeline(
                        model_name=model_name,
                        stage_name="stage1_6s",
                        out_dir=model_dir / "stage1_6s",
                        split_data_scaled=stage1_scaled,
                        split_data_raw_y=stage1_raw,
                        y_scaler=stage1_scaler,
                        metric_targets=LOW_FIDELITY_TARGET_COLUMNS,
                        args=args,
                        device=device,
                        show_progress=not args.no_progress,
                    )
                    save_json(stage1_metrics, model_dir / "stage1_6s" / "metrics.json")

                    train_df_stage2 = train_df.copy()
                    val_df_stage2 = val_df.copy()
                    test_df_stage2 = test_df.copy()
                    for i, t in enumerate(LOW_FIDELITY_TARGET_COLUMNS):
                        col = f"pred_{t}"
                        train_df_stage2[col] = stage1_preds["train"][:, i]
                        val_df_stage2[col] = stage1_preds["val"][:, i]
                        test_df_stage2[col] = stage1_preds["test"][:, i]

                    stage2_numeric = [feature_map[c] for c in NUMERIC_FEATURES if c in feature_map] + [f"pred_{t}" for t in LOW_FIDELITY_TARGET_COLUMNS]
                    stage2_scaled, stage2_raw, stage2_scaler = _build_stage_data(
                        train_df=train_df_stage2,
                        val_df=val_df_stage2,
                        test_df=test_df_stage2,
                        numeric_cols=stage2_numeric,
                        categorical_cols=categorical_cols,
                        target_cols=RESIDUAL_TARGET_COLUMNS,
                        preprocess_path=model_dir / "stage2_residual" / "preprocessor.joblib",
                        scaler_path=model_dir / "stage2_residual" / "target_scaler.joblib",
                    )
                    stage2_metrics, stage2_preds, stage2_runtime, _ = _train_single_pipeline(
                        model_name=model_name,
                        stage_name="stage2_residual",
                        out_dir=model_dir / "stage2_residual",
                        split_data_scaled=stage2_scaled,
                        split_data_raw_y=stage2_raw,
                        y_scaler=stage2_scaler,
                        metric_targets=RESIDUAL_TARGET_COLUMNS,
                        args=args,
                        device=device,
                        show_progress=not args.no_progress,
                    )
                    save_json(stage2_metrics, model_dir / "stage2_residual" / "metrics.json")

                    final_preds = {}
                    metrics = {}
                    split_truth_for_details = {}
                    for split_name in ["train", "val", "test"]:
                        low_pred = stage1_preds[split_name]
                        residual_pred = stage2_preds[split_name]
                        final = low_pred + residual_pred
                        final_preds[split_name] = final.astype(np.float32)
                        y_true = df.iloc[idx_map[split_name]][HIGH_FIDELITY_TARGET_COLUMNS].to_numpy(dtype=np.float32)
                        split_truth_for_details[split_name] = y_true
                        metrics[split_name] = compute_metrics(y_true=y_true, y_pred=final_preds[split_name], targets=TARGET_COLUMNS)

                    test_preds = final_preds["test"]
                    y_test_metric = test_df[HIGH_FIDELITY_TARGET_COLUMNS].to_numpy(dtype=np.float32)
                    pred_df = pd.DataFrame(test_preds, columns=[f"pred_{t}" for t in TARGET_COLUMNS])
                    true_df = pd.DataFrame(y_test_metric, columns=[f"true_{t}" for t in TARGET_COLUMNS])
                    pred_out = pd.concat([true_df, pred_df], axis=1)
                    pred_out.to_csv(model_dir / "predictions.csv", index=False)
                    pred_out.to_csv(model_dir / "test_predictions.csv", index=False)
                    save_scatter_plots(y_true=y_test_metric, y_pred=test_preds, targets=TARGET_COLUMNS, out_dir=model_dir / "plots")
                    save_residual_histograms(y_true=y_test_metric, y_pred=test_preds, targets=TARGET_COLUMNS, out_dir=model_dir / "plots")
                    save_json(metrics, model_dir / "metrics.json")

                    runtime_row = None
                    if stage2_runtime is not None:
                        runtime_row = {"model": model_name, "stage": "deployable_residual", **{k: v for k, v in stage2_runtime.items() if k != "stage"}}
                    preds = final_preds
                    targets_for_details = TARGET_COLUMNS

                all_metric_rows.extend(_rows_from_metrics(model_name=model_name, split_mode=split_mode, fidelity_mode=fidelity_mode, metrics=metrics))
                if runtime_row is not None:
                    runtime_row["split_mode"] = split_mode
                    runtime_row["fidelity_mode"] = fidelity_mode
                    all_runtime_rows.append(runtime_row)
                split_frames = {"train": train_df, "val": val_df, "test": test_df}
                band_metrics_df, coefficient_metrics_df, band_coefficient_metrics_df = _build_detailed_error_tables(
                    split_frames=split_frames,
                    split_preds=preds,
                    split_truth=split_truth_for_details,
                    targets=targets_for_details,
                    band_col=feature_map["band"],
                )
                band_metrics_df.to_csv(model_dir / "per_band_metrics.csv", index=False)
                coefficient_metrics_df.to_csv(model_dir / "per_coefficient_metrics.csv", index=False)
                band_coefficient_metrics_df.to_csv(model_dir / "per_band_coefficient_metrics.csv", index=False)

                for row_df, collector in [
                    (band_metrics_df, all_band_metric_rows),
                    (coefficient_metrics_df, all_coefficient_metric_rows),
                    (band_coefficient_metrics_df, all_band_coefficient_metric_rows),
                ]:
                    if row_df.empty:
                        continue
                    row_df = row_df.copy()
                    row_df["model"] = model_name
                    row_df["split_mode"] = split_mode
                    row_df["fidelity_mode"] = fidelity_mode
                    collector.extend(row_df.to_dict(orient="records"))

            mode_metrics_df = pd.DataFrame([r for r in all_metric_rows if r["split_mode"] == split_mode and r["fidelity_mode"] == fidelity_mode]).sort_values(["model", "split", "target"])
            mode_metrics_df.to_csv(mode_dir / "combined_metrics.csv", index=False)
            mode_band_metrics_df = pd.DataFrame([r for r in all_band_metric_rows if r["split_mode"] == split_mode and r["fidelity_mode"] == fidelity_mode]).sort_values(["model", "split", "band"])
            mode_coefficient_metrics_df = pd.DataFrame([r for r in all_coefficient_metric_rows if r["split_mode"] == split_mode and r["fidelity_mode"] == fidelity_mode]).sort_values(["model", "split", "coefficient"])
            mode_band_coefficient_metrics_df = pd.DataFrame(
                [r for r in all_band_coefficient_metric_rows if r["split_mode"] == split_mode and r["fidelity_mode"] == fidelity_mode]
            ).sort_values(["model", "split", "band", "coefficient"])
            mode_band_metrics_df.to_csv(mode_dir / "band_metrics_table.csv", index=False)
            mode_coefficient_metrics_df.to_csv(mode_dir / "coefficient_metrics_table.csv", index=False)
            mode_band_coefficient_metrics_df.to_csv(mode_dir / "band_coefficient_metrics_table.csv", index=False)
            mode_best_worst_band_df = _build_best_worst_table(mode_band_metrics_df[mode_band_metrics_df["split"] == "test"], entity_col="band")
            mode_best_worst_coefficient_df = _build_best_worst_table(
                mode_coefficient_metrics_df[mode_coefficient_metrics_df["split"] == "test"], entity_col="coefficient"
            )
            mode_best_worst_band_df.to_csv(mode_dir / "best_worst_band_metrics.csv", index=False)
            mode_best_worst_coefficient_df.to_csv(mode_dir / "best_worst_coefficient_metrics.csv", index=False)
            mode_runtime_df = pd.DataFrame([r for r in all_runtime_rows if r["split_mode"] == split_mode and r["fidelity_mode"] == fidelity_mode])
            if not mode_runtime_df.empty:
                mode_runtime_df = mode_runtime_df.sort_values("avg_test_inference_sec")
                mode_runtime_df.to_csv(mode_dir / "runtime_table.csv", index=False)
                save_json(mode_runtime_df.to_dict(orient="records"), mode_dir / "runtime_table.json")
            _build_mode_report(
                mode_metrics_df,
                mode_runtime_df,
                mode_band_metrics_df,
                mode_coefficient_metrics_df,
                mode_band_coefficient_metrics_df,
                mode_dir / "model_comparison.md",
                split_mode=split_mode,
                fidelity_mode=fidelity_mode,
            )

        all_metrics_df = pd.DataFrame(all_metric_rows).sort_values(["fidelity_mode", "split_mode", "model", "split", "target"]).reset_index(drop=True)
        all_metrics_df.to_csv(fidelity_root / "combined_metrics.csv", index=False)
        all_band_metrics_df = pd.DataFrame(all_band_metric_rows).sort_values(["fidelity_mode", "split_mode", "model", "split", "band"]).reset_index(drop=True)
        all_coefficient_metrics_df = pd.DataFrame(all_coefficient_metric_rows).sort_values(["fidelity_mode", "split_mode", "model", "split", "coefficient"]).reset_index(drop=True)
        all_band_coefficient_metrics_df = pd.DataFrame(all_band_coefficient_metric_rows).sort_values(
            ["fidelity_mode", "split_mode", "model", "split", "band", "coefficient"]
        ).reset_index(drop=True)
        all_band_metrics_df.to_csv(fidelity_root / "band_metrics_table.csv", index=False)
        all_coefficient_metrics_df.to_csv(fidelity_root / "coefficient_metrics_table.csv", index=False)
        all_band_coefficient_metrics_df.to_csv(fidelity_root / "band_coefficient_metrics_table.csv", index=False)
        _build_best_worst_table(all_band_metrics_df[all_band_metrics_df["split"] == "test"], entity_col="band").to_csv(
            fidelity_root / "best_worst_band_metrics.csv", index=False
        )
        _build_best_worst_table(
            all_coefficient_metrics_df[all_coefficient_metrics_df["split"] == "test"], entity_col="coefficient"
        ).to_csv(fidelity_root / "best_worst_coefficient_metrics.csv", index=False)

        accuracy_table = all_metrics_df[(all_metrics_df["split"] == "test") & (all_metrics_df["target"] == "overall")][
            ["fidelity_mode", "split_mode", "model", "rmse", "mae", "r2", "mape", "smape"]
        ].sort_values(["split_mode", "rmse"])
        accuracy_table.to_csv(fidelity_root / "accuracy_table.csv", index=False)
        save_json(split_summaries, fidelity_root / "split_summaries.json")

        runtime_table = pd.DataFrame(all_runtime_rows)
        if not runtime_table.empty:
            runtime_table = runtime_table.sort_values(["split_mode", "avg_test_inference_sec"]).reset_index(drop=True)
            runtime_table.to_csv(fidelity_root / "runtime_table.csv", index=False)
            save_json(runtime_table.to_dict(orient="records"), fidelity_root / "runtime_table.json")

        accuracy_runtime = pd.DataFrame()
        if not runtime_table.empty and not accuracy_table.empty:
            accuracy_runtime = accuracy_table.merge(runtime_table, on=["fidelity_mode", "split_mode", "model"], how="left")
            accuracy_runtime.to_csv(fidelity_root / "accuracy_runtime_summary.csv", index=False)

        plots_dir = fidelity_root / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        save_rmse_bar(accuracy_table, plots_dir / "rmse_by_model_and_split.png")
        save_runtime_bar(runtime_table, plots_dir / "runtime_by_model.png")
        save_accuracy_runtime_scatter(accuracy_runtime, plots_dir / "accuracy_vs_runtime.png")
        _build_global_report(accuracy_df=accuracy_table, runtime_df=runtime_table, accuracy_runtime_df=accuracy_runtime, out_path=fidelity_root / "model_comparison.md", fidelity_mode=fidelity_mode)

        run_config = {
            "data": args.data,
            "data_6s": args.data_6s,
            "data_lrt": args.data_lrt,
            "output_dir": str(out_dir.resolve()),
            "fidelity_mode": fidelity_mode,
            "requested_fidelity_mode": args.fidelity_mode,
            "resolved_fidelity_modes": fidelity_modes_to_run,
            "exclude_b10": bool(args.exclude_b10),
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "seed": args.seed,
            "models": args.models,
            "mlp_arch": args.mlp_arch,
            "kan_arch": args.kan_arch,
            "activation": args.activation,
            "use_layernorm": bool(args.use_layernorm),
            "deployable_stage_sizing": bool(args.deployable_stage_sizing),
            "lambda_phys": args.lambda_phys,
            "num_workers": args.num_workers,
            "gpu": args.gpu,
            "amp": bool(args.amp),
            "patience": args.patience,
            "split_mode": args.split_mode,
            "ood_quantile": args.ood_quantile,
            "benchmark_runtime": False,
            "runtime_repeats": None,
            "runtime_batch_size": None,
            "rtm_row_time_sec": None,
            "rf_trees": args.rf_trees,
            "rf_progress_steps": args.rf_progress_steps,
            "xgb_trees": args.xgb_trees,
            "no_progress": bool(args.no_progress),
        }
        save_json(run_config, fidelity_root / "run_config.json")
        mode_run_summaries.append({"fidelity_mode": fidelity_mode, "output_dir": str(fidelity_root.resolve())})
        print(f"Training run complete for '{fidelity_mode}'. Outputs saved to: {fidelity_root.resolve()}")

    if len(mode_run_summaries) > 1:
        save_json(mode_run_summaries, out_dir / "multi_fidelity_run_summary.json")


if __name__ == "__main__":
    main()
