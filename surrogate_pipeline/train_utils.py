import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import r2_score

matplotlib.use("Agg")
PERCENT_METRIC_EPS = 1e-6


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device(gpu_arg: str) -> torch.device:
    if str(gpu_arg).lower() == "cpu":
        return torch.device("cpu")
    try:
        gpu_index = int(gpu_arg)
    except ValueError as exc:
        raise ValueError("--gpu must be 'cpu' or an integer GPU index.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available but GPU index was provided.")
    device_count = torch.cuda.device_count()
    if gpu_index < 0 or gpu_index >= device_count:
        raise RuntimeError(f"Requested GPU index {gpu_index} but available device count is {device_count}.")
    return torch.device(f"cuda:{gpu_index}")


def _canonical_target_name(name: str) -> Optional[str]:
    n = str(name).strip().lower()
    if n.startswith("delta_"):
        return None
    for suffix in ("_6s", "_lrt"):
        if n.endswith(suffix):
            n = n[: -len(suffix)]
    return n


def physics_penalty(pred: torch.Tensor, target_names: Optional[List[str]] = None) -> torch.Tensor:
    if target_names is None:
        rho_path = pred[:, 0]
        t_total = pred[:, 1]
        spher_alb = pred[:, 2]
        penalties = torch.relu(-rho_path) + torch.relu(-t_total) + torch.relu(t_total - 1.0) + torch.relu(-spher_alb)
        return penalties.mean()

    penalties = torch.zeros(pred.shape[0], device=pred.device, dtype=pred.dtype)
    constrained_dims = 0
    for idx, name in enumerate(target_names):
        canonical = _canonical_target_name(name)
        if canonical == "rho_path":
            penalties = penalties + torch.relu(-pred[:, idx])
            constrained_dims += 1
        elif canonical == "t_total":
            penalties = penalties + torch.relu(-pred[:, idx]) + torch.relu(pred[:, idx] - 1.0)
            constrained_dims += 1
        elif canonical == "spher_alb":
            penalties = penalties + torch.relu(-pred[:, idx])
            constrained_dims += 1

    if constrained_dims == 0:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    return penalties.mean()


def compute_error_metrics(y_true: np.ndarray, y_pred: np.ndarray, eps: float = PERCENT_METRIC_EPS) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch for metric computation: y_true={y_true.shape}, y_pred={y_pred.shape}")

    abs_err = np.abs(y_pred - y_true)
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae = float(np.mean(abs_err))

    denom_mape = np.maximum(np.abs(y_true), eps)
    mape = float(np.mean(abs_err / denom_mape) * 100.0)

    denom_smape = np.maximum(np.abs(y_pred) + np.abs(y_true), eps)
    smape = float(np.mean((2.0 * abs_err) / denom_smape) * 100.0)

    if y_true.ndim == 1:
        r2 = float(r2_score(y_true, y_pred))
    else:
        r2 = float(r2_score(y_true, y_pred, multioutput="uniform_average"))
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape, "smape": smape}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, targets: List[str]) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for i, target in enumerate(targets):
        metrics[target] = compute_error_metrics(y_true[:, i], y_pred[:, i], eps=PERCENT_METRIC_EPS)
    metrics["overall"] = compute_error_metrics(y_true, y_pred, eps=PERCENT_METRIC_EPS)
    return metrics


def save_loss_curve(history_df, path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="train")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_scatter_plots(y_true: np.ndarray, y_pred: np.ndarray, targets: List[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, target in enumerate(targets):
        plt.figure(figsize=(6, 6))
        plt.scatter(y_true[:, i], y_pred[:, i], alpha=0.25, s=10)
        min_v = min(float(y_true[:, i].min()), float(y_pred[:, i].min()))
        max_v = max(float(y_true[:, i].max()), float(y_pred[:, i].max()))
        plt.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=1.0)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f"Predicted vs Actual: {target}")
        plt.tight_layout()
        plt.savefig(out_dir / f"scatter_{target}.png", dpi=170)
        plt.close()


def save_residual_histograms(y_true: np.ndarray, y_pred: np.ndarray, targets: List[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, target in enumerate(targets):
        residuals = y_pred[:, i] - y_true[:, i]
        plt.figure(figsize=(7, 5))
        plt.hist(residuals, bins=60, alpha=0.8)
        plt.xlabel("Residual")
        plt.ylabel("Frequency")
        plt.title(f"Residual Distribution: {target}")
        plt.tight_layout()
        plt.savefig(out_dir / f"residual_hist_{target}.png", dpi=170)
        plt.close()


def save_json(data, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_runtime_bar(runtime_df, out_path: Path) -> None:
    if runtime_df.empty:
        return
    data = runtime_df.sort_values("avg_test_inference_sec")
    plt.figure(figsize=(9, 5))
    plt.bar(data["model"], data["avg_test_inference_sec"])
    plt.ylabel("Average Test Inference Time (s)")
    plt.xlabel("Model")
    plt.title("Inference Runtime by Model")
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def save_rmse_bar(accuracy_df, out_path: Path) -> None:
    if accuracy_df.empty:
        return
    data = accuracy_df.sort_values(["split_mode", "rmse"])
    labels = [f"{m}:{s}" for m, s in zip(data["model"], data["split_mode"])]
    plt.figure(figsize=(12, 5))
    plt.bar(labels, data["rmse"])
    plt.ylabel("Test Overall RMSE")
    plt.xlabel("Model:Split")
    plt.title("Overall RMSE by Model and Split")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def save_accuracy_runtime_scatter(df, out_path: Path) -> None:
    if df.empty:
        return
    plt.figure(figsize=(7, 5))
    for split_mode, sub in df.groupby("split_mode"):
        plt.scatter(sub["avg_test_inference_sec"], sub["rmse"], label=split_mode, s=55, alpha=0.85)
        for _, row in sub.iterrows():
            plt.annotate(str(row["model"]), (row["avg_test_inference_sec"], row["rmse"]), fontsize=8)
    plt.xlabel("Average Test Inference Time (s)")
    plt.ylabel("Test Overall RMSE")
    plt.title("Accuracy vs Runtime")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()
