import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from surrogate_pipeline.data_utils import (
    CATEGORICAL_FEATURES,
    LOW_FIDELITY_TARGET_COLUMNS,
    NUMERIC_FEATURES,
)
from surrogate_pipeline.models import build_model
from train_surrogates import _build_stage_data, _get_split, _load_dual_fidelity, _resolve_merged_feature_columns


def _count_trainable_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _prepare_shared_dims(data_6s: str, data_lrt: str, split_mode: str, seed: int, exclude_b10: bool) -> Tuple[int, int, Dict]:
    df, _ = _load_dual_fidelity(data_6s, data_lrt, exclude_b10=exclude_b10)
    feature_map = _resolve_merged_feature_columns(df)
    idx_map, _, split_meta = _get_split(df=df, split_mode=split_mode, seed=seed, ood_quantile=0.85)
    train_df = df.iloc[idx_map["train"]].copy()
    val_df = df.iloc[idx_map["val"]].copy()
    test_df = df.iloc[idx_map["test"]].copy()

    numeric_cols = [feature_map[c] for c in NUMERIC_FEATURES if c in feature_map] + LOW_FIDELITY_TARGET_COLUMNS
    categorical_cols = [feature_map[c] for c in CATEGORICAL_FEATURES]
    target_cols = [f"{c}_lrt" for c in ["rho_path", "T_total", "spher_alb"]]

    preproc_tmp = Path("results/reviewer_kan_complexity/_tmp/preprocessor.joblib")
    scaler_tmp = Path("results/reviewer_kan_complexity/_tmp/target_scaler.joblib")
    split_data_scaled, _, _ = _build_stage_data(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        target_cols=target_cols,
        preprocess_path=preproc_tmp,
        scaler_path=scaler_tmp,
    )
    in_dim = int(split_data_scaled["train"]["x"].shape[1])
    out_dim = int(split_data_scaled["train"]["y"].shape[1])
    return in_dim, out_dim, split_meta


def _build_train_command(args, model_name: str, mlp_arch: str, kan_arch: str, out_dir: Path, epochs: int) -> List[str]:
    cmd = [
        "python",
        "train_surrogates.py",
        "--data_6s",
        args.data_6s,
        "--data_lrt",
        args.data_lrt,
        "--output_dir",
        str(out_dir),
        "--fidelity_mode",
        "oracle_residual",
        "--models",
        model_name,
        "--split_mode",
        args.split_mode,
        "--epochs",
        str(epochs),
        "--batch_size",
        str(args.batch_size),
        "--patience",
        str(args.patience),
        "--seed",
        str(args.seed),
        "--gpu",
        str(args.gpu),
        "--num_workers",
        str(args.num_workers),
        "--mlp_arch",
        mlp_arch,
        "--kan_arch",
        kan_arch,
        "--lambda_phys",
        str(args.lambda_phys),
        "--no_progress",
    ]
    if args.amp:
        cmd.append("--amp")
    if args.exclude_b10:
        cmd.append("--exclude_b10")
    return cmd


def _warmup(args, warmup_dir: Path) -> float:
    """Run a 1-epoch dummy training to warm OS page cache, CUDA driver, cuDNN, AMP."""
    if warmup_dir.exists():
        shutil.rmtree(warmup_dir, ignore_errors=True)
    warmup_dir.mkdir(parents=True, exist_ok=True)
    cmd = _build_train_command(
        args=args,
        model_name="mlp",
        mlp_arch="baseline",
        kan_arch="baseline",
        out_dir=warmup_dir,
        epochs=1,
    )
    print(f"[warmup] Running 1-epoch dummy training to warm caches...")
    print(f"[warmup] cmd: {' '.join(cmd)}")
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True)
    elapsed = float(time.perf_counter() - t0)
    print(f"[warmup] Done in {elapsed:.1f} s (this time is discarded).")
    return elapsed


def _collect_run_metrics(run_root: Path, model_name: str) -> Tuple[Dict, pd.DataFrame]:
    model_dir = run_root / "oracle_residual" / "splits" / "standard" / "models" / model_name
    metrics = _load_json(model_dir / "metrics.json")
    history = pd.read_csv(model_dir / "history.csv")
    return metrics, history


def _latex_table(df: pd.DataFrame) -> str:
    latex_df = df.copy()
    latex = latex_df.to_latex(index=False, escape=True)
    return "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            "\\resizebox{\\linewidth}{!}{%",
            latex,
            "}",
            "\\end{table}",
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compact reviewer experiment: KAN complexity / parameter efficiency / convergence table.\n"
                    "Performs a warm-up dummy run first so that the four measured runs share warm OS / CUDA caches."
    )
    parser.add_argument("--data_6s", default="data/qavalid_intersection_libradtran_6s_50k_13b/dataset_rows_6s.jsonl")
    parser.add_argument("--data_lrt", default="data/qavalid_intersection_libradtran_6s_50k_13b/dataset_rows_libradtran.jsonl")
    parser.add_argument("--split_mode", default="standard", choices=["standard"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--exclude_b10", action="store_true")
    parser.add_argument("--lambda_phys", type=float, default=1.0)
    parser.add_argument("--skip_warmup", action="store_true",
                        help="Skip the warm-up dummy run (warm-up is ON by default).")
    parser.add_argument("--reuse_artifacts", action="store_true",
                        help="Reuse existing per-model artifacts if present. Default behavior is to retrain so timings are warm.")
    args = parser.parse_args()

    output_root = Path("results/reviewer_kan_complexity")
    output_root.mkdir(parents=True, exist_ok=True)
    runs_root = output_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    in_dim, out_dim, split_meta = _prepare_shared_dims(
        data_6s=args.data_6s,
        data_lrt=args.data_lrt,
        split_mode=args.split_mode,
        seed=args.seed,
        exclude_b10=args.exclude_b10,
    )

    warmup_elapsed = None
    if not args.skip_warmup:
        warmup_dir = runs_root / "_warmup"
        warmup_elapsed = _warmup(args, warmup_dir)

    configs = [
        {"run_key": "fc_base", "model_key": "mlp", "mlp_arch": "baseline", "kan_arch": "baseline",
         "model_label": "FC-base", "arch_str": "MLP([d_in, 256, 128, d_out])"},
        {"run_key": "fc_wide", "model_key": "mlp", "mlp_arch": "wider_gelu", "kan_arch": "baseline",
         "model_label": "FC-wide", "arch_str": "MLP([d_in, 512, 256, 128, d_out]), GELU"},
        {"run_key": "kan_residual_no_phys", "model_key": "kan", "mlp_arch": "baseline", "kan_arch": "baseline",
         "model_label": "Residual KAN", "arch_str": "KAN([d_in, 256, 128, d_out]), lambda_phys=0"},
        {"run_key": "pkan_main_phys", "model_key": "pkan", "mlp_arch": "baseline", "kan_arch": "baseline",
         "model_label": "pKANrtm", "arch_str": "KAN([d_in, 256, 128, d_out]), lambda_phys=1"},
    ]

    assumptions: List[str] = []
    if not args.skip_warmup:
        assumptions.append(
            f"A 1-epoch warm-up dummy run was executed before the timed runs to populate the OS page cache "
            f"(~1 GB of JSONL) and the CUDA driver state. Warm-up wall time = {warmup_elapsed:.1f} s (excluded)."
        )
    else:
        assumptions.append("Warm-up was skipped (--skip_warmup). Timings may include cold-cache overhead on the first run.")

    rows: List[Dict] = []

    for cfg in configs:
        run_dir = runs_root / cfg["run_key"]
        model_dir = run_dir / "oracle_residual" / "splits" / "standard" / "models" / cfg["model_key"]
        history_path = model_dir / "history.csv"
        metrics_path = model_dir / "metrics.json"
        runtime_meta_path = model_dir / "reviewer_runtime_meta.json"

        reused = args.reuse_artifacts and history_path.exists() and metrics_path.exists()
        total_train_sec = None

        if not reused:
            cmd = _build_train_command(
                args=args,
                model_name=cfg["model_key"],
                mlp_arch=cfg["mlp_arch"],
                kan_arch=cfg["kan_arch"],
                out_dir=run_dir,
                epochs=args.epochs,
            )
            t0 = time.perf_counter()
            subprocess.run(cmd, check=True)
            total_train_sec = float(time.perf_counter() - t0)
            _save_json(runtime_meta_path, {"total_train_time_sec": total_train_sec, "command": cmd,
                                            "warmup_done": not args.skip_warmup})
            assumptions.append(
                f"Trained `{cfg['run_key']}` ({cfg['model_label']}) with epochs={args.epochs}, "
                f"patience={args.patience}, batch_size={args.batch_size}."
            )
        else:
            assumptions.append(f"Reused existing artifacts for `{cfg['run_key']}` ({cfg['model_label']}).")
            if runtime_meta_path.exists():
                total_train_sec = float(_load_json(runtime_meta_path).get("total_train_time_sec"))

        metrics, history = _collect_run_metrics(run_dir, cfg["model_key"])
        best_idx = int(history["val_loss"].idxmin())
        best_val_epoch = int(history.loc[best_idx, "epoch"])
        best_val_loss = float(history.loc[best_idx, "val_loss"])
        epochs_ran = int(len(history))
        avg_epoch_time = float(total_train_sec / epochs_ran) if (total_train_sec is not None and epochs_ran > 0) else None

        model_obj, _ = build_model(
            model_name=cfg["model_key"],
            in_dim=in_dim,
            out_dim=out_dim,
            stage_name="single_stage",
            mlp_arch=cfg["mlp_arch"],
            kan_arch=cfg["kan_arch"],
            activation="relu",
            use_layernorm=False,
            deployable_stage_sizing=False,
        )
        n_params = _count_trainable_params(model_obj)

        rows.append({
            "Model": cfg["model_label"],
            "Architecture": cfg["arch_str"],
            "Trainable parameters": n_params,
            "Best validation epoch": best_val_epoch,
            "Best validation loss": best_val_loss,
            "Test RMSE": float(metrics["test"]["overall"]["rmse"]),
            "Test MAE": float(metrics["test"]["overall"]["mae"]),
            "Test R2": float(metrics["test"]["overall"]["r2"]),
            "Test SMAPE (%)": float(metrics["test"]["overall"]["smape"]),
            "Total training time (s)": total_train_sec,
            "Average epoch time (s)": avg_epoch_time,
        })

    df = pd.DataFrame(rows)
    df = df[
        [
            "Model",
            "Architecture",
            "Trainable parameters",
            "Best validation epoch",
            "Best validation loss",
            "Test RMSE",
            "Test MAE",
            "Test R2",
            "Test SMAPE (%)",
            "Total training time (s)",
            "Average epoch time (s)",
        ]
    ]

    csv_path = output_root / "kan_complexity_table.csv"
    json_path = output_root / "kan_complexity_results.json"
    tex_path = output_root / "kan_complexity_table.tex"
    md_path = output_root / "kan_complexity_summary.md"

    df.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "split_mode": args.split_mode,
                "split_meta": split_meta,
                "data_6s": args.data_6s,
                "data_lrt": args.data_lrt,
                "preprocessed_input_dim": in_dim,
                "output_dim": out_dim,
                "epochs_budget": args.epochs,
                "patience": args.patience,
                "warmup_done": not args.skip_warmup,
                "warmup_wall_time_sec": warmup_elapsed,
                "val_loss_definition": "MSE on standardized target space (logged by training loop)",
                "results": df.to_dict(orient="records"),
                "assumptions": assumptions,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    tex_path.write_text(_latex_table(df), encoding="utf-8")

    md_lines = [
        "# Reviewer KAN Complexity Check",
        "",
        f"- Split mode: `{args.split_mode}`",
        f"- Data 6S: `{args.data_6s}`",
        f"- Data libRadtran: `{args.data_lrt}`",
        f"- Preprocessed input dim `d_in = {in_dim}`, output dim `d_out = {out_dim}`",
        f"- Epoch budget = {args.epochs}, patience = {args.patience}",
        f"- Warm-up dummy run executed: {'yes' if not args.skip_warmup else 'no'}"
        + (f" (warm-up wall time = {warmup_elapsed:.1f} s, excluded)" if warmup_elapsed is not None else ""),
        "",
        "## Results",
        "",
        df.to_markdown(index=False),
        "",
        "## Assumptions",
        "",
    ] + [f"- {a}" for a in assumptions]
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Saved CSV: {csv_path.resolve()}")
    print(f"Saved JSON: {json_path.resolve()}")
    print(f"Saved TeX: {tex_path.resolve()}")
    print(f"Saved Markdown: {md_path.resolve()}")
    print("")
    print(df.to_markdown(index=False))
    print("")
    print("Assumptions:")
    for item in assumptions:
        print(f"- {item}")


if __name__ == "__main__":
    main()
