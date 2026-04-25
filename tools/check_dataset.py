import argparse
import json
from pathlib import Path

import pandas as pd

from surrogate_pipeline.data_utils import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMNS,
    infer_expected_rows_per_state,
    load_jsonl,
    normalize_split_values,
    resolve_column_names,
)


def _as_int(value) -> int:
    return int(value) if value is not None else 0


def main():
    parser = argparse.ArgumentParser(description="Sanity-check surrogate dataset JSONL.")
    parser.add_argument("--data", required=True, help="Path to dataset JSONL.")
    parser.add_argument("--output_dir", default="runs/dataset_check", help="Output directory for reports.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_jsonl(args.data)
    required_cols = CATEGORICAL_FEATURES + NUMERIC_FEATURES + TARGET_COLUMNS
    resolved = resolve_column_names(df, required=required_cols, optional=["state_id", "split", "qa_valid"])

    mapped = {}
    for logical_name, actual_name in resolved.items():
        if actual_name is not None:
            mapped[logical_name] = actual_name

    state_col = resolved["state_id"]
    split_col = resolved["split"]
    qa_col = resolved["qa_valid"]
    band_col = resolved["band"]

    rows_per_split = pd.DataFrame()
    unique_states_per_split = pd.DataFrame()
    split_valid = False
    split_leakage = {
        "train_val_overlap": None,
        "train_test_overlap": None,
        "val_test_overlap": None,
    }

    if split_col is not None:
        normalized_split = normalize_split_values(df[split_col])
        split_valid = normalized_split.notna().all()
        if split_valid:
            rows_per_split = normalized_split.value_counts().rename_axis("split").reset_index(name="rows")
            if state_col is not None:
                tmp = pd.DataFrame({"split": normalized_split, "state_id": df[state_col].astype(str)})
                unique_states_per_split = (
                    tmp.drop_duplicates().groupby("split")["state_id"].nunique().rename("unique_states").reset_index()
                )
                state_split = tmp.drop_duplicates()
                state_groups = state_split.groupby("state_id")["split"].nunique()
                if (state_groups == 1).all():
                    state_to_split = state_split.groupby("state_id")["split"].first()
                    train_states = set(state_to_split[state_to_split == "train"].index)
                    val_states = set(state_to_split[state_to_split == "val"].index)
                    test_states = set(state_to_split[state_to_split == "test"].index)
                    split_leakage = {
                        "train_val_overlap": len(train_states & val_states),
                        "train_test_overlap": len(train_states & test_states),
                        "val_test_overlap": len(val_states & test_states),
                    }

    rows_per_band = df[band_col].value_counts().rename_axis("band").reset_index(name="rows")

    qa_false_count = _as_int((~df[qa_col].astype(bool)).sum()) if qa_col is not None else None
    monotonicity_failed_count = _as_int(df["monotonicity_failed"].astype(bool).sum()) if "monotonicity_failed" in df.columns else None
    clamp_counts = {
        "T_total_was_clamped_true_count": _as_int(df["T_total_was_clamped"].astype(bool).sum()) if "T_total_was_clamped" in df.columns else None,
        "rho_path_was_clamped_true_count": _as_int(df["rho_path_was_clamped"].astype(bool).sum()) if "rho_path_was_clamped" in df.columns else None,
        "spher_alb_was_clamped_true_count": _as_int(df["spher_alb_was_clamped"].astype(bool).sum()) if "spher_alb_was_clamped" in df.columns else None,
    }

    numeric_columns = [resolved[c] for c in NUMERIC_FEATURES + TARGET_COLUMNS]
    numeric_stats = []
    for logical_name, actual_col in zip(NUMERIC_FEATURES + TARGET_COLUMNS, numeric_columns):
        series = pd.to_numeric(df[actual_col], errors="coerce")
        numeric_stats.append(
            {
                "logical_column": logical_name,
                "actual_column": actual_col,
                "count": int(series.notna().sum()),
                "min": float(series.min()),
                "max": float(series.max()),
                "mean": float(series.mean()),
                "std": float(series.std(ddof=0)),
            }
        )
    numeric_stats_df = pd.DataFrame(numeric_stats)

    expected_rows_per_state, state_counts = infer_expected_rows_per_state(df, state_col=state_col, band_col=band_col)
    mismatched_states = []
    if expected_rows_per_state is not None and not state_counts.empty:
        bad = state_counts[state_counts != expected_rows_per_state]
        mismatched_states = [{"state_id": str(idx), "row_count": int(v)} for idx, v in bad.items()]

    state_count_distribution = pd.DataFrame()
    if state_col is not None and not state_counts.empty:
        state_count_distribution = (
            state_counts.value_counts().sort_index().rename_axis("rows_per_state").reset_index(name="num_states")
        )

    total_unique_states = int(df[state_col].nunique()) if state_col is not None else None
    checks = {
        "required_columns_present": True,
        "state_id_present": state_col is not None,
        "split_column_present": split_col is not None,
        "split_column_valid": bool(split_valid) if split_col is not None else None,
        "no_state_leakage": None
        if split_col is None or not split_valid
        else bool(all((split_leakage[k] or 0) == 0 for k in split_leakage)),
        "uniform_rows_per_state": None if expected_rows_per_state is None else len(mismatched_states) == 0,
    }

    ready_for_training = bool(
        checks["required_columns_present"]
        and checks["state_id_present"]
        and (checks["no_state_leakage"] in [True, None])
        and (checks["uniform_rows_per_state"] in [True, None])
    )
    checks["uniform_rows_per_state"] = (
        None if checks["uniform_rows_per_state"] is None else bool(checks["uniform_rows_per_state"])
    )

    summary = {
        "data_path": str(Path(args.data).resolve()),
        "total_rows": int(len(df)),
        "total_unique_state_id": total_unique_states,
        "resolved_columns": mapped,
        "rows_per_split": rows_per_split.to_dict(orient="records"),
        "unique_states_per_split": unique_states_per_split.to_dict(orient="records"),
        "rows_per_band": rows_per_band.to_dict(orient="records"),
        "qa_valid_false_count": qa_false_count,
        "monotonicity_failed_true_count": monotonicity_failed_count,
        "clamp_counts": clamp_counts,
        "numeric_stats": numeric_stats_df.to_dict(orient="records"),
        "state_rows_expected": expected_rows_per_state,
        "state_rows_mismatch_count": len(mismatched_states),
        "state_rows_mismatches_preview": mismatched_states[:200],
        "split_leakage": split_leakage,
        "checks": checks,
        "ready_for_training": ready_for_training,
    }

    (out_dir / "dataset_check_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    rows_per_split.to_csv(out_dir / "rows_per_split.csv", index=False)
    unique_states_per_split.to_csv(out_dir / "unique_states_per_split.csv", index=False)
    rows_per_band.to_csv(out_dir / "rows_per_band.csv", index=False)
    numeric_stats_df.to_csv(out_dir / "numeric_stats.csv", index=False)
    state_count_distribution.to_csv(out_dir / "state_count_distribution.csv", index=False)

    report_lines = [
        "Dataset Sanity Check Report",
        "",
        f"Data path: {Path(args.data).resolve()}",
        f"Total rows: {len(df):,}",
        f"Total unique state_id: {total_unique_states if total_unique_states is not None else 'N/A'}",
        f"Rows per split available: {'yes' if not rows_per_split.empty else 'no'}",
        f"Unique states per split available: {'yes' if not unique_states_per_split.empty else 'no'}",
        f"Rows per band: {rows_per_band.to_dict(orient='records')}",
        f"qa_valid false count: {qa_false_count if qa_false_count is not None else 'N/A'}",
        f"monotonicity_failed true count: {monotonicity_failed_count if monotonicity_failed_count is not None else 'N/A'}",
        f"Clamp counts: {clamp_counts}",
        f"Expected rows per state: {expected_rows_per_state if expected_rows_per_state is not None else 'N/A'}",
        f"State row mismatch count: {len(mismatched_states)}",
        f"Split leakage: {split_leakage}",
        f"Checks: {checks}",
        f"Dataset ready for training: {ready_for_training}",
        "",
        "Numeric stats are saved in numeric_stats.csv.",
    ]
    (out_dir / "dataset_check_report.txt").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Saved dataset check outputs to: {out_dir.resolve()}")
    print(f"Dataset ready for training: {ready_for_training}")


if __name__ == "__main__":
    main()
