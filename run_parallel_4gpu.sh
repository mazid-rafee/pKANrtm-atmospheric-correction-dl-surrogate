#!/usr/bin/env bash
set -euo pipefail

PARENT="${PARENT:-runs/mf_parallel_4gpu_variants}"
mkdir -p "$PARENT/logs"

PYBIN="${PYBIN:-}"
if [[ -z "$PYBIN" ]]; then
  for cand in python python3; do
    if command -v "$cand" >/dev/null 2>&1; then
      if "$cand" -c "import pandas" >/dev/null 2>&1; then
        PYBIN="$cand"
        break
      fi
    fi
  done
fi

if [[ -z "$PYBIN" ]]; then
  echo "No Python interpreter with pandas found on PATH."
  echo "Activate your env or run with: PYBIN=/path/to/python bash run_parallel_4gpu.sh"
  exit 1
fi

echo "Using Python interpreter: $PYBIN"
echo "Parent output directory: $PARENT"

# Architecture sweep axes (independent sweeps).
MLP_ARCHES=("baseline" "wider_gelu" "layernorm_gelu" "residual" "shared_trunk_multihead")
KAN_ARCHES=("baseline" "small" "balanced_deep" "large_deep" "shared_trunk_multihead")

# Global architecture toggles (can override via env).
ACTIVATION="${ACTIVATION:-relu}"                    # relu|gelu
USE_LAYERNORM="${USE_LAYERNORM:-0}"                 # 0|1
DEPLOYABLE_STAGE_SIZING="${DEPLOYABLE_STAGE_SIZING:-1}"  # 0|1
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"                   # >=1 concurrent runs per GPU queue
EXCLUDE_B10="${EXCLUDE_B10:-0}"                     # 0|1 (default include B10)

if ! [[ "$JOBS_PER_GPU" =~ ^[0-9]+$ ]] || [[ "$JOBS_PER_GPU" -lt 1 ]]; then
  echo "JOBS_PER_GPU must be a positive integer (got: $JOBS_PER_GPU)"
  exit 1
fi

# Common train args (can be extended by setting EXTRA_ARGS env var).
BASE_ARGS=(
  --data_6s "data/qavalid_intersection_libradtran_6s_50k_13b/dataset_rows_6s.jsonl"
  --data_lrt "data/qavalid_intersection_libradtran_6s_50k_13b/dataset_rows_libradtran.jsonl"
  --models mlp kan pmlp pkan
  --no_progress
)

if [[ "$EXCLUDE_B10" == "1" ]]; then
  BASE_ARGS+=(--exclude_b10)
fi

if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=(${EXTRA_ARGS})
  BASE_ARGS+=("${EXTRA_ARR[@]}")
fi

run_bundle () {
  local bundle="$1"
  local fid="$2"
  local split="$3"
  local gpu="$4"

  (
    set -euo pipefail
    local max_jobs="$JOBS_PER_GPU"
    local total=$(( ${#MLP_ARCHES[@]} + ${#KAN_ARCHES[@]} ))
    local idx=0
    echo "[$bundle] GPU $gpu queue started with JOBS_PER_GPU=$max_jobs"

    launch_variant () {
      local sweep_axis="$1"
      local mlp_arch="$2"
      local kan_arch="$3"
      local tag="axis-${sweep_axis}__m-${mlp_arch}__k-${kan_arch}__act-${ACTIVATION}__ln-${USE_LAYERNORM}"
      local out_dir="$PARENT/$bundle/$tag"
      local log_file="$PARENT/logs/${bundle}__${tag}.log"

      echo "[$bundle][$idx/$total] $tag on GPU $gpu"

      local -a cmd=(
        "$PYBIN" train_surrogates.py
        "${BASE_ARGS[@]}"
        --output_dir "$out_dir"
        --fidelity_mode "$fid"
        --split_mode "$split"
        --gpu "$gpu"
        --mlp_arch "$mlp_arch"
        --kan_arch "$kan_arch"
        --activation "$ACTIVATION"
      )
      if [[ "$USE_LAYERNORM" == "1" ]]; then
        cmd+=(--use_layernorm)
      fi
      if [[ "$DEPLOYABLE_STAGE_SIZING" == "1" ]]; then
        cmd+=(--deployable_stage_sizing)
      fi
      "${cmd[@]}" > "$log_file" 2>&1 &
    }

    throttle_queue () {
      while true; do
        local running
        running=$(jobs -pr | wc -l | tr -d ' ')
        if [[ "$running" -lt "$max_jobs" ]]; then
          break
        fi
        wait -n
      done
    }

    # Sweep MLP variants while keeping KAN at baseline.
    for mlp_arch in "${MLP_ARCHES[@]}"; do
      idx=$((idx + 1))
      throttle_queue
      launch_variant "mlp" "$mlp_arch" "baseline"
    done

    # Sweep KAN variants while keeping MLP at baseline.
    for kan_arch in "${KAN_ARCHES[@]}"; do
      idx=$((idx + 1))
      throttle_queue
      launch_variant "kan" "baseline" "$kan_arch"
    done

    wait
    echo "[$bundle] GPU $gpu queue completed."
  ) &
}

# Same 4-GPU strategy as before: each GPU handles one (fidelity, split) bundle.
run_bundle "oracle_standard_gpu3"      "oracle_residual"     "standard"    "3"
run_bundle "oracle_ood_gpu5"           "oracle_residual"     "ood_aod_cwv" "5"
run_bundle "deployable_standard_gpu6"  "deployable_residual" "standard"    "6"
run_bundle "deployable_ood_gpu7"       "deployable_residual" "ood_aod_cwv" "7"

wait

PARENT_DIR="$PARENT" "$PYBIN" - <<'PY'
from pathlib import Path
import os
import pandas as pd

parent = Path(os.environ["PARENT_DIR"])
bundles = [
    ("oracle_standard_gpu3", "oracle_residual", "standard"),
    ("oracle_ood_gpu5", "oracle_residual", "ood_aod_cwv"),
    ("deployable_standard_gpu6", "deployable_residual", "standard"),
    ("deployable_ood_gpu7", "deployable_residual", "ood_aod_cwv"),
]

def parse_variant_tag(tag: str):
    out = {
        "variant_tag": tag,
        "sweep_axis": None,
        "mlp_arch": None,
        "kan_arch": None,
        "activation": None,
        "use_layernorm": None,
    }
    for part in tag.split("__"):
        if part.startswith("axis-"):
            out["sweep_axis"] = part[5:]
        elif part.startswith("m-"):
            out["mlp_arch"] = part[2:]
        elif part.startswith("k-"):
            out["kan_arch"] = part[2:]
        elif part.startswith("act-"):
            out["activation"] = part[4:]
        elif part.startswith("ln-"):
            out["use_layernorm"] = part[3:] == "1"
    return out

acc_tables = []
rt_tables = []
band_tables = []
coeff_tables = []
band_coeff_tables = []

for bundle, fid, split in bundles:
    bundle_root = parent / bundle
    if not bundle_root.exists():
        continue

    for variant_dir in sorted([p for p in bundle_root.iterdir() if p.is_dir()]):
        info = parse_variant_tag(variant_dir.name)

        acc_path = variant_dir / fid / "accuracy_table.csv"
        if acc_path.exists():
            acc_df = pd.read_csv(acc_path)
            acc_df["run_subfolder"] = bundle
            acc_df["variant_tag"] = info["variant_tag"]
            acc_df["sweep_axis"] = info["sweep_axis"]
            acc_df["mlp_arch"] = info["mlp_arch"]
            acc_df["kan_arch"] = info["kan_arch"]
            acc_df["activation"] = info["activation"]
            acc_df["use_layernorm"] = info["use_layernorm"]
            acc_df["expected_fidelity_mode"] = fid
            acc_df["expected_split_mode"] = split
            acc_tables.append(acc_df)

        rt_path = variant_dir / fid / "runtime_table.csv"
        if rt_path.exists():
            rt_df = pd.read_csv(rt_path)
            rt_df["run_subfolder"] = bundle
            rt_df["variant_tag"] = info["variant_tag"]
            rt_df["sweep_axis"] = info["sweep_axis"]
            rt_df["mlp_arch"] = info["mlp_arch"]
            rt_df["kan_arch"] = info["kan_arch"]
            rt_df["activation"] = info["activation"]
            rt_df["use_layernorm"] = info["use_layernorm"]
            rt_df["expected_fidelity_mode"] = fid
            rt_df["expected_split_mode"] = split
            rt_tables.append(rt_df)

        band_path = variant_dir / fid / "band_metrics_table.csv"
        if band_path.exists():
            band_df = pd.read_csv(band_path)
            band_df["run_subfolder"] = bundle
            band_df["variant_tag"] = info["variant_tag"]
            band_df["sweep_axis"] = info["sweep_axis"]
            band_df["mlp_arch"] = info["mlp_arch"]
            band_df["kan_arch"] = info["kan_arch"]
            band_df["activation"] = info["activation"]
            band_df["use_layernorm"] = info["use_layernorm"]
            band_df["expected_fidelity_mode"] = fid
            band_df["expected_split_mode"] = split
            band_tables.append(band_df)

        coeff_path = variant_dir / fid / "coefficient_metrics_table.csv"
        if coeff_path.exists():
            coeff_df = pd.read_csv(coeff_path)
            coeff_df["run_subfolder"] = bundle
            coeff_df["variant_tag"] = info["variant_tag"]
            coeff_df["sweep_axis"] = info["sweep_axis"]
            coeff_df["mlp_arch"] = info["mlp_arch"]
            coeff_df["kan_arch"] = info["kan_arch"]
            coeff_df["activation"] = info["activation"]
            coeff_df["use_layernorm"] = info["use_layernorm"]
            coeff_df["expected_fidelity_mode"] = fid
            coeff_df["expected_split_mode"] = split
            coeff_tables.append(coeff_df)

        band_coeff_path = variant_dir / fid / "band_coefficient_metrics_table.csv"
        if band_coeff_path.exists():
            band_coeff_df = pd.read_csv(band_coeff_path)
            band_coeff_df["run_subfolder"] = bundle
            band_coeff_df["variant_tag"] = info["variant_tag"]
            band_coeff_df["sweep_axis"] = info["sweep_axis"]
            band_coeff_df["mlp_arch"] = info["mlp_arch"]
            band_coeff_df["kan_arch"] = info["kan_arch"]
            band_coeff_df["activation"] = info["activation"]
            band_coeff_df["use_layernorm"] = info["use_layernorm"]
            band_coeff_df["expected_fidelity_mode"] = fid
            band_coeff_df["expected_split_mode"] = split
            band_coeff_tables.append(band_coeff_df)

if acc_tables:
    acc = pd.concat(acc_tables, ignore_index=True)
    acc = acc.sort_values(
        ["split_mode", "fidelity_mode", "mlp_arch", "kan_arch", "rmse", "model"]
    ).reset_index(drop=True)
    acc.to_csv(parent / "final_comparison.csv", index=False)
    (parent / "final_comparison.md").write_text(
        "\n".join(["# Final Accuracy Comparison", "", acc.to_markdown(index=False)]),
        encoding="utf-8",
    )

if rt_tables:
    rt = pd.concat(rt_tables, ignore_index=True)
    sort_col = "avg_test_inference_sec" if "avg_test_inference_sec" in rt.columns else rt.columns[0]
    rt = rt.sort_values(
        [c for c in ["split_mode", "fidelity_mode", "mlp_arch", "kan_arch", sort_col] if c in rt.columns]
    ).reset_index(drop=True)
    rt.to_csv(parent / "final_runtime_comparison.csv", index=False)
    (parent / "final_runtime_comparison.md").write_text(
        "\n".join(["# Final Runtime Comparison", "", rt.to_markdown(index=False)]),
        encoding="utf-8",
    )

if band_tables:
    band = pd.concat(band_tables, ignore_index=True)
    band = band.sort_values(
        ["split_mode", "fidelity_mode", "mlp_arch", "kan_arch", "model", "band", "rmse", "smape"]
    ).reset_index(drop=True)
    band.to_csv(parent / "final_band_metrics_comparison.csv", index=False)
    (parent / "final_band_metrics_comparison.md").write_text(
        "\n".join(["# Final Band-wise Metrics Comparison", "", band.to_markdown(index=False)]),
        encoding="utf-8",
    )

if coeff_tables:
    coeff = pd.concat(coeff_tables, ignore_index=True)
    coeff = coeff.sort_values(
        ["split_mode", "fidelity_mode", "mlp_arch", "kan_arch", "model", "coefficient", "rmse", "smape"]
    ).reset_index(drop=True)
    coeff.to_csv(parent / "final_coefficient_metrics_comparison.csv", index=False)
    (parent / "final_coefficient_metrics_comparison.md").write_text(
        "\n".join(["# Final Coefficient-wise Metrics Comparison", "", coeff.to_markdown(index=False)]),
        encoding="utf-8",
    )

if band_coeff_tables:
    band_coeff = pd.concat(band_coeff_tables, ignore_index=True)
    band_coeff = band_coeff.sort_values(
        ["split_mode", "fidelity_mode", "mlp_arch", "kan_arch", "model", "band", "coefficient", "rmse", "smape"]
    ).reset_index(drop=True)
    band_coeff.to_csv(parent / "final_band_coefficient_metrics_comparison.csv", index=False)
    (parent / "final_band_coefficient_metrics_comparison.md").write_text(
        "\n".join(["# Final Band x Coefficient Metrics Comparison", "", band_coeff.to_markdown(index=False)]),
        encoding="utf-8",
    )

if acc_tables or rt_tables or band_tables or coeff_tables or band_coeff_tables:
    lines = ["# Final Combined Comparison", ""]
    if acc_tables:
        acc_full = pd.read_csv(parent / "final_comparison.csv")
        lines.extend(["## Accuracy", "", acc_full.to_markdown(index=False), ""])
        split_priority = ["standard", "ood_aod_cwv"]
        top_chunks = []
        for split_name in split_priority:
            split_df = acc_full[acc_full["split_mode"] == split_name]
            if not split_df.empty:
                top_chunks.append(split_df.nsmallest(5, "rmse"))
        if top_chunks:
            top10_acc = pd.concat(top_chunks, ignore_index=True)
        else:
            top10_acc = acc_full.nsmallest(10, "rmse")
        lines.extend(["## Top-10 Accuracy (Lowest RMSE)", "", top10_acc.to_markdown(index=False), ""])
        acc_mlp = acc_full[acc_full["model"].isin(["mlp", "pmlp"])]
        acc_kan = acc_full[acc_full["model"].isin(["kan", "pkan"])]
        if not acc_mlp.empty:
            lines.extend(["## Best MLP-Family Result", "", acc_mlp.nsmallest(1, "rmse").to_markdown(index=False), ""])
        if not acc_kan.empty:
            lines.extend(["## Best KAN-Family Result", "", acc_kan.nsmallest(1, "rmse").to_markdown(index=False), ""])
    if rt_tables:
        rt_full = pd.read_csv(parent / "final_runtime_comparison.csv")
        lines.extend(["## Runtime", "", rt_full.to_markdown(index=False), ""])
        if "avg_test_inference_sec" in rt_full.columns:
            top10_rt = rt_full.nsmallest(10, "avg_test_inference_sec")
            lines.extend(["## Top-10 Runtime (Fastest)", "", top10_rt.to_markdown(index=False), ""])
    if band_tables:
        band_full = pd.read_csv(parent / "final_band_metrics_comparison.csv")
        lines.extend(["## Band-wise Metrics", "", band_full.to_markdown(index=False), ""])
        if "split_mode" in band_full.columns:
            band_test = band_full[band_full["split_mode"] == "ood_aod_cwv"]
            if band_test.empty:
                band_test = band_full
            lines.extend(
                [
                    "## Top-10 Band-wise (Lowest RMSE)",
                    "",
                    band_test.nsmallest(10, "rmse").to_markdown(index=False),
                    "",
                ]
            )
            lines.extend(
                [
                    "## Top-10 Band-wise (Lowest sMAPE)",
                    "",
                    band_test.nsmallest(10, "smape").to_markdown(index=False),
                    "",
                ]
            )
    if coeff_tables:
        coeff_full = pd.read_csv(parent / "final_coefficient_metrics_comparison.csv")
        lines.extend(["## Coefficient-wise Metrics", "", coeff_full.to_markdown(index=False), ""])
        if "split_mode" in coeff_full.columns:
            coeff_test = coeff_full[coeff_full["split_mode"] == "ood_aod_cwv"]
            if coeff_test.empty:
                coeff_test = coeff_full
            lines.extend(
                [
                    "## Top-10 Coefficient-wise (Lowest RMSE)",
                    "",
                    coeff_test.nsmallest(10, "rmse").to_markdown(index=False),
                    "",
                ]
            )
            lines.extend(
                [
                    "## Top-10 Coefficient-wise (Lowest sMAPE)",
                    "",
                    coeff_test.nsmallest(10, "smape").to_markdown(index=False),
                    "",
                ]
            )
    if band_coeff_tables:
        band_coeff_full = pd.read_csv(parent / "final_band_coefficient_metrics_comparison.csv")
        lines.extend(["## Band x Coefficient Metrics", "", band_coeff_full.to_markdown(index=False), ""])
        if "split_mode" in band_coeff_full.columns:
            bc_test = band_coeff_full[band_coeff_full["split_mode"] == "ood_aod_cwv"]
            if bc_test.empty:
                bc_test = band_coeff_full
            lines.extend(
                [
                    "## Top-10 Band x Coefficient (Lowest RMSE)",
                    "",
                    bc_test.nsmallest(10, "rmse").to_markdown(index=False),
                    "",
                ]
            )
            lines.extend(
                [
                    "## Top-10 Band x Coefficient (Lowest sMAPE)",
                    "",
                    bc_test.nsmallest(10, "smape").to_markdown(index=False),
                    "",
                ]
            )
    lines.extend(
        [
            "## CSV Exports",
            "",
            "- final accuracy: `final_comparison.csv`",
            "- final runtime: `final_runtime_comparison.csv`",
            "- final band-wise: `final_band_metrics_comparison.csv`",
            "- final coefficient-wise: `final_coefficient_metrics_comparison.csv`",
            "- final band x coefficient: `final_band_coefficient_metrics_comparison.csv`",
            "",
        ]
    )
    if acc_tables and rt_tables:
        if {"fidelity_mode", "split_mode", "model", "mlp_arch", "kan_arch", "activation", "use_layernorm", "variant_tag", "run_subfolder"}.issubset(acc_full.columns) and {"fidelity_mode", "split_mode", "model", "mlp_arch", "kan_arch", "activation", "use_layernorm", "variant_tag", "run_subfolder"}.issubset(rt_full.columns):
            merged = acc_full.merge(
                rt_full,
                on=[
                    "fidelity_mode",
                    "split_mode",
                    "model",
                    "mlp_arch",
                    "kan_arch",
                    "activation",
                    "use_layernorm",
                    "variant_tag",
                    "run_subfolder",
                    "expected_fidelity_mode",
                    "expected_split_mode",
                ],
                how="left",
                suffixes=("", "_rt"),
            )
            if "avg_test_inference_sec" in merged.columns:
                top10_paretoish = merged.sort_values(["rmse", "avg_test_inference_sec"]).head(10)
                lines.extend(["## Top-10 Accuracy Then Runtime", "", top10_paretoish.to_markdown(index=False), ""])
    (parent / "final_all_comparison.md").write_text("\n".join(lines), encoding="utf-8")

print("Saved parent outputs in:", parent)
PY