# Atmospheric Correction Surrogate Pipeline

This repository contains a full workflow for atmospheric correction surrogate modeling:

- state-manifest generation
- libRadtran dataset generation
- 6S dataset generation
- dataset sanity checks
- surrogate training/evaluation (single- and multi-fidelity)
- runtime benchmarking (`benchmark_runtime.py`)
- multi-GPU architecture sweeps (`run_parallel_4gpu.sh`)

Primary target variables are:

- `rho_path`
- `T_total`
- `spher_alb`

---

## Current Project Entry Points

- `data_generator/generate_shared_states.py`  
  Build a normalized shared state manifest (`state_manifest.jsonl`).
- `data_generator/generate_libradtran_dataset.py`  
  Generate libRadtran rows from a shared manifest.
- `data_generator/generate_6s_dataset.py`  
  Generate 6S rows from the same shared manifest.
- `check_dataset.py`  
  Validate dataset structure/distribution/readiness.
- `train_surrogates.py`  
  Train/evaluate models and generate reports/metrics.
- `benchmark_runtime.py`  
  Benchmark RTM + surrogate runtime (tables + markdown + machine info).
- `run_parallel_4gpu.sh`  
  Architecture sweep launcher with final parent-level aggregation.

---

## Environment Setup

Use your existing environment (example: `pylrt`):

```bash
conda activate pylrt
python -m pip install -r requirements.txt
python -m pip install "git+https://github.com/Blealtan/efficient-kan.git"
```

For 6S generation, ensure the 6S executable is available on `PATH` for Py6S.

---

## End-to-End Workflow

## 1) Generate shared states

```bash
python data_generator/generate_shared_states.py \
  --n_states 50000 \
  --output_dir "data/shared_states_50k" \
  --seed 42
```

Produces:

- `state_manifest.jsonl`
- `summary.json`

## 2) Generate libRadtran dataset from the shared manifest

```bash
python data_generator/generate_libradtran_dataset.py \
  --states_manifest "data/shared_states_50k/state_manifest.jsonl" \
  --output_dir "data/generated_libradtran_50k_13b" \
  --bands "B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B10,B11,B12" \
  --rho1 0.2 \
  --rho2 0.6 \
  --max_workers 8 \
  --copy_manifest
```

Produces:

- `dataset_rows.jsonl`
- `summary.json`
- optional `state_manifest.jsonl` copy

## 3) Generate 6S dataset from the same manifest

```bash
python data_generator/generate_6s_dataset.py \
  --states_manifest "data/shared_states_50k/state_manifest.jsonl" \
  --output_dir "data/generated_6s_50k_13b" \
  --bands "B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B10,B11,B12" \
  --rho1 0.2 \
  --rho2 0.6 \
  --max_workers 4 \
  --max_in_flight 32 \
  --copy_manifest
```

Produces:

- `dataset_rows.jsonl`
- `summary.json`
- `errors.jsonl`

## 4) Sanity-check dataset

```bash
python check_dataset.py \
  --data "data/generated_libradtran_50k_13b/dataset_rows.jsonl" \
  --output_dir "runs/dataset_check_lrt_50k"
```

Produces:

- `dataset_check_summary.json`
- `dataset_check_report.txt`
- CSV summaries (row counts, split stats, numeric stats, etc.)

---

## Training and Evaluation (`train_surrogates.py`)

Supported fidelity mode (project focus):

- `oracle_residual`

Supported models (project focus):

- `kan`
- `pkan`

Supported split modes:

- `standard`
- `ood_aod_cwv`
- `both`

### Example: oracle residual (KAN/PKAN only)

```bash
python train_surrogates.py \
  --data_6s "data/generated_6s_50k_13b/dataset_rows.jsonl" \
  --data_lrt "data/generated_libradtran_50k_13b/dataset_rows.jsonl" \
  --output_dir "runs/mf_oracle_kan_pkan" \
  --fidelity_mode oracle_residual \
  --models kan pkan \
  --split_mode both \
  --epochs 100 \
  --batch_size 2048 \
  --gpu 0 \
  --amp \
  --exclude_b10
```

### Important note on in-training runtime timing

`train_surrogates.py` currently force-disables in-training runtime benchmarking (even if runtime flags are passed).  
Use `benchmark_runtime.py` for runtime/timing analysis.

---

## Metrics and Reporting (current)

For each split, overall and per-dimension metrics include:

- RMSE
- MAE
- R2
- MAPE (epsilon-safe denominator)
- sMAPE (epsilon-safe denominator)

The report documents epsilon (`eps = 1e-6`) and includes the near-zero-target caution.

### Detailed breakdowns generated

- band-wise aggregated metrics across coefficients
- coefficient-wise aggregated metrics across bands
- full band x coefficient matrix
- best/worst band/coefficient by RMSE and sMAPE

### Key training outputs

At fidelity root (`<output_dir>/<fidelity_mode>/`):

- `combined_metrics.csv`
- `accuracy_table.csv`
- `band_metrics_table.csv`
- `coefficient_metrics_table.csv`
- `band_coefficient_metrics_table.csv`
- `best_worst_band_metrics.csv`
- `best_worst_coefficient_metrics.csv`
- `model_comparison.md`
- `runtime_table.csv` (when available)
- `splits/standard/` and/or `splits/ood_aod_cwv/`

At per-model split directory (`.../splits/<split_mode>/models/<model>/`):

- `metrics.json`
- `test_predictions.csv`
- `per_band_metrics.csv`
- `per_coefficient_metrics.csv`
- `per_band_coefficient_metrics.csv`
- neural: checkpoints/history/loss plots
- tree: serialized model artifacts

---

## Multi-GPU Architecture Sweep (`run_parallel_4gpu.sh`)

Launches architecture sweeps over fixed bundles:

- oracle + standard
- oracle + OOD

Default fixed GPUs are currently:

- `3`, `5`, `6`, `7`

Important defaults in script:

- includes `--exclude_b10`
- `JOBS_PER_GPU` controls queue depth per GPU
- base models are `kan pkan`

### Example

```bash
JOBS_PER_GPU=2 \
PARENT="runs/mf_parallel_4gpu_oracle_kan_pkan" \
EXTRA_ARGS="--epochs 100 --batch_size 2048 --amp" \
bash run_parallel_4gpu.sh
```

### Parent-level aggregated outputs (current)

- `final_comparison.csv` / `.md`
- `final_runtime_comparison.csv` / `.md`
- `final_band_metrics_comparison.csv` / `.md`
- `final_coefficient_metrics_comparison.csv` / `.md`
- `final_band_coefficient_metrics_comparison.csv` / `.md`
- `final_all_comparison.md`

---

## Runtime Benchmarking (`benchmark_runtime.py`)

This is the current benchmark entry point (replaces older naming).

It benchmarks:

- libRadtran CPU baseline
- 6S runtime
- surrogate pipelines discovered from mode/run mappings
- optional SRF sensitivity analysis

### Example

```bash
python benchmark_runtime.py \
  --data "data/generated_libradtran_50k_13b/dataset_rows.jsonl" \
  --mode_run_root oracle_residual="runs/mf_parallel_4gpu_oracle_kan_pkan" \
  --models kan pkan \
  --split_mode test \
  --devices cpu cuda:0 \
  --batch_sizes 1 256 \
  --n_samples 256 \
  --repeats 5 \
  --output_dir "benchmarks/runtime_latest"
```

Produces:

- `benchmark_results.csv`
- `benchmark_results_all.csv`
- `benchmark_results_batch1.csv`
- `benchmark_results_throughput.csv`
- `benchmark_results.md`
- `machine_info.json`
- `machine_info.md`
- `raw_timings.json`
- `sampled_inputs.csv`

---

## Notes

- Splits are state-aware to avoid leakage.
- `qa_valid` is used when present.
- `pkan` applies physics-regularized loss (`--lambda_phys`).
- `--exclude_b10` removes `band == B10` before split/train/eval.
