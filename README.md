# pKANrtm

**Physics-guided multi-fidelity surrogate for atmospheric correction coefficients**

Official code release for:

> **Multi-Fidelity Emulation of Atmospheric Correction Coefficients with Physics-Guided Kolmogorov–Arnold Networks**  
> Md Abdullah Al Mazid, Naphtali Rishe  
> *Remote Sensing* **2026**, *18*(11), 1826  
> [https://doi.org/10.3390/rs18111826](https://doi.org/10.3390/rs18111826) · [https://www.mdpi.com/2072-4292/18/11/1826](https://www.mdpi.com/2072-4292/18/11/1826)

---

## Overview

This repository implements **pKANrtm**, a physics-guided Kolmogorov–Arnold Network (KAN) that emulates libRadtran atmospheric correction coefficients from paired **6S → libRadtran** multi-fidelity simulations. Atmospheric and geometric states are sampled with Latin Hypercube Sampling; targets are **path reflectance** (`rho_path`), **total transmittance** (`T_total`), and **spherical albedo** (`spher_alb`) for Sentinel-2 bands (SRF-aware).

The pipeline covers:

- shared state-manifest generation and paired RTM dataset builds (6S + libRadtran)
- dataset validation, surrogate training/evaluation (oracle-residual KAN / pKAN)
- runtime benchmarking and multi-GPU architecture sweeps
- paper diagnostic figures and GONA RadCalNet real-scene validation

---

## Citation

If you use this code or the associated datasets, please cite:

```bibtex
@Article{mazid2026pkanrtm,
  author  = {Mazid, Md Abdullah Al and Rishe, Naphtali},
  title   = {Multi-Fidelity Emulation of Atmospheric Correction Coefficients with Physics-Guided Kolmogorov--Arnold Networks},
  journal = {Remote Sensing},
  volume  = {18},
  number  = {11},
  pages   = {1826},
  year    = {2026},
  doi     = {10.3390/rs18111826}
}
```

Plain text: Mazid, M.A.A.; Rishe, N. Multi-Fidelity Emulation of Atmospheric Correction Coefficients with Physics-Guided Kolmogorov–Arnold Networks. *Remote Sens.* **2026**, *18*, 1826. https://doi.org/10.3390/rs18111826

---

## Repository layout

| Path | Role |
|------|------|
| `data_generator/` | Shared states, libRadtran and 6S dataset generation |
| `surrogate_pipeline/` | Model definitions, data loading, training utilities |
| `train_surrogates.py` | Main training and evaluation entry point |
| `tools/` | `check_dataset.py`, `benchmark_runtime.py`, plotting helpers |
| `scripts/` | Paper diagnostic figure generation |
| `analysis/` | Supplementary analysis (e.g. KAN complexity table) |
| `real_scene_validation/` | GONA Sentinel-2 + RadCalNet matchup validation |
| `data/` | Generated datasets (not all shipped in git; see workflow below) |
| `runs/` | Training outputs, checkpoints, metrics |
| `results/` | Compiled tables and paper artifacts |
| `benchmarks/` | Runtime benchmark helpers |
| `run_parallel_4gpu.sh` | Multi-GPU architecture sweep launcher |

---

## Environment setup

Use a Python environment with PyTorch and RTM dependencies (example: `pylrt`):

```bash
conda activate pylrt
python -m pip install -r requirements.txt
python -m pip install "git+https://github.com/Blealtan/efficient-kan.git"
```

**External RTMs (not installed by pip):**

- **6S** — on `PATH` for Py6S dataset generation
- **libRadtran** — `uvspec` for high-fidelity dataset generation

Sentinel-2 spectral response functions live under `data/sentinel_srf/` (see `tools/extract_s2_srf.py` if you need to rebuild them).

---

## End-to-end workflow

### 1) Generate shared states

```bash
python data_generator/generate_shared_states.py \
  --n_states 50000 \
  --output_dir "data/shared_states_50k" \
  --seed 42
```

Produces: `state_manifest.jsonl`, `summary.json`

### 2) Generate libRadtran dataset

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

Produces: `dataset_rows.jsonl`, `summary.json`

### 3) Generate 6S dataset (same manifest)

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

Produces: `dataset_rows.jsonl`, `summary.json`, `errors.jsonl`

### 4) Sanity-check a dataset

```bash
python tools/check_dataset.py \
  --data "data/generated_libradtran_50k_13b/dataset_rows.jsonl" \
  --output_dir "runs/dataset_check_lrt_50k"
```

Produces: `dataset_check_summary.json`, `dataset_check_report.txt`, CSV summaries

---

## Training and evaluation

**Entry point:** `train_surrogates.py`

| Setting | Options (paper focus) |
|---------|------------------------|
| Fidelity | `oracle_residual` |
| Models | `kan`, `pkan` |
| Splits | `standard`, `ood_aod_cwv`, `both` |

### Example: oracle-residual KAN / pKAN

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

**Note:** In-training runtime timing is disabled inside `train_surrogates.py`. Use `tools/benchmark_runtime.py` for timing studies.

### Metrics and outputs

Per split: RMSE, MAE, R², MAPE, sMAPE (with `eps = 1e-6` for ratio metrics). Reports include band-wise, coefficient-wise, and full band×coefficient tables.

Under `<output_dir>/<fidelity_mode>/`:

- `combined_metrics.csv`, `accuracy_table.csv`, band/coefficient tables
- `model_comparison.md`
- `splits/standard/` and/or `splits/ood_aod_cwv/`

Per model (`.../splits/<split>/models/<model>/`): `metrics.json`, `test_predictions.csv`, checkpoints, `history.csv`

---

## Multi-GPU architecture sweep

```bash
JOBS_PER_GPU=2 \
PARENT="runs/mf_parallel_4gpu_oracle_kan_pkan" \
EXTRA_ARGS="--epochs 100 --batch_size 2048 --amp" \
bash run_parallel_4gpu.sh
```

Defaults: models `kan pkan`, `--exclude_b10`, GPUs `3 5 6 7` (override in script). Parent run aggregates `final_comparison.md`, `final_runtime_comparison.md`, and related CSVs.

---

## Runtime benchmarking

```bash
python tools/benchmark_runtime.py \
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

Outputs: `benchmark_results.md`, `benchmark_results.csv`, `machine_info.json`, `raw_timings.json`

---

## Real-scene validation (GONA / RadCalNet)

Scripts under `real_scene_validation/` validate the trained surrogate against **Sen2Cor L2A**, **L1C TOA**, and **RadCalNet** in situ reflectance at the GONA site (Sentinel-2A, 2018-05-25).

| Script | Purpose |
|--------|---------|
| `check_matchup_quality.py` | ROI, cloud/SCL, and RadCalNet matchup diagnostics |
| `run_gona_real_scene_validation.py` | Full band-wise validation vs RadCalNet |
| `plot_spectral_validation.py` | Spectral validation figure (PDF/PNG) |

Place scene data under `real_scene_validation/data/GONA 25th May 2018/` (L1C/L2A SAFE subsets + `GONA01_2018_145_v04.09.nc`). Point `--checkpoint` to a trained pKAN weights file (e.g. `runs/fig05_surrogate_pkan/.../pkan/best.pt`).

```bash
python real_scene_validation/check_matchup_quality.py --help
python real_scene_validation/run_gona_real_scene_validation.py --help
```

---

## Paper figures and supplementary analysis

```bash
python scripts/make_paper_diagnostic_figures.py
```

Figures are written under `results/complied/figures/paper_diagnostics/`.

```bash
python analysis/reviewer_kan_complexity_table.py
```

Writes KAN vs FC comparison tables under `results/reviewer_kan_complexity/`.

---

## Implementation notes

- Splits are **state-aware** to prevent leakage across train/val/test.
- Rows with `qa_valid == false` are excluded when the flag is present.
- **`pkan`** adds a physics-consistency penalty (`--lambda_phys`) in coefficient space.
- **`--exclude_b10`** drops `band == B10` before split/train/eval (circus band; common in paper experiments).
- Low-fidelity inputs are 6S coefficients; the model predicts the **residual to libRadtran** and reconstructs high-fidelity targets.

---

## License and contact

Software is provided to reproduce the methods and results described in the *Remote Sensing* article above. For questions about the publication, use the corresponding author contact listed on the [journal page](https://www.mdpi.com/2072-4292/18/11/1826).
