# Surrogate Comparison Report (oracle_residual)

MAPE/sMAPE denominator epsilon:
`eps = 1.0e-06`

For near-zero targets, MAPE can be unstable; treat sMAPE as the more robust percentage error metric.

## Overall Summary Table

| fidelity_mode   | split_mode   | model   |      rmse |        mae |       r2 |   mape |   smape |
|:----------------|:-------------|:--------|----------:|-----------:|---------:|-------:|--------:|
| oracle_residual | ood_aod_cwv  | pkan    | 0.0136466 | 0.00764949 | 0.990418 | 3517.1 | 7.40311 |

## CSV Exports

- overall metrics: `combined_metrics.csv`, `accuracy_table.csv`
- band-wise metrics: `band_metrics_table.csv`
- coefficient-wise metrics: `coefficient_metrics_table.csv`
- band x coefficient matrix: `band_coefficient_metrics_table.csv`
- best/worst summaries: `best_worst_band_metrics.csv`, `best_worst_coefficient_metrics.csv`
