# Model Comparison (oracle_residual, ood_aod_cwv)

MAPE/sMAPE denominator epsilon:
`eps = 1.0e-06`

For near-zero targets, MAPE can be unstable; treat sMAPE as the more robust percentage error metric.

## Overall Metrics (All Splits)

| model   | split   |       rmse |        mae |       r2 |    mape |   smape |
|:--------|:--------|-----------:|-----------:|---------:|--------:|--------:|
| pkan    | test    | 0.0136466  | 0.00764949 | 0.990418 | 3517.1  | 7.40311 |
| pkan    | train   | 0.00750629 | 0.00275457 | 0.99329  | 3400.83 | 5.20321 |
| pkan    | val     | 0.00834512 | 0.00285852 | 0.991216 | 3939.06 | 5.29038 |

## Band-wise Metrics (Test Split)

| split   | band   |   n_rows |   n_elements |      rmse |        mae |         r2 |         mape |     smape | model   | split_mode   | fidelity_mode   |
|:--------|:-------|---------:|-------------:|----------:|-----------:|-----------:|-------------:|----------:|:--------|:-------------|:----------------|
| test    | B11    |    13847 |        41541 | 0.0075807 | 0.00470288 |   0.965295 |      8.01974 |   6.52706 | pkan    | ood_aod_cwv  | oracle_residual |
| test    | B7     |    13847 |        41541 | 0.0106725 | 0.00644619 |   0.98554  |      3.0028  |   3.01093 | pkan    | ood_aod_cwv  | oracle_residual |
| test    | B8A    |    13847 |        41541 | 0.011165  | 0.00654564 |   0.983696 |      3.23569 |   3.24427 | pkan    | ood_aod_cwv  | oracle_residual |
| test    | B12    |    13847 |        41541 | 0.0114796 | 0.00586181 |   0.914135 |     12.213   |  10.727   | pkan    | ood_aod_cwv  | oracle_residual |
| test    | B6     |    13847 |        41541 | 0.0117112 | 0.0073029  |   0.982375 |      3.35454 |   3.38841 | pkan    | ood_aod_cwv  | oracle_residual |
| test    | B1     |    13847 |        41541 | 0.0118819 | 0.00720638 |   0.983678 |      2.33347 |   2.34768 | pkan    | ood_aod_cwv  | oracle_residual |
| test    | B2     |    13847 |        41541 | 0.0122014 | 0.00742625 |   0.983776 |      2.618   |   2.62888 | pkan    | ood_aod_cwv  | oracle_residual |
| test    | B3     |    13847 |        41541 | 0.0128571 | 0.00790873 |   0.980823 |      3.00699 |   3.02064 | pkan    | ood_aod_cwv  | oracle_residual |
| test    | B4     |    13847 |        41541 | 0.0130683 | 0.00811873 |   0.978892 |      3.44309 |   3.47197 | pkan    | ood_aod_cwv  | oracle_residual |
| test    | B9     |    13847 |        41541 | 0.0151601 | 0.00932519 |   0.92782  |     28.1659  |  29.4182  | pkan    | ood_aod_cwv  | oracle_residual |
| test    | B8     |    13847 |        41541 | 0.0161438 | 0.00947729 |   0.967223 |      3.9859  |   4.05293 | pkan    | ood_aod_cwv  | oracle_residual |
| test    | B5     |    13847 |        41541 | 0.0180409 | 0.0104514  |   0.949096 |      4.51192 |   4.63374 | pkan    | ood_aod_cwv  | oracle_residual |
| test    | B10    |     1476 |         4428 | 0.0481945 | 0.0172229  | -54.9792   | 398732       | 123.411   | pkan    | ood_aod_cwv  | oracle_residual |

## Coefficient-wise Metrics (Test Split)

| split   | target    | coefficient   |   n_rows |       rmse |        mae |       r2 |      mape |   smape | model   | split_mode   | fidelity_mode   |
|:--------|:----------|:--------------|---------:|-----------:|-----------:|---------:|----------:|--------:|:--------|:-------------|:----------------|
| test    | rho_path  | rho_path      |   167640 | 0.00430145 | 0.00284592 | 0.993581 |   10.4259 | 8.78171 | pkan    | ood_aod_cwv  | oracle_residual |
| test    | spher_alb | spher_alb     |   167640 | 0.00940729 | 0.00459943 | 0.986911 | 2326.94   | 5.92841 | pkan    | ood_aod_cwv  | oracle_residual |
| test    | T_total   | T_total       |   167640 | 0.021253   | 0.0155031  | 0.990762 | 8213.94   | 7.49922 | pkan    | ood_aod_cwv  | oracle_residual |

## Best/Worst Bands by RMSE and sMAPE (Test Split)

| model   | metric   | best_band   |   best_value | worst_band   |   worst_value |
|:--------|:---------|:------------|-------------:|:-------------|--------------:|
| pkan    | rmse     | B11         |    0.0075807 | B10          |     0.0481945 |
| pkan    | smape    | B1          |    2.34768   | B10          |   123.411     |

## Best/Worst Coefficients by RMSE and sMAPE (Test Split)

| model   | metric   | best_coefficient   |   best_value | worst_coefficient   |   worst_value |
|:--------|:---------|:-------------------|-------------:|:--------------------|--------------:|
| pkan    | rmse     | rho_path           |   0.00430145 | T_total             |      0.021253 |
| pkan    | smape    | spher_alb          |   5.92841    | rho_path            |      8.78171  |

## Band x Coefficient Matrix (Test Split)

| split   | band   | target    | coefficient   |   n_rows |       rmse |        mae |          r2 |         mape |     smape | model   | split_mode   | fidelity_mode   |
|:--------|:-------|:----------|:--------------|---------:|-----------:|-----------:|------------:|-------------:|----------:|:--------|:-------------|:----------------|
| test    | B1     | T_total   | T_total       |    13847 | 0.0188484  | 0.0128133  |    0.978008 |      2.5934  |   2.65255 | pkan    | ood_aod_cwv  | oracle_residual |
| test    | B1     | rho_path  | rho_path      |    13847 | 0.00621107 | 0.00460054 |    0.985504 |      2.86041 |   2.84455 | pkan    | ood_aod_cwv  | oracle_residual |
| test    | B1     | spher_alb | spher_alb     |    13847 | 0.00544959 | 0.00420533 |    0.98752  |      1.54661 |   1.54595 | pkan    | ood_aod_cwv  | oracle_residual |
| test    | B10    | T_total   | T_total       |     1476 | 0.0239572  | 0.0199286  | -130.778    | 932357       | 189.263   | pkan    | ood_aod_cwv  | oracle_residual |
| test    | B10    | rho_path  | rho_path      |     1476 | 0.00349026 | 0.00279797 |  -34.5973   |    273.473   | 109.907   | pkan    | ood_aod_cwv  | oracle_residual |
| test    | B10    | spher_alb | spher_alb     |     1476 | 0.0798874  | 0.028942   |    0.437872 | 263565       |  71.0621  | pkan    | ood_aod_cwv  | oracle_residual |
| test    | B11    | T_total   | T_total       |    13847 | 0.0117781  | 0.00868357 |    0.968111 |      1.02255 |   1.02957 | pkan    | ood_aod_cwv  | oracle_residual |
| test    | B11    | rho_path  | rho_path      |    13847 | 0.00246814 | 0.00149097 |    0.964967 |     12.6083  |  10.6695  | pkan    | ood_aod_cwv  | oracle_residual |
| test    | B11    | spher_alb | spher_alb     |    13847 | 0.00525228 | 0.00393409 |    0.962806 |     10.4284  |   7.88215 | pkan    | ood_aod_cwv  | oracle_residual |
| test    | B12    | T_total   | T_total       |    13847 | 0.0192694  | 0.0131017  |    0.859021 |      1.73326 |   1.76608 | pkan    | ood_aod_cwv  | oracle_residual |
| test    | B12    | rho_path  | rho_path      |    13847 | 0.00157634 | 0.00103523 |    0.94339  |     21.6835  |  18.7252  | pkan    | ood_aod_cwv  | oracle_residual |
| test    | B12    | spher_alb | spher_alb     |    13847 | 0.00464214 | 0.00344852 |    0.939995 |     13.2221  |  11.6897  | pkan    | ood_aod_cwv  | oracle_residual |
| test    | B2     | T_total   | T_total       |    13847 | 0.0196468  | 0.0141071  |    0.976523 |      2.51331 |   2.5722  | pkan    | ood_aod_cwv  | oracle_residual |
| test    | B2     | rho_path  | rho_path      |    13847 | 0.00566937 | 0.00396695 |    0.984812 |      3.45623 |   3.42512 | pkan    | ood_aod_cwv  | oracle_residual |
| test    | B2     | spher_alb | spher_alb     |    13847 | 0.00533714 | 0.00420469 |    0.989994 |      1.88446 |   1.88931 | pkan    | ood_aod_cwv  | oracle_residual |
| test    | B3     | T_total   | T_total       |    13847 | 0.0208165  | 0.0154246  |    0.971639 |      2.67764 |   2.74357 | pkan    | ood_aod_cwv  | oracle_residual |
| test    | B3     | rho_path  | rho_path      |    13847 | 0.00555967 | 0.00388145 |    0.981687 |      4.16982 |   4.13892 | pkan    | ood_aod_cwv  | oracle_residual |
| test    | B3     | spher_alb | spher_alb     |    13847 | 0.00562827 | 0.00442016 |    0.989143 |      2.17351 |   2.17942 | pkan    | ood_aod_cwv  | oracle_residual |
| test    | B4     | T_total   | T_total       |    13847 | 0.0213591  | 0.0164266  |    0.964221 |      2.58939 |   2.64488 | pkan    | ood_aod_cwv  | oracle_residual |
| test    | B4     | rho_path  | rho_path      |    13847 | 0.00488928 | 0.00345161 |    0.983362 |      4.97928 |   4.9874  | pkan    | ood_aod_cwv  | oracle_residual |
| test    | B4     | spher_alb | spher_alb     |    13847 | 0.00567664 | 0.00447801 |    0.989092 |      2.76059 |   2.78364 | pkan    | ood_aod_cwv  | oracle_residual |
| test    | B5     | T_total   | T_total       |    13847 | 0.030374   | 0.0236533  |    0.876551 |      4.15293 |   4.29253 | pkan    | ood_aod_cwv  | oracle_residual |
| test    | B5     | rho_path  | rho_path      |    13847 | 0.00405756 | 0.00290054 |    0.984661 |      5.72897 |   5.86917 | pkan    | ood_aod_cwv  | oracle_residual |
| test    | B5     | spher_alb | spher_alb     |    13847 | 0.00611422 | 0.00480043 |    0.986078 |      3.65386 |   3.73952 | pkan    | ood_aod_cwv  | oracle_residual |
| test    | B6     | T_total   | T_total       |    13847 | 0.0190211  | 0.0144794  |    0.971884 |      2.05696 |   2.09003 | pkan    | ood_aod_cwv  | oracle_residual |
| test    | B6     | rho_path  | rho_path      |    13847 | 0.00401463 | 0.00275725 |    0.987842 |      4.67285 |   4.68695 | pkan    | ood_aod_cwv  | oracle_residual |
| test    | B6     | spher_alb | spher_alb     |    13847 | 0.00579109 | 0.00467204 |    0.987398 |      3.33381 |   3.38826 | pkan    | ood_aod_cwv  | oracle_residual |
| test    | B7     | T_total   | T_total       |    13847 | 0.0173495  | 0.0128009  |    0.977972 |      1.76182 |   1.78589 | pkan    | ood_aod_cwv  | oracle_residual |
| test    | B7     | rho_path  | rho_path      |    13847 | 0.00370171 | 0.0024734  |    0.989308 |      4.42856 |   4.40597 | pkan    | ood_aod_cwv  | oracle_residual |
| test    | B7     | spher_alb | spher_alb     |    13847 | 0.00519603 | 0.00406427 |    0.989339 |      2.81802 |   2.84094 | pkan    | ood_aod_cwv  | oracle_residual |
| test    | B8     | T_total   | T_total       |    13847 | 0.0270946  | 0.0211032  |    0.927902 |      3.05791 |   3.13392 | pkan    | ood_aod_cwv  | oracle_residual |
| test    | B8     | rho_path  | rho_path      |    13847 | 0.00381404 | 0.00263557 |    0.987064 |      5.23696 |   5.29378 | pkan    | ood_aod_cwv  | oracle_residual |
| test    | B8     | spher_alb | spher_alb     |    13847 | 0.00576159 | 0.00469309 |    0.986702 |      3.66285 |   3.73109 | pkan    | ood_aod_cwv  | oracle_residual |
| test    | B8A    | T_total   | T_total       |    13847 | 0.0183535  | 0.0134944  |    0.973398 |      1.7843  |   1.81036 | pkan    | ood_aod_cwv  | oracle_residual |
| test    | B8A    | rho_path  | rho_path      |    13847 | 0.0033721  | 0.00220582 |    0.989182 |      4.77049 |   4.73719 | pkan    | ood_aod_cwv  | oracle_residual |
| test    | B8A    | spher_alb | spher_alb     |    13847 | 0.00507445 | 0.00393668 |    0.988508 |      3.15228 |   3.18526 | pkan    | ood_aod_cwv  | oracle_residual |
| test    | B9     | T_total   | T_total       |    13847 | 0.0246995  | 0.0194777  |    0.943815 |     33.7303  |  44.0942  | pkan    | ood_aod_cwv  | oracle_residual |
| test    | B9     | rho_path  | rho_path      |    13847 | 0.00404861 | 0.00275677 |    0.890001 |     22.4767  |  24.8176  | pkan    | ood_aod_cwv  | oracle_residual |
| test    | B9     | spher_alb | spher_alb     |    13847 | 0.00793933 | 0.00574109 |    0.949644 |     28.2907  |  19.3428  | pkan    | ood_aod_cwv  | oracle_residual |

## CSV Exports

- overall metrics: `combined_metrics.csv`
- band-wise metrics: `band_metrics_table.csv`
- coefficient-wise metrics: `coefficient_metrics_table.csv`
- band x coefficient matrix: `band_coefficient_metrics_table.csv`
- best/worst summaries: `best_worst_band_metrics.csv`, `best_worst_coefficient_metrics.csv`

## Full Per-Target Metrics (All Splits)

| fidelity_mode   | split_mode   | model   | split   | target    |       rmse |         mae |       r2 |       mape |   smape |
|:----------------|:-------------|:--------|:--------|:----------|-----------:|------------:|---------:|-----------:|--------:|
| oracle_residual | ood_aod_cwv  | pkan    | test    | T_total   | 0.021253   | 0.0155031   | 0.990762 | 8213.94    | 7.49922 |
| oracle_residual | ood_aod_cwv  | pkan    | test    | overall   | 0.0136466  | 0.00764949  | 0.990418 | 3517.1     | 7.40311 |
| oracle_residual | ood_aod_cwv  | pkan    | test    | rho_path  | 0.00430145 | 0.00284592  | 0.993581 |   10.4259  | 8.78171 |
| oracle_residual | ood_aod_cwv  | pkan    | test    | spher_alb | 0.00940729 | 0.00459943  | 0.986911 | 2326.94    | 5.92841 |
| oracle_residual | ood_aod_cwv  | pkan    | train   | T_total   | 0.00850785 | 0.00548803  | 0.99849  | 7142.13    | 4.51252 |
| oracle_residual | ood_aod_cwv  | pkan    | train   | overall   | 0.00750629 | 0.00275457  | 0.99329  | 3400.83    | 5.20321 |
| oracle_residual | ood_aod_cwv  | pkan    | train   | rho_path  | 0.00150797 | 0.000989052 | 0.998823 |    8.37525 | 6.60695 |
| oracle_residual | ood_aod_cwv  | pkan    | train   | spher_alb | 0.00971472 | 0.00178663  | 0.982558 | 3051.99    | 4.49015 |
| oracle_residual | ood_aod_cwv  | pkan    | val     | T_total   | 0.00894907 | 0.00568719  | 0.998326 | 7850.69    | 4.55846 |
| oracle_residual | ood_aod_cwv  | pkan    | val     | overall   | 0.00834512 | 0.00285852  | 0.991216 | 3939.06    | 5.29038 |
| oracle_residual | ood_aod_cwv  | pkan    | val     | rho_path  | 0.00156011 | 0.00100922  | 0.998727 |    8.5026  | 6.69894 |
| oracle_residual | ood_aod_cwv  | pkan    | val     | spher_alb | 0.0112429  | 0.00187915  | 0.976595 | 3958       | 4.61373 |