# Best Final Combination for Paper

## Main Paper Setup

### Tables
- Standard overall accuracy
- OOD overall accuracy

### Figures
- Standard vs OOD slope chart
- Coefficient-wise accuracy figure
- Band-wise heatmap
- Predicted vs true scatter for best model

This gives the strongest non-redundant balance for the main paper.

## What Not To Do

Avoid redundant pairs such as:
- A table with overall RMSE/MAE/R2/SMAPE for the same models
- Plus a grouped bar chart showing those exact same metrics for the same models

That is typically redundant in papers.

## Better Reporting Pattern

- Keep full exact values in tables.
- Use figures to show a different perspective, such as:
  - Standard vs OOD gap
  - Coefficient-level behavior
  - Band-level behavior

## Paper Folder Convention

Use this folder for paper assets and planning:
- `results/for_paper/`

Suggested organization:
- `results/for_paper/tables/`
- `results/for_paper/figures/`
- `results/for_paper/notes/`
