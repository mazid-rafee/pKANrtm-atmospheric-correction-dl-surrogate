"""
Build Standard and OOD overall-accuracy tables (paper reporting plan § tables).

Writes CSV files under ``results/for_paper/tables/`` with ``kan`` and ``pkan``
(five ``kan_arch`` variants each → 10 rows) and RMSE, MAE, R², SMAPE as metric
columns (oracle residual).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

FIDELITY = "oracle_residual"
MODELS = ("kan", "pkan")
VARIANT_ORDER = (
    "baseline",
    "balanced_deep",
    "large_deep",
    "shared_trunk_multihead",
    "small",
)
METRIC_COLS = ("rmse", "mae", "r2", "smape")


def parse_md_tables(md_path: Path) -> list[pd.DataFrame]:
    lines = md_path.read_text(encoding="utf-8").splitlines()
    tables: list[pd.DataFrame] = []
    i = 0
    while i < len(lines) - 1:
        l = lines[i].strip()
        l2 = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if l.startswith("|") and l.endswith("|") and l2.startswith("|") and ("---" in l2 or ":---" in l2):
            headers = [c.strip() for c in l.strip("|").split("|")]
            rows = []
            j = i + 2
            while j < len(lines):
                r = lines[j].strip()
                if not (r.startswith("|") and r.endswith("|")):
                    break
                vals = [c.strip() for c in r.strip("|").split("|")]
                if len(vals) == len(headers):
                    rows.append(vals)
                j += 1
            tables.append(pd.DataFrame(rows, columns=headers))
            i = j
        else:
            i += 1
    return tables


def load_overall_accuracy(md_path: Path) -> pd.DataFrame:
    want = {"fidelity_mode", "split_mode", "model", "rmse", "mae", "r2", "smape", "kan_arch"}
    chosen = None
    for t in parse_md_tables(md_path):
        cols = set(t.columns)
        if not want.issubset(cols) or "band" in cols or "coefficient" in cols:
            continue
        if len(t) == 20:
            chosen = t
            break
        if chosen is None or abs(len(t) - 20) < abs(len(chosen) - 20):
            chosen = t
    if chosen is None:
        raise RuntimeError("Could not find overall Accuracy markdown table.")
    return chosen


def build_split_table(ov: pd.DataFrame, split_mode: str) -> pd.DataFrame:
    sub = ov[
        (ov["fidelity_mode"] == FIDELITY)
        & (ov["split_mode"] == split_mode)
        & (ov["model"].isin(MODELS))
    ].copy()
    for c in METRIC_COLS:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    rows = []
    for model in MODELS:
        for arch in VARIANT_ORDER:
            r = sub[(sub["model"] == model) & (sub["kan_arch"] == arch)]
            if r.empty:
                rows.append(
                    {"model": model, "kan_arch": arch, **{c: float("nan") for c in METRIC_COLS}}
                )
            else:
                row = r.iloc[0]
                rows.append(
                    {"model": model, "kan_arch": arch, **{c: float(row[c]) for c in METRIC_COLS}}
                )
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--accuracy_md", type=Path, default=Path("results/inference_accuracy_benchmark.md"))
    p.add_argument("--out_dir", type=Path, default=Path("results/for_paper/tables"))
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    md_path = args.accuracy_md if args.accuracy_md.is_absolute() else root / args.accuracy_md
    out_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir

    ov = load_overall_accuracy(md_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    std = build_split_table(ov, "standard")
    ood = build_split_table(ov, "ood_aod_cwv")

    p_std = out_dir / "table_standard_overall_accuracy_variants.csv"
    p_ood = out_dir / "table_ood_overall_accuracy_variants.csv"
    std.to_csv(p_std, index=False)
    ood.to_csv(p_ood, index=False)
    print(p_std)
    print(p_ood)


if __name__ == "__main__":
    main()
