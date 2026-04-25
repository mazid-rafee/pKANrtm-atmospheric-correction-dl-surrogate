"""
Coefficient-wise RMSE grouped bar chart (Standard vs OOD), one PNG per model × KAN variant.

Matches the style of ``fig_coefficient_wise_accuracy.png`` (grouped bars by coefficient).

Reads ``results/inference_accuracy_benchmark.md`` (Coefficient-wise Metrics table) and writes
``coefficient_wise_accuracy/{model}/fig_coefficient_wise_accuracy__{kan_arch}.png`` under the
figures output directory (default ``results/for_paper/figures/``).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
COEF_ORDER = ("T_total", "rho_path", "spher_alb")
SPLITS = ("standard", "ood_aod_cwv")
SPLIT_LABELS = {"standard": "standard", "ood_aod_cwv": "ood_aod_cwv"}


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


def load_coefficient_wise_table(md_path: Path) -> pd.DataFrame:
    want = {"split", "target", "coefficient", "n_rows", "rmse", "model", "split_mode", "kan_arch"}
    tables = parse_md_tables(md_path)
    best = None
    for t in tables:
        cols = set(t.columns)
        if not want.issubset(cols):
            continue
        if "band" in cols:
            continue
        if best is None or len(t) > len(best):
            best = t
    if best is None:
        raise RuntimeError("No Coefficient-wise Metrics table found in markdown.")
    return best


def plot_one(df: pd.DataFrame, out_path: Path, dpi: int = 180) -> None:
    """df: rows for one model + kan_arch + oracle; includes both split_mode values."""
    rmse_by: dict[tuple[str, str], float] = {}
    for _, r in df.iterrows():
        coef = str(r["coefficient"])
        sp = str(r["split_mode"])
        rmse_by[(coef, sp)] = float(r["rmse"])

    x = np.arange(len(COEF_ORDER))
    width = 0.36
    std_vals = [rmse_by.get((c, "standard"), np.nan) for c in COEF_ORDER]
    ood_vals = [rmse_by.get((c, "ood_aod_cwv"), np.nan) for c in COEF_ORDER]

    fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
    ax.bar(x - width / 2, std_vals, width, label=SPLIT_LABELS["standard"], color="#1f77b4", edgecolor="white", linewidth=0.5)
    ax.bar(x + width / 2, ood_vals, width, label=SPLIT_LABELS["ood_aod_cwv"], color="#ff7f0e", edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(list(COEF_ORDER))
    ax.set_ylabel("Mean RMSE")
    ax.set_xlabel("")
    ax.legend(frameon=True, loc="upper left")
    ax.grid(axis="y", linestyle="-", alpha=0.35)
    ax.set_axisbelow(True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--accuracy_md",
        type=Path,
        default=Path("results/inference_accuracy_benchmark.md"),
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path("results/for_paper/figures"),
    )
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    md_path = args.accuracy_md if args.accuracy_md.is_absolute() else root / args.accuracy_md
    out_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir

    coef_df = load_coefficient_wise_table(md_path)
    coef_df["rmse"] = pd.to_numeric(coef_df["rmse"], errors="coerce")

    base = coef_df[
        (coef_df["fidelity_mode"] == FIDELITY)
        & (coef_df["model"].isin(MODELS))
        & (coef_df["split_mode"].isin(SPLITS))
        & (coef_df["coefficient"].isin(COEF_ORDER))
    ].copy()

    for model in MODELS:
        for arch in VARIANT_ORDER:
            sub = base[(base["model"] == model) & (base["kan_arch"] == arch)].copy()
            if len(sub) < len(COEF_ORDER) * len(SPLITS):
                print("skip incomplete", model, arch, "n=", len(sub))
                continue
            out = out_dir / "coefficient_wise_accuracy" / model / f"fig_coefficient_wise_accuracy__{arch}.png"
            plot_one(sub, out)
            print(out)


if __name__ == "__main__":
    main()
