"""
Build band×coefficient SMAPE heatmaps (standard | OOD) per model and KAN architecture variant.

Reads ``results/inference_accuracy_benchmark.md`` (Band x Coefficient Metrics table) and writes
``band_wise_heatmap/{model}/fig_band_wise_heatmap_both_splits__{kan_arch}.png`` under the output
directory (default ``results/for_paper/figures/``).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

CMAP = "RdYlGn_r"
FIDELITY = "oracle_residual"
MODELS = ("kan", "pkan")
VARIANT_ORDER = (
    "baseline",
    "balanced_deep",
    "large_deep",
    "shared_trunk_multihead",
    "small",
)
BAND_ORDER = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B9",
    "B10",
    "B11",
    "B12",
]


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


def load_band_coeff_table(md_path: Path) -> pd.DataFrame:
    tables = parse_md_tables(md_path)
    want = {"split", "band", "target", "coefficient", "n_rows", "smape", "model", "split_mode", "kan_arch"}
    best = None
    for t in tables:
        if want.issubset(set(t.columns)) and "fidelity_mode" in t.columns:
            if best is None or len(t) > len(best):
                best = t
    if best is None:
        raise RuntimeError("No Band x Coefficient style table found in markdown.")
    return best


def _fmt_smape_tick(z_log1p: float, pos: int | None = None) -> str:
    s = float(np.expm1(z_log1p))
    if not np.isfinite(s):
        return ""
    if s < 1:
        return f"{s:.2f}"
    if s < 10:
        return f"{s:.2f}"
    if s < 100:
        return f"{s:.1f}"
    return f"{s:.0f}"


def _nice_smape_ticks(vmin_l: float, vmax_l: float, max_ticks: int = 8) -> np.ndarray:
    smin, smax = float(np.expm1(vmin_l)), float(np.expm1(vmax_l))
    lo = max(0.0, smin * 0.95)
    hi = smax * 1.02
    if hi <= lo:
        return np.array([vmin_l, vmax_l])
    anchors = np.geomspace(max(lo, 1e-6), hi, num=40)
    z = np.log1p(anchors)
    z = z[(z >= vmin_l) & (z <= vmax_l)]
    if len(z) < 2:
        return np.linspace(vmin_l, vmax_l, 5)
    idx = np.linspace(0, len(z) - 1, num=min(max_ticks, len(z)), dtype=int)
    ticks = np.unique(np.clip(z[idx], vmin_l, vmax_l))
    ticks = np.sort(ticks)
    if ticks[0] > vmin_l + 1e-9:
        ticks = np.concatenate([[vmin_l], ticks])
    if ticks[-1] < vmax_l - 1e-9:
        ticks = np.concatenate([ticks, [vmax_l]])
    ticks = np.unique(np.clip(ticks, vmin_l, vmax_l))
    return _dedupe_close_log1p(ticks, eps=0.055)


def _dedupe_close_log1p(tick_z: np.ndarray, eps: float = 0.055) -> np.ndarray:
    z = np.sort(np.unique(np.asarray(tick_z, dtype=float)))
    out = [float(z[0])]
    for t in z[1:]:
        if t - out[-1] < eps:
            continue
        out.append(float(t))
    return np.array(out, dtype=float)


def _drop_crowded_end_ticks(tick_z: np.ndarray) -> np.ndarray:
    z = np.sort(np.unique(np.asarray(tick_z, dtype=float)))
    if len(z) < 4:
        return z
    z = np.concatenate([[z[0]], z[2:]])
    if len(z) >= 3:
        z = np.concatenate([z[:-2], [z[-1]]])
    return z


def pivot_log1p_smape(
    bdf: pd.DataFrame, split_mode: str, band_order: list[str], coef_order: list[str]
) -> tuple[pd.DataFrame, np.ndarray]:
    x = (
        bdf[bdf["split_mode"] == split_mode]
        .groupby(["band", "coefficient"], as_index=False)["smape"]
        .mean()
        .pivot(index="band", columns="coefficient", values="smape")
    )
    x = x.reindex(index=[b for b in band_order if b in x.index], columns=coef_order)
    arr_log = np.log1p(x.values.astype(float))
    return x, arr_log


def plot_one(
    bdf: pd.DataFrame,
    out_path: Path,
    figsize: tuple[float, float] = (12, 5),
    dpi: int = 180,
) -> None:
    band_order = [b for b in BAND_ORDER if b in set(bdf["band"].astype(str))]
    coef_order = sorted(bdf["coefficient"].dropna().unique().tolist())
    arrs: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
    vmin_l = vmax_l = None
    for sp in ("standard", "ood_aod_cwv"):
        x, arr_log = pivot_log1p_smape(bdf, sp, band_order, coef_order)
        arrs[sp] = (x, arr_log)
        fin = arr_log[np.isfinite(arr_log)]
        if fin.size:
            vmin_l = float(fin.min()) if vmin_l is None else min(vmin_l, float(fin.min()))
            vmax_l = float(fin.max()) if vmax_l is None else max(vmax_l, float(fin.max()))

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    im_last = None
    for ax, sp, split_title in zip(
        axes, ("standard", "ood_aod_cwv"), ("Standard split", "OOD split")
    ):
        x, arr_log = arrs[sp]
        im = ax.imshow(arr_log, aspect="auto", vmin=vmin_l, vmax=vmax_l, cmap=CMAP)
        im_last = im
        ax.set_yticks(np.arange(len(x.index)))
        ax.set_yticklabels(list(x.index))
        ax.set_xticks(np.arange(len(x.columns)))
        ax.set_xticklabels(list(x.columns), rotation=0, ha="center")
        ax.set_title(split_title)
        ax.set_xticks(np.arange(-0.5, len(x.columns), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(x.index), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.6, alpha=0.75)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_xlabel("")
        ax.set_ylabel("")

    assert im_last is not None
    cbar = fig.colorbar(im_last, ax=axes.ravel().tolist(), shrink=0.95)
    tick_z = _nice_smape_ticks(vmin_l, vmax_l, max_ticks=9)
    tick_z = _drop_crowded_end_ticks(tick_z)
    cbar.set_ticks(tick_z)
    cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_smape_tick))
    cbar.set_label("SMAPE")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--accuracy_md",
        type=Path,
        default=Path("results/inference_accuracy_benchmark.md"),
        help="Markdown export with Band x Coefficient table.",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path("results/for_paper/figures"),
        help="Directory for PNG outputs.",
    )
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    md_path = args.accuracy_md if args.accuracy_md.is_absolute() else root / args.accuracy_md
    out_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir

    bandcoef = load_band_coeff_table(md_path)
    bandcoef["smape"] = pd.to_numeric(bandcoef["smape"], errors="coerce")

    base = bandcoef[
        (bandcoef["fidelity_mode"] == FIDELITY) & (bandcoef["model"].isin(MODELS))
    ].copy()

    for model in MODELS:
        for arch in VARIANT_ORDER:
            bdf = base[(base["model"] == model) & (base["kan_arch"] == arch)].copy()
            if bdf.empty:
                print("skip empty", model, arch)
                continue
            out = out_dir / "band_wise_heatmap" / model / f"fig_band_wise_heatmap_both_splits__{arch}.png"
            plot_one(bdf, out)
            print(out)


if __name__ == "__main__":
    main()
