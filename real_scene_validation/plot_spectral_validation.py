#!/usr/bin/env python3
"""
Publication-quality spectral validation figure for the pKANrtm paper.
Gobabeb (GONA) RadCalNet site, 2018-05-25, Sentinel-2A T33KWP.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# ── Data ──────────────────────────────────────────────────────────────────────

bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
wvl = np.array([520, 560, 654, 701, 743, 779, 789, 871, 1639, 2256], dtype=float)

toa     = np.array([0.179, 0.203, 0.277, 0.290, 0.311, 0.329, 0.306, 0.328, 0.397, 0.326])
l2a     = np.array([0.140, 0.205, 0.290, 0.323, 0.332, 0.338, 0.336, 0.332, 0.433, 0.385])
rc      = np.array([0.140, 0.206, 0.287, 0.305, 0.324, 0.334, 0.331, 0.327, 0.408, 0.350])
s6      = np.array([0.158, 0.217, 0.303, 0.323, 0.341, 0.352, 0.346, 0.338, 0.428, 0.375])
pkan    = np.array([0.141, 0.204, 0.289, 0.303, 0.326, 0.336, 0.329, 0.326, 0.412, 0.354])

# ── Style ─────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "0.75",
    "legend.fontsize": 8.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
})

# ── Figure ────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(5.5, 3.8))

ax.plot(wvl, toa, marker="v", markersize=4.5, linewidth=0.9,
        color="#AAAAAA", markeredgecolor="#888888", markeredgewidth=0.4,
        linestyle="--", label="TOA (L1C)", zorder=2)

ax.plot(wvl, l2a, marker="s", markersize=4.5, linewidth=1.0,
        color="#1f77b4", markeredgecolor="white", markeredgewidth=0.3,
        linestyle="-", label="ESA Sen2Cor L2A", zorder=3)

ax.plot(wvl, s6, marker="^", markersize=5, linewidth=1.0,
        color="#d62728", markeredgecolor="white", markeredgewidth=0.3,
        linestyle="-", label="6S-only", zorder=3)

ax.plot(wvl, pkan, marker="D", markersize=4.5, linewidth=1.1,
        color="#2ca02c", markeredgecolor="white", markeredgewidth=0.3,
        linestyle="-", label="6S + pKANrtm", zorder=4)

ax.plot(wvl, rc, marker="o", markersize=6.5, linewidth=2.0,
        color="black", markerfacecolor="white", markeredgecolor="black",
        markeredgewidth=1.2, linestyle="-", label="RadCalNet BOA",
        zorder=5)

# band labels along the RadCalNet curve
for i, b in enumerate(bands):
    offset_y = 8
    if b in ("B08", "B8A"):
        offset_y = -12
    if b == "B07":
        offset_y = 10
    ax.annotate(b, (wvl[i], rc[i]),
                textcoords="offset points", xytext=(0, offset_y),
                fontsize=6.5, ha="center", color="0.35",
                fontstyle="italic")

ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Reflectance")

ax.set_xlim(460, 2340)
ax.set_ylim(0.08, 0.48)

ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

ax.grid(True, which="major", linewidth=0.35, color="0.82", zorder=0)
ax.grid(True, which="minor", linewidth=0.18, color="0.90", zorder=0)

ax.legend(loc="upper left", borderpad=0.5, handlelength=2.2,
          handletextpad=0.6, labelspacing=0.35)

ax.text(0.98, 0.03,
        "Gobabeb (GONA) · 2018-05-25 · S2A T33KWP",
        transform=ax.transAxes, fontsize=7, color="0.45",
        ha="right", va="bottom")

fig.tight_layout()

# ── Save ──────────────────────────────────────────────────────────────────────

outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(outdir, exist_ok=True)

fig.savefig(os.path.join(outdir, "gona_radcalnet_spectral_validation.pdf"))
fig.savefig(os.path.join(outdir, "gona_radcalnet_spectral_validation.png"))
plt.close(fig)

print(f"Saved to {outdir}/gona_radcalnet_spectral_validation.{{pdf,png}}")
