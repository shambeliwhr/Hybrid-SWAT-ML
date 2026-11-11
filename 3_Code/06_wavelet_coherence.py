# -*- coding: utf-8 -*-
"""
06_wavelet_coherence.py
=========================
Continuous Wavelet Transform (CWT) based Wavelet Coherence analysis,
as specified in the procedure document ("Very Q1-friendly" -- commonly
published in Journal of Hydrology / HESS / STOTEN).

Analyzes periodic co-variability between:
    Fig. 12  Rainfall (P)   <-> ENSO (Ni\u00f1o 3.4)
    Fig. 13  Discharge (Q)  <-> IOD (DMI)
    Fig. 14  Discharge (Q)  <-> ENSO (Ni\u00f1o 3.4)

Method: Grinsted et al. (2004) wavelet coherence (Morlet wavelet),
via the `pycwt` package (Torrence & Compo, 1998 CWT implementation).
Significance tested against red-noise (AR1) background via Monte Carlo.

X-axes on all three figures show every calendar year, rotated 90
degrees, per user request.

Outputs
-------
- figures/fig12_wavelet_coherence_rainfall_ENSO.png
- figures/fig13_wavelet_coherence_discharge_IOD.png
- figures/fig14_wavelet_coherence_discharge_ENSO.png
- tables/06_wavelet_coherence_summary.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import pycwt as wavelet

from config import FIG_DIR, TAB_DIR, set_publication_style, COLORS
from data_prep import build_master, _read
from config import F_CLIM

set_publication_style()

master = build_master()
clim = _read(F_CLIM)
df = master.merge(clim, on="Date", how="left").sort_values("Date").reset_index(drop=True)

dt = 1.0 / 12.0  # monthly data expressed in years
t = np.arange(len(df)) * dt
years = df["Date"].dt.year + (df["Date"].dt.month - 1) / 12.0


def standardize(x):
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / x.std()


def plot_wavelet_coherence(s1, s2, label1, label2, fname, title, seed=42):
    """Compute and plot wavelet coherence between two standardized series."""
    y1 = standardize(s1)
    y2 = standardize(s2)

    np.random.seed(seed)
    WCT, aWCT, coi, freq, sig = wavelet.wct(
        y1, y2, dt, dj=1 / 12, s0=2 * dt, sig=True,
        significance_level=0.90, wavelet="morlet", mc_count=150,
    )
    period = 1.0 / freq  # years

    fig, ax = plt.subplots(figsize=(15, 7))

    # Wavelet coherence power surface
    levels = np.linspace(0, 1, 11)
    cf = ax.contourf(df["Date"], period, WCT, levels=levels, cmap="jet", extend="both")

    # 90% significance contour (against red-noise background).
    # `sig` from pycwt is a 1D per-scale threshold; broadcast to 2D and
    # contour where WCT/threshold crosses 1 (i.e. WCT exceeds its threshold).
    sig_ratio = WCT / sig[:, np.newaxis]
    ax.contour(df["Date"], period, sig_ratio, [1.0], colors="k", linewidths=2.0)

    # Cone of influence (region affected by edge effects) -- hatched out
    ax.plot(df["Date"], coi, color="black", lw=2.5)
    ax.fill_between(df["Date"], coi, period.max(), color="white", alpha=0.55, hatch="//")

    # Phase arrows (subsampled) -- direction of lead/lag relationship
    na, nb = 8, 3   # subsample in time / scale for readability
    ax.quiver(
        df["Date"][::na], period[::nb],
        np.cos(aWCT[::nb, ::na]), np.sin(aWCT[::nb, ::na]),
        units="width", angles="uv", pivot="mid",
        scale=32, headwidth=4, headlength=5, color="black", alpha=0.75,
    )

    ax.set_yscale("log", base=2)
    ax.set_ylim(period.min(), min(period.max(), 16))
    ax.set_ylabel("Period (years)")
    ax.set_title(title)

    # --- X-axis: every calendar year, rotated 90 degrees (per request) ---
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center")
    ax.set_xlabel("Year")
    ax.set_xlim(df["Date"].min(), df["Date"].max())

    for spine in ax.spines.values():
        spine.set_linewidth(2.5)

    cbar = fig.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label("Wavelet coherence")

    fig.text(0.13, 0.85, f"{label1} vs {label2}", fontsize=16,
              bbox=dict(facecolor="white", edgecolor="black", linewidth=1.5, alpha=0.85))

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, fname))
    plt.close(fig)
    print(f"Saved {fname}")
    return WCT, period


# ---------------------------------------------------------------------
# Fig. 12: Rainfall (P) <-> ENSO
# ---------------------------------------------------------------------
WCT_P_ENSO, period = plot_wavelet_coherence(
    df["P"], df["ENSO_t"], "Rainfall (P)", "ENSO (Ni\u00f1o 3.4)",
    "fig12_wavelet_coherence_rainfall_ENSO.png",
    "Wavelet Coherence: Basin Rainfall vs. ENSO \u2013 El Diem",
)

# ---------------------------------------------------------------------
# Fig. 13: Discharge (Q) <-> IOD
# ---------------------------------------------------------------------
WCT_Q_IOD, _ = plot_wavelet_coherence(
    df["Q_obs"], df["IOD_t"], "Discharge (Q)", "IOD (DMI)",
    "fig13_wavelet_coherence_discharge_IOD.png",
    "Wavelet Coherence: Discharge vs. IOD \u2013 El Diem",
)

# ---------------------------------------------------------------------
# Fig. 14: Discharge (Q) <-> ENSO
# ---------------------------------------------------------------------
WCT_Q_ENSO, _ = plot_wavelet_coherence(
    df["Q_obs"], df["ENSO_t"], "Discharge (Q)", "ENSO (Ni\u00f1o 3.4)",
    "fig14_wavelet_coherence_discharge_ENSO.png",
    "Wavelet Coherence: Discharge vs. ENSO \u2013 El Diem",
)

# ---------------------------------------------------------------------
# Summary table: mean coherence by period band (annual / 2-4yr ENSO band / interannual)
# ---------------------------------------------------------------------
def band_summary(WCT, period, bands):
    rows = []
    for name, (lo, hi) in bands.items():
        mask = (period >= lo) & (period <= hi)
        rows.append({"Band": name, "Period_years": f"{lo}-{hi}", "Mean_coherence": round(WCT[mask, :].mean(), 3)})
    return pd.DataFrame(rows)

bands = {"Annual (~1yr)": (0.8, 1.2), "ENSO band (2-7yr)": (2, 7), "Interannual (1-2yr)": (1.2, 2)}

summary = pd.concat([
    band_summary(WCT_P_ENSO, period, bands).assign(Pair="Rainfall-ENSO"),
    band_summary(WCT_Q_IOD, period, bands).assign(Pair="Discharge-IOD"),
    band_summary(WCT_Q_ENSO, period, bands).assign(Pair="Discharge-ENSO"),
], ignore_index=True)[["Pair", "Band", "Period_years", "Mean_coherence"]]

summary.to_csv(os.path.join(TAB_DIR, "06_wavelet_coherence_summary.csv"), index=False)
print("\n=== Wavelet coherence band summary ===")
print(summary.to_string(index=False))
print("\nWavelet coherence analysis complete.")
