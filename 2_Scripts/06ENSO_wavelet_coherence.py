"""
06 - Wavelet Coherence Analysis
==================================
Continuous Wavelet Transform (Morlet) and Wavelet Coherence between:
  - Rainfall <-> ENSO (Nino3.4)
  - Discharge <-> IOD (DMI)
following Torrence & Compo (1998) / Grinsted et al. (2004).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pycwt as wavelet
import warnings
import os, sys
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_hydro import set_plot_style

set_plot_style()
DATA = "D:/Flow/Discharge_lag"
FIG = "D:/Flow/Discharge_lag/figs"
TAB = "D:/Flow/Discharge_lag/tables"
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

base = pd.read_csv(f"{DATA}/ElDiem_ML_BASE.csv")
base["Date"] = pd.to_datetime(base["Date"], format="%Y-%m")
base = base.sort_values("Date").reset_index(drop=True)

dt = 1 / 12.0  # monthly data, time step in years


def standardize(x):
    return (x - np.mean(x)) / np.std(x)


def wavelet_coherence(x, y, label_x, label_y, fname):
    x = standardize(np.asarray(x, dtype=float))
    y = standardize(np.asarray(y, dtype=float))
    mother = wavelet.Morlet(6)
    WCT, aWCT, corr_coi, freq, sig = wavelet.wct(
        x, y, dt, sig=False, mother=mother)
    period = 1 / freq
    t = 1986 + np.arange(len(x)) * dt

    fig, ax = plt.subplots(figsize=(11, 5.5))
    levels = np.linspace(0, 1, 11)
    cf = ax.contourf(t, np.log2(period), WCT, levels, cmap="jet", extend="both")
    ax.fill(np.concatenate([t, t[-1:], t[-1:], t[:1], t[:1]]),
            np.concatenate([np.log2(corr_coi), [1e-9], np.log2(period[-1:]),
                             np.log2(period[-1:]), [1e-9]]),
            "k", alpha=0.3, hatch="x")
    yticks = 2.0 ** np.arange(np.floor(np.log2(period.min())), np.ceil(np.log2(period.max())) + 1)
    ax.set_yticks(np.log2(yticks))
    ax.set_yticklabels([f"{v:.2g}" for v in yticks])
    ax.set_ylabel("Period (years)")
    ax.set_xlabel("Year")
    ax.set_title(f"Wavelet Coherence: {label_x} vs {label_y}")
    cbar = plt.colorbar(cf, ax=ax, label="Coherence")
    cbar.ax.tick_params(labelsize=14)
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)
    plt.tight_layout()
    plt.savefig(f"{FIG}/{fname}", bbox_inches="tight", dpi=220)
    plt.close()
    return WCT, period


print("Computing wavelet coherence: Rainfall vs ENSO ...")
WCT1, period1 = wavelet_coherence(base["P"], base["ENSO"], "Rainfall", "ENSO (Niño 3.4)",
                                   "fig12_wavelet_coherence_rainfall_ENSO.png")

print("Computing wavelet coherence: Discharge vs IOD ...")
WCT2, period2 = wavelet_coherence(base["Q_obs"], base["IOD"], "Discharge", "IOD (DMI)",
                                   "fig13_wavelet_coherence_discharge_IOD.png")

print("Computing wavelet coherence: Discharge vs ENSO ...")
WCT3, period3 = wavelet_coherence(base["Q_obs"], base["ENSO"], "Discharge", "ENSO (Niño 3.4)",
                                   "fig14_wavelet_coherence_discharge_ENSO.png")

# Summary of mean coherence by band (interannual 2-7yr = ENSO band; ~3-5yr also IOD)
def band_mean(WCT, period, lo, hi):
    mask = (period >= lo) & (period <= hi)
    return np.nanmean(WCT[mask, :])


summary = pd.DataFrame({
    "Pair": ["Rainfall-ENSO", "Discharge-IOD", "Discharge-ENSO"],
    "Mean_coherence_2-7yr_band": [
        band_mean(WCT1, period1, 2, 7),
        band_mean(WCT2, period2, 2, 7),
        band_mean(WCT3, period3, 2, 7),
    ],
    "Mean_coherence_full_spectrum": [
        np.nanmean(WCT1), np.nanmean(WCT2), np.nanmean(WCT3),
    ],
}).round(3)
summary.to_csv(f"{TAB}/table12_wavelet_coherence_summary.csv", index=False)
print(summary)
print("\n=== 06_wavelet_coherence.py complete ===")
