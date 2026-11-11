"""
02 - SWAT Baseline Performance Evaluation
==========================================
Evaluate Q_SWAT against Q_obs using standard hydrological metrics,
following Moriasi et al. (2007) performance ratings.

Study-specific periods for the SWAT stand-alone model (as specified):
    Calibration : 1986-01 to 2005-12  (20 yr, 240 mo)
    Validation  : 2006-01 to 2014-12  ( 9 yr, 108 mo)

NOTE: these SWAT calibration/validation periods are intentionally DIFFERENT
from the ML train/validation/test periods used in Experiments A and B
(see utils_hydro.py and fig0_data_split_scheme.png). This is standard
practice: the SWAT model was calibrated/validated independently and in
advance; the ML layer is then trained and tested on its own, later-focused
temporal split so that the fully independent ML test window (2010-2014)
falls inside the SWAT validation window, allowing a fair, like-for-like
SWAT-vs-ML comparison over that common 2010-2014 period (see Experiments
A & B and the final comparison script).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_hydro import set_plot_style, hydro_metrics, cal_val_split, SWAT_CAL_END

set_plot_style()
DATA = "D:/Flow/Discharge_lag"
FIG = "D:/Flow/Discharge_lag/figs"
TAB = "D:/Flow/Discharge_lag/tables"
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

df = pd.read_csv(f"{DATA}/ElDiem_ML_BASE.csv")
df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m")
df = df.sort_values("Date").reset_index(drop=True)


def rating(nse):
    if nse > 0.75: return "Very good"
    if nse > 0.65: return "Good"
    if nse > 0.50: return "Satisfactory"
    return "Unsatisfactory"


# ---------------------------------------------------------------
# Calibration (1986-2005) / Validation (2006-2014) split
# ---------------------------------------------------------------
cal, val = cal_val_split(df, cal_end=SWAT_CAL_END)
print(f"Calibration: {cal.shape[0]} obs ({cal.Date.min().date()}-{cal.Date.max().date()})")
print(f"Validation:  {val.shape[0]} obs ({val.Date.min().date()}-{val.Date.max().date()})")

m_all = hydro_metrics(df["Q_obs"], df["Q_SWAT"])
m_cal = hydro_metrics(cal["Q_obs"], cal["Q_SWAT"])
m_val = hydro_metrics(val["Q_obs"], val["Q_SWAT"])

results = pd.DataFrame(
    {"Full_period(1986-2014)": m_all,
     "Calibration(1986-2005)": m_cal,
     "Validation(2006-2014)": m_val}
).T
results["NSE_rating"] = results["NSE"].apply(rating)
results = results[["NSE", "NSE_rating", "R2", "RMSE", "MAE", "PBIAS", "KGE"]].round(3)
results.to_csv(f"{TAB}/table4_SWAT_performance.csv")
print(results)

# ---------------------------------------------------------------
# Plot 1: full hydrograph with Cal/Val shading + 1:1 scatter (full period)
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
axes[0].axvspan(df["Date"].min(), pd.Timestamp(SWAT_CAL_END), color="#4C72B0", alpha=0.08, lw=0)
axes[0].axvspan(pd.Timestamp(SWAT_CAL_END), df["Date"].max(), color="#DD8452", alpha=0.10, lw=0)
axes[0].axvline(pd.Timestamp(SWAT_CAL_END), color="black", linestyle=":", linewidth=1.8, alpha=0.75, zorder=3)
axes[0].plot(df["Date"], df["Q_obs"], color="black", lw=2.2, label="Q_obs")
axes[0].plot(df["Date"], df["Q_SWAT"], color="red", lw=1.8, alpha=0.8, label="Q_SWAT")
axes[0].set_ylabel("Discharge (m$^3$/s)")
axes[0].set_xlabel("Year")
axes[0].set_title(f"SWAT Simulated vs. Observed Discharge\nFull period NSE={m_all['NSE']:.2f}, "
                   f"R2={m_all['R2']:.2f}, PBIAS={m_all['PBIAS']:.1f}%")
cal_patch = mpatches.Patch(color="#4C72B0", alpha=0.25, label="Calibration")
val_patch = mpatches.Patch(color="#DD8452", alpha=0.25, label="Validation")
handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(handles + [cal_patch, val_patch], labels + [cal_patch.get_label(), val_patch.get_label()],
               frameon=True, framealpha=0.9, fontsize=12.5, loc="upper right")

axes[1].scatter(cal["Q_obs"], cal["Q_SWAT"], s=22, alpha=0.6, color="#4C72B0", label="Calibration")
axes[1].scatter(val["Q_obs"], val["Q_SWAT"], s=22, alpha=0.6, color="#DD8452", label="Validation")
lims = [0, max(df["Q_obs"].max(), df["Q_SWAT"].max()) * 1.05]
axes[1].plot(lims, lims, "k--", lw=1.8, label="1:1 line")
axes[1].set_xlim(lims); axes[1].set_ylim(lims)
axes[1].set_xlabel("Q_obs (m$^3$/s)"); axes[1].set_ylabel("Q_SWAT (m$^3$/s)")
axes[1].set_title("Observed vs. Simulated Scatter")
axes[1].legend(frameon=False, fontsize=12)
plt.tight_layout()
plt.savefig(f"{FIG}/fig4_SWAT_performance.png", bbox_inches="tight")
plt.close()
print("Saved: fig4_SWAT_performance.png")

print("\n=== 02_swat_evaluation.py complete ===")
