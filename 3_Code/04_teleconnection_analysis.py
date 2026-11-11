# -*- coding: utf-8 -*-
"""
04_teleconnection_analysis.py
==============================
Climate teleconnection (ENSO / IOD / NAO) attribution analysis for
El Diem monthly discharge and basin rainfall:

 1. Pearson correlation of Q_obs and P against ENSO/IOD at lags 0, 1, 3 months
    (uses the pre-computed lagged indices in ElDiem_Clim.csv)
 2. Standardized anomaly time-series overlay (visual teleconnection check)
 3. Composite analysis: mean discharge anomaly in El Nino / Neutral / La Nina
    and positive/negative IOD years (Kiremt-season, JJAS, focus)

Outputs
-------
- tables/04_teleconnection_correlations.csv
- tables/04_ENSO_IOD_composite.csv
- figures/04_correlation_heatmap.png
- figures/04_anomaly_timeseries.png
- figures/04_composite_boxplot.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from config import FIG_DIR, TAB_DIR, set_publication_style, COLORS
from data_prep import build_master, _read
from config import F_CLIM

set_publication_style()

master = build_master()
clim = _read(F_CLIM)  # Date, ENSO_t, ENSO_t-1, ENSO_t-3, IOD_t, IOD_t-1, IOD_t-3

df = master.merge(clim, on="Date", how="left")

# ---------------------------------------------------------------------
# 1. Pearson correlations: Q_obs & P vs ENSO/IOD at lags 0/1/3 months,
#    plus NAO (lag 0, from master) for completeness.
#
#    IMPORTANT: raw monthly discharge/rainfall are dominated by the
#    strong annual (Kiremt) seasonal cycle, which swamps any
#    interannual teleconnection signal. Following standard climate-
#    teleconnection practice (e.g. Nicholson, 2017; Korecha & Barnston,
#    2007), correlations are computed on DESEASONALIZED ANOMALIES
#    (monthly value minus its 1986-2014 monthly climatology) rather
#    than on the raw series.
# ---------------------------------------------------------------------
def deseasonalize(s, month):
    clim_mean = s.groupby(month).transform("mean")
    return s - clim_mean

df["Q_obs_anom"] = deseasonalize(df["Q_obs"], df["Date"].dt.month)
df["PCP_anom"] = deseasonalize(df["P"], df["Date"].dt.month)

lag_cols = ["ENSO_t", "ENSO_t-1", "ENSO_t-3", "IOD_t", "IOD_t-1", "IOD_t-3"]
targets = {"Q_obs anomaly ": df["Q_obs_anom"],
           "PCP anomaly ": df["PCP_anom"]}
# (deseasonalized)
rows = []
for tname, series in targets.items():
    for col in lag_cols + ["NAO"]:
        r, p = stats.pearsonr(series, df[col])
        rows.append({"Target": tname, "Climate_index": col, "r": round(r, 3),
                      "p_value": round(p, 4), "Significant_0.05": p < 0.05})
corr_tbl = pd.DataFrame(rows)
corr_tbl.to_csv(os.path.join(TAB_DIR, "04_teleconnection_correlations.csv"), index=False)
print("=== Teleconnection correlations (Pearson r) ===")
print(corr_tbl.to_string(index=False))

# ---------------------------------------------------------------------
# 2. Correlation heatmap
# ---------------------------------------------------------------------
pivot_r = corr_tbl.pivot(index="Climate_index", columns="Target", values="r").loc[lag_cols + ["NAO"]]
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(pivot_r.values, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
ax.set_xticks(range(len(pivot_r.columns)))
ax.set_xticklabels(pivot_r.columns, rotation=0, ha="center")
ax.set_yticks(range(len(pivot_r.index)))
ax.set_yticklabels(pivot_r.index)
for i in range(pivot_r.shape[0]):
    for j in range(pivot_r.shape[1]):
        val = pivot_r.values[i, j]
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                 color="white" if abs(val) > 0.35 else "black", fontsize=16, fontweight="bold")
cbar = fig.colorbar(im, ax=ax, shrink=1)
cbar.set_label("Pearson r")
ax.set_title("Climate Teleconnection Correlations \u2013 El Diem")
for spine in ax.spines.values():
    spine.set_linewidth(2.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "04_correlation_heatmap.png"))
plt.close(fig)

# ---------------------------------------------------------------------
# 3. Standardized anomaly time series: Q_obs vs ENSO_t vs IOD_t
# ---------------------------------------------------------------------
def zscore(s):
    return (s - s.mean()) / s.std()

fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

axes[0].plot(df["Date"], zscore(df["Q_obs_anom"]), color=COLORS["Q_obs"], lw=2.5, label="Discharge anomaly (z)")
axes[0].plot(df["Date"], df["ENSO_t"], color=COLORS["ENSO"], lw=2.5, ls="--", label="ENSO (Ni\u00f1o 3.4)")
axes[0].axhline(0, color="gray", lw=1.5)
axes[0].set_ylabel("Standardized anomaly")
axes[0].set_title("Discharge Anomaly vs. ENSO")
axes[0].legend(loc="upper right", ncol=2, frameon=True, fontsize=16)
for spine in axes[0].spines.values():
    spine.set_linewidth(2.5)

axes[1].plot(df["Date"], zscore(df["Q_obs_anom"]), color=COLORS["Q_obs"], lw=2.5, label="Discharge anomaly (z)")
axes[1].plot(df["Date"], df["IOD_t"], color=COLORS["IOD"], lw=2.5, ls="--", label="IOD (DMI)")
axes[1].axhline(0, color="gray", lw=1.5)
axes[1].set_ylabel("Standardized anomaly")
axes[1].set_xlabel("Year")
axes[1].set_title("Discharge Anomaly vs. IOD")
axes[1].legend(loc="upper right", ncol=2, frameon=True, fontsize=16)
for spine in axes[1].spines.values():
    spine.set_linewidth(2.5)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "04_anomaly_timeseries.png"))
plt.close(fig)

# ---------------------------------------------------------------------
# 4. Composite analysis: Kiremt-season (Jun-Sep) discharge under
#    El Nino / Neutral / La Nina and +/- IOD conditions
# ---------------------------------------------------------------------
df["Month"] = df["Date"].dt.month
df["Year"] = df["Date"].dt.year
kiremt = df[df["Month"].isin([6, 7, 8, 9])].copy()

kiremt["ENSO_phase"] = pd.cut(kiremt["ENSO_t"], bins=[-10, -0.5, 0.5, 10],
                                labels=["La Ni\u00f1a (<-0.5)", "Neutral", "El Ni\u00f1o (>0.5)"])
kiremt["IOD_phase"] = pd.cut(kiremt["IOD_t"], bins=[-10, -0.4, 0.4, 10],
                               labels=["Negative IOD", "Neutral", "Positive IOD"])

composite = kiremt.groupby("ENSO_phase", observed=True)["Q_obs"].agg(["mean", "std", "count"]).reset_index()
composite.to_csv(os.path.join(TAB_DIR, "04_ENSO_IOD_composite.csv"), index=False)
print("\n=== Kiremt-season (JJAS) discharge composite by ENSO phase ===")
print(composite.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
order_enso = ["La Ni\u00f1a (<-0.5)", "Neutral", "El Ni\u00f1o (>0.5)"]
data_enso = [kiremt.loc[kiremt["ENSO_phase"] == g, "Q_obs"].dropna() for g in order_enso]
bp1 = axes[0].boxplot(data_enso, tick_labels=order_enso, patch_artist=True,
                        boxprops=dict(linewidth=2.5), whiskerprops=dict(linewidth=2.5),
                        capprops=dict(linewidth=2.5), medianprops=dict(linewidth=2.5, color="black"))
for patch, c in zip(bp1["boxes"], ["#9ecae1", "#dddddd", "#fc9272"]):
    patch.set_facecolor(c)
axes[0].set_ylabel(r"Kiremt discharge (m$^3$/s)")
axes[0].set_title("Discharge by ENSO Phase (JJAS)")
plt.setp(axes[0].get_xticklabels(), rotation=20, ha="right")
for spine in axes[0].spines.values():
    spine.set_linewidth(2.5)

order_iod = ["Negative IOD", "Neutral", "Positive IOD"]
data_iod = [kiremt.loc[kiremt["IOD_phase"] == g, "Q_obs"].dropna() for g in order_iod]
bp2 = axes[1].boxplot(data_iod, tick_labels=order_iod, patch_artist=True,
                        boxprops=dict(linewidth=2.5), whiskerprops=dict(linewidth=2.5),
                        capprops=dict(linewidth=2.5), medianprops=dict(linewidth=2.5, color="black"))
for patch, c in zip(bp2["boxes"], ["#9ecae1", "#dddddd", "#fc9272"]):
    patch.set_facecolor(c)
axes[1].set_title("Discharge by IOD Phase (JJAS)")
plt.setp(axes[1].get_xticklabels(), rotation=20, ha="right")
for spine in axes[1].spines.values():
    spine.set_linewidth(2.5)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "04_composite_boxplot.png"))
plt.close(fig)

print("\nTeleconnection analysis complete. Tables and figures written to OUTPUT_DIR.")
