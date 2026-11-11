# -*- coding: utf-8 -*-
"""
01_swat_standalone.py
======================
Evaluates the SWAT stand-alone (physically-based) monthly discharge
simulation at El Diem against observed discharge.

Calibration : 1986-01 to 2005-12
Validation  : 2006-01 to 2014-12

Outputs
-------
- tables/01_SWAT_performance.csv
- figures/01_SWAT_hydrograph.png
- figures/01_SWAT_scatter.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from config import (F_QOB_SWAT, FIG_DIR, TAB_DIR, set_publication_style, COLORS,
                     SWAT_CAL_START, SWAT_CAL_END, SWAT_VAL_START, SWAT_VAL_END)
from metrics_utils import metrics_table
from data_prep import _read, split_by_date

set_publication_style()

# ---------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------
df = _read(F_QOB_SWAT)  # Date, Q_obs, Q_SWAT

cal = split_by_date(df, SWAT_CAL_START, SWAT_CAL_END)
val = split_by_date(df, SWAT_VAL_START, SWAT_VAL_END)

# ---------------------------------------------------------------------
# 2. Performance metrics
# ---------------------------------------------------------------------
tbl = metrics_table([
    ("Calibration (1986-2005)", cal["Q_obs"], cal["Q_SWAT"]),
    ("Validation (2006-2014)", val["Q_obs"], val["Q_SWAT"]),
    ("Full period (1986-2014)", df["Q_obs"], df["Q_SWAT"]),
])
tbl.to_csv(os.path.join(TAB_DIR, "01_SWAT_performance.csv"), index=False)
print("\n=== SWAT Stand-Alone Performance ===")
print(tbl.to_string(index=False))

# ---------------------------------------------------------------------
# 3. Hydrograph (observed vs simulated), calibration/validation shaded
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 6))

ax.axvspan(cal["Date"].min(), cal["Date"].max(), color=COLORS["cal"], alpha=0.5,
           label="Calibration period", zorder=0)
ax.axvspan(val["Date"].min(), val["Date"].max(), color=COLORS["val"], alpha=0.5,
           label="Validation period", zorder=0)

ax.plot(df["Date"], df["Q_obs"], color=COLORS["Q_obs"], lw=2.5, label="Observed ($Q_{obs}$)")
ax.plot(df["Date"], df["Q_SWAT"], color=COLORS["Q_SWAT"], lw=2.5, ls="--", label="SWAT simulated ($Q_{SWAT}$)")

ax.set_xlabel("Year")
ax.set_ylabel(r"Discharge (m$^3$/s)")
ax.set_title("SWAT Stand-Alone Monthly Discharge Simulation \u2013 El Diem")
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.legend(loc="upper right", frameon=True, ncol=2)
for spine in ax.spines.values():
    spine.set_linewidth(2.5)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "01_SWAT_hydrograph.png"))
plt.close(fig)

# ---------------------------------------------------------------------
# 4. Scatter plot: observed vs simulated (cal vs val colored)
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(cal["Q_obs"], cal["Q_SWAT"], s=60, color=COLORS["cal"].replace("#", "#"),
           edgecolor="darkgreen", linewidth=1.2, alpha=0.85, label="Calibration")
ax.scatter(val["Q_obs"], val["Q_SWAT"], s=60, color=COLORS["val"],
           edgecolor="darkorange", linewidth=1.2, alpha=0.85, label="Validation")

lims = [0, max(df["Q_obs"].max(), df["Q_SWAT"].max()) * 1.05]
ax.plot(lims, lims, color="black", lw=2.5, ls="-", label="1:1 line")

# Combined regression line
coef = np.polyfit(df["Q_obs"], df["Q_SWAT"], 1)
xr = np.array(lims)
ax.plot(xr, coef[0] * xr + coef[1], color="red", lw=2.5, ls="--",
         label=f"Fit: y={coef[0]:.2f}x+{coef[1]:.1f}")

ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel(r"Observed discharge (m$^3$/s)")
ax.set_ylabel(r"SWAT simulated discharge (m$^3$/s)")
ax.set_title("SWAT Observed vs. Simulated \u2013 El Diem")
ax.legend(loc="upper left", frameon=True)
for spine in ax.spines.values():
    spine.set_linewidth(2.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "01_SWAT_scatter.png"))
plt.close(fig)

print("\nSWAT stand-alone evaluation complete. Figures and table written to OUTPUT_DIR.")
