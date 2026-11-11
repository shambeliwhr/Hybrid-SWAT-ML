"""
01 - Data Preparation, EDA, and Pearson Correlation Analysis
Upper Blue Nile Basin (El Diem station) - Hybrid SWAT-XGBoost-SHAP framework
============================================================================
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_hydro import set_plot_style, SWAT_CAL_END, SWAT_VAL_END, ML_TRAIN_END, ML_VAL_END

set_plot_style()
DATA = "D:/Flow/Discharge_lag"
FIG = "D:/Flow/Discharge_lag/figs"
TAB = "D:/Flow/Discharge_lag/tables"
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

# ---------------------------------------------------------------
# 1. Load master datasets
# ---------------------------------------------------------------
base = pd.read_csv(f"{DATA}/ElDiem_ML_BASE.csv")
clim = pd.read_csv(f"{DATA}/ElDiem_Clim.csv")

base["Date"] = pd.to_datetime(base["Date"], format="%Y-%m")
clim["Date"] = pd.to_datetime(clim["Date"], format="%Y-%m")
base = base.sort_values("Date").reset_index(drop=True)
clim = clim.sort_values("Date").reset_index(drop=True)

print("Base dataset:", base.shape, base["Date"].min(), "->", base["Date"].max())
print(base.describe().T[["mean", "std", "min", "max"]])

# ---------------------------------------------------------------
# 1b. Data-split scheme figure — SWAT (Cal/Val) vs. ML (Train/Val/Test)
#     Documenting the two DIFFERENT period conventions used later in the
#     workflow, as specified for this study:
#       SWAT stand-alone : Calibration 1986-2005 | Validation 2006-2014
#       ML experiments   : Train 1986-1997 | Validation 1998-2005 | Test 2006-2014
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 3.4))
row_swat, row_ml = 1, 0
bar_h = 0.55

swat_segments = [("Calibration\n1986–2005", "1986-01-01", "2005-12-31", "#4C72B0"),
                  ("Validation\n2006–2014", "2006-01-01", "2014-12-31", "#DD8452")]
ml_segments = [("Train\n1986–1997", "1986-01-01", "1997-12-31", "#55A868"),
               ("Validation\n1998–2005", "1998-01-01", "2005-12-31", "#C44E52"),
               ("Test\n2006–2014", "2006-01-01", "2014-12-31", "#8172B2")]

for label, start, end, color in swat_segments:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    ax.barh(row_swat, (e - s).days, left=s, height=bar_h, color=color, edgecolor="black", linewidth=1.2)
    ax.text(s + (e - s) / 2, row_swat, label, ha="center", va="center", fontsize=13, color="white", fontweight="bold")

for label, start, end, color in ml_segments:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    ax.barh(row_ml, (e - s).days, left=s, height=bar_h, color=color, edgecolor="black", linewidth=1.2)
    ax.text(s + (e - s) / 2, row_ml, label, ha="center", va="center", fontsize=13, color="white", fontweight="bold")

ax.set_yticks([row_ml, row_swat])
ax.set_yticklabels(["ML Experiments\n(A & B)", "SWAT\nStand-alone"])
ax.set_xlabel("Year")
ax.set_xlim(pd.Timestamp("1985-06-01"), pd.Timestamp("2015-06-01"))
ax.set_ylim(-0.6, 1.6)
ax.set_title("Data-Split Scheme: SWAT Calibration/Validation vs. ML Train/Validation/Test")
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
ax.tick_params(left=False)
plt.tight_layout()
plt.savefig(f"{FIG}/fig0_data_split_scheme.png", bbox_inches="tight")
plt.close()
print("Saved: fig0_data_split_scheme.png")

# ---------------------------------------------------------------
# 2. Descriptive statistics table
# ---------------------------------------------------------------
desc = base.drop(columns=["Date"]).describe().T
desc["CV_%"] = 100 * desc["std"] / desc["mean"]
desc.to_csv(f"{TAB}/table1_descriptive_statistics.csv")
print("\nSaved: table1_descriptive_statistics.csv")

# ---------------------------------------------------------------
# 3. Time series plot: P, Q_obs, Q_SWAT
# ---------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
axes[0].bar(base["Date"], base["P"], width=25, color="#1f77b4")
axes[0].set_ylabel("Rainfall (mm/mo)")
axes[0].set_title("El Diem (Upper Blue Nile) — Monthly Hydroclimate Series, 1986–2014")

axes[1].plot(base["Date"], base["Q_obs"], color="black", lw=2.5, label="Q_obs")
axes[1].plot(base["Date"], base["Q_SWAT"], color="red", lw=2.2, alpha=0.75, label="Q_SWAT")
axes[1].axvspan(pd.Timestamp("1986-01-01"), pd.Timestamp(ML_TRAIN_END), color="#55A868", alpha=0.08, lw=0)
axes[1].axvspan(pd.Timestamp(ML_TRAIN_END), pd.Timestamp(ML_VAL_END), color="#C44E52", alpha=0.08, lw=0)
axes[1].axvspan(pd.Timestamp(ML_VAL_END), pd.Timestamp("2014-12-31"), color="#8172B2", alpha=0.10, lw=0)
# Dotted vertical lines marking the Train/Validation and Validation/Test
# boundaries, drawn on all three panels (shared x-axis) for clarity.
for ax in axes:
    ax.axvline(pd.Timestamp(ML_TRAIN_END), color="black", linestyle=":", linewidth=1.8, alpha=0.75, zorder=3)
    ax.axvline(pd.Timestamp(ML_VAL_END), color="black", linestyle=":", linewidth=1.8, alpha=0.75, zorder=3)
axes[1].set_ylabel("Discharge (m$^3$/s)")
ymax = max(base["Q_obs"].max(), base["Q_SWAT"].max()) * 1.08
axes[1].set_ylim(top=ymax)
band_label_y = ymax * 0.97
axes[1].text(pd.Timestamp("1986-01-01") + (pd.Timestamp(ML_TRAIN_END) - pd.Timestamp("1986-01-01")) / 2,
             band_label_y, "ML Train", ha="center", va="top", fontsize=12.5, color="#2E6B37", fontweight="bold")
axes[1].text(pd.Timestamp(ML_TRAIN_END) + (pd.Timestamp(ML_VAL_END) - pd.Timestamp(ML_TRAIN_END)) / 2,
             band_label_y, "ML Validation", ha="center", va="top", fontsize=12.5, color="#8C2F35", fontweight="bold")
axes[1].text(pd.Timestamp(ML_VAL_END) + (pd.Timestamp("2014-12-31") - pd.Timestamp(ML_VAL_END)) / 2,
             band_label_y, "ML Test", ha="center", va="top", fontsize=12.5, color="#4E4384", fontweight="bold")
axes[1].legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=13)

axes[2].plot(clim["Date"], clim["ENSO_t"], color="darkorange", lw=2.2, label="ENSO (Niño 3.4)")
axes[2].plot(clim["Date"], clim["IOD_t"], color="seagreen", lw=2.2, label="IOD (DMI)")
axes[2].axhline(0, color="grey", lw=1.2)
axes[2].set_ylabel("Index (°C)")
axes[2].legend(loc="upper right", frameon=False)
axes[2].set_xlabel("Year")

plt.tight_layout()
plt.savefig(f"{FIG}/fig1_timeseries_overview.png", bbox_inches="tight")
plt.close()
print("Saved: fig1_timeseries_overview.png")

# ---------------------------------------------------------------
# 4. Pearson correlation: climate teleconnections vs rainfall/discharge
#    (as specified in the procedure: ENSO vs rainfall, IOD vs discharge,
#     tested at lag 0, 1 and 3 months)
# ---------------------------------------------------------------
merged = base.merge(clim, on="Date", how="inner")
qvar = pd.read_csv(f"{DATA}/ElDiem_Qvar.csv")[["Date", "Q_t-1", "Q_t-2", "Q_t-3"]]
qvar["Date"] = pd.to_datetime(qvar["Date"], format="%Y-%m")
qvar = qvar.rename(columns={"Q_t-1": "Q_lag1", "Q_t-2": "Q_lag2", "Q_t-3": "Q_lag3"})
merged = merged.merge(qvar, on="Date", how="inner")

pairs = [
    ("ENSO_t", "P", "ENSO (lag0) vs Rainfall"),
    ("ENSO_t-1", "P", "ENSO (lag1) vs Rainfall"),
    ("ENSO_t-3", "P", "ENSO (lag3) vs Rainfall"),
    ("IOD_t", "Q_obs", "IOD (lag0) vs Discharge"),
    ("IOD_t-1", "Q_obs", "IOD (lag1) vs Discharge"),
    ("IOD_t-3", "Q_obs", "IOD (lag3) vs Discharge"),
    ("ENSO_t", "Q_obs", "ENSO (lag0) vs Discharge"),
    ("IOD_t", "P", "IOD (lag0) vs Rainfall"),
    ("NAO", "Q_obs", "NAO (lag0) vs Discharge"),
]

rows = []
for x, y, label in pairs:
    if x not in merged.columns or y not in merged.columns:
        continue
    r, p = stats.pearsonr(merged[x], merged[y])
    rows.append({"Pair": label, "Var_X": x, "Var_Y": y, "Pearson_r": round(r, 3),
                 "p_value": round(p, 4), "Significant_95%": "Yes" if p < 0.05 else "No"})

corr_table = pd.DataFrame(rows)
corr_table.to_csv(f"{TAB}/table2_pearson_teleconnections.csv", index=False)
print("\nPearson correlation (climate teleconnections):")
print(corr_table.to_string(index=False))

# ---------------------------------------------------------------
# 5. Full correlation heatmap of predictors + target, INCLUDING
#    lagged observed discharge (Q_lag1/2/3) and ENSO/IOD at lag-1
#    and lag-3 months (not just contemporaneous lag-0), so both the
#    basin-memory structure and the nonlinear-but-real teleconnection
#    lags identified later (Section 4, wavelet coherence) are visible
#    here.
# ---------------------------------------------------------------
corr_vars = ["P", "Tmax", "Tmin", "RH", "SR", "WS", "Q_obs", "Q_SWAT",
             "Q_lag1", "Q_lag2", "Q_lag3",
             "ENSO_t", "ENSO_t-1", "ENSO_t-3", "IOD_t", "IOD_t-1", "IOD_t-3", "NAO"]
corr_labels = ["P", "Tmax", "Tmin", "RH", "SR", "WS", "Q_obs", "Q_SWAT",
               "Q (lag1)", "Q (lag2)", "Q (lag3)",
               "ENSO (lag0)", "ENSO (lag1)", "ENSO (lag3)",
               "IOD (lag0)", "IOD (lag1)", "IOD (lag3)", "NAO"]
corr_matrix = merged[corr_vars].corr(method="pearson")
corr_matrix.index = corr_labels
corr_matrix.columns = corr_labels
corr_matrix.to_csv(f"{TAB}/table3_full_correlation_matrix.csv")

fig, ax = plt.subplots(figsize=(14.5, 12.5))
im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr_labels))); ax.set_xticklabels(corr_labels, rotation=90, ha="right")
ax.set_yticks(range(len(corr_labels))); ax.set_yticklabels(corr_labels)
# thicker grid lines between cells to make the matrix easier to read
ax.set_xticks(np.arange(-0.5, len(corr_labels), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(corr_labels), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=2.5)
ax.tick_params(which="minor", bottom=False, left=False)
for spine in ax.spines.values():
    spine.set_linewidth(2.0)
for i in range(len(corr_labels)):
    for j in range(len(corr_labels)):
        ax.text(j, i, f"{corr_matrix.values[i,j]:.2f}", ha="center", va="center",
                 fontsize=10.5, color="black")
cbar = plt.colorbar(im, label="Pearson r")
cbar.outline.set_linewidth(2.0)
ax.set_title("Correlation Matrix — Hydroclimate Variables & Teleconnection Indices")
plt.tight_layout()
plt.savefig(f"{FIG}/fig2_correlation_heatmap.png", bbox_inches="tight")
plt.close()
print("Saved: fig2_correlation_heatmap.png")

# ---------------------------------------------------------------
# 6. Seasonality (monthly climatology) of rainfall and discharge
# ---------------------------------------------------------------
base["month"] = base["Date"].dt.month
clim_month = base.groupby("month")[["P", "Q_obs", "Q_SWAT"]].mean()
fig, ax1 = plt.subplots(figsize=(9, 6))
ax1.bar(clim_month.index, clim_month["P"], color="#a6cee3", label="Rainfall")
ax1.set_ylabel("Mean monthly rainfall (mm)")
ax1.set_xlabel("Month")
ax2 = ax1.twinx()
ax2.plot(clim_month.index, clim_month["Q_obs"], color="black", marker="o", lw=2.8, markersize=8, label="Q_obs")
ax2.plot(clim_month.index, clim_month["Q_SWAT"], color="red", marker="s", lw=2.8, markersize=8, label="Q_SWAT")
ax2.set_ylabel("Mean monthly discharge (m$^3$/s)")
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc="upper left", frameon=False)
ax1.set_title("Seasonal Climatology — El Diem, 1986–2014")
plt.tight_layout()
plt.savefig(f"{FIG}/fig3_seasonal_climatology.png", bbox_inches="tight")
plt.close()
print("Saved: fig3_seasonal_climatology.png")

print("\n=== 01_eda_correlation.py complete ===")
