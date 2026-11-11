"""
07 - Standardized Precipitation Index (SPI) & Extreme-Event Analysis
========================================================================
SPI computed at 12-month timescale via gamma-distribution fitting
(McKee et al., 1993). Cross-tabulated against ENSO phase and discharge
deficits to link teleconnections with hydrological extremes.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_hydro import set_plot_style

set_plot_style()
DATA = "D:/Flow/Discharge_lag"
FIG = "D:/Flow/Discharge_lag/figs"
TAB = "D:/Flow/Discharge_lag/tables"
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

df = pd.read_csv(f"{DATA}/ElDiem_ML_BASE.csv")
df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m")
df = df.sort_values("Date").reset_index(drop=True)


def compute_spi(precip, scale=12):
    """SPI via rolling accumulation + gamma fit per calendar month."""
    s = pd.Series(precip.values, index=pd.to_datetime(df["Date"]))
    accum = s.rolling(scale).sum()
    spi = pd.Series(index=accum.index, dtype=float)
    months = accum.index.month
    for m in range(1, 13):
        vals = accum[months == m].dropna()
        if len(vals) < 5:
            continue
        # handle zeros with a mixed distribution
        zero_frac = (vals == 0).mean()
        nz = vals[vals > 0]
        if len(nz) >= 5:
            a, loc, scale_ = stats.gamma.fit(nz, floc=0)
            cdf = zero_frac + (1 - zero_frac) * stats.gamma.cdf(vals, a, loc=loc, scale=scale_)
        else:
            cdf = stats.rankdata(vals) / (len(vals) + 1)
        spi.loc[vals.index] = stats.norm.ppf(np.clip(cdf, 1e-6, 1 - 1e-6))
    return spi


spi12 = compute_spi(df["P"], scale=12)
df["SPI12"] = spi12.values


def spi_class(v):
    if pd.isna(v): return np.nan
    if v <= -2: return "Extreme drought"
    if v <= -1.5: return "Severe drought"
    if v <= -1: return "Moderate drought"
    if v < 1: return "Near normal"
    if v < 1.5: return "Moderately wet"
    if v < 2: return "Very wet"
    return "Extremely wet"


df["SPI_class"] = df["SPI12"].apply(spi_class)

# ENSO phase classification (ONI-like: >=0.5 El Nino, <=-0.5 La Nina)
def enso_phase(v):
    if v >= 0.5: return "El Nino"
    if v <= -0.5: return "La Nina"
    return "Neutral"


df["ENSO_phase"] = df["ENSO"].apply(enso_phase)

# discharge anomaly (standardized)
df["Q_anom"] = (df["Q_obs"] - df["Q_obs"].mean()) / df["Q_obs"].std()

# Cross-tab: mean discharge anomaly & drought frequency by ENSO phase
xtab = df.groupby("ENSO_phase").agg(
    mean_SPI12=("SPI12", "mean"),
    mean_Q_anom=("Q_anom", "mean"),
    pct_drought_months=("SPI12", lambda x: 100 * (x <= -1).mean()),
    n_months=("SPI12", "count"),
).round(2)
xtab.to_csv(f"{TAB}/table13_ENSO_phase_drought_discharge.csv")
print(xtab)

# --- Plot: SPI12 timeseries with ENSO shading + Q anomaly ---
fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True)
axes[0].bar(df["Date"], df["SPI12"], width=25,
            color=np.where(df["SPI12"] < 0, "firebrick", "steelblue"))
axes[0].axhline(-1, color="k", lw=0.6, ls="--")
axes[0].axhline(1, color="k", lw=0.6, ls="--")
axes[0].set_ylabel("SPI-12")
axes[0].set_title("12-Month Standardized Precipitation Index (SPI-12) — El Diem")

axes[1].plot(df["Date"], df["Q_anom"], color="black", lw=1)
axes[1].fill_between(df["Date"], 0, df["Q_anom"],
                      where=df["Q_anom"] < 0, color="firebrick", alpha=0.4, label="Deficit")
axes[1].fill_between(df["Date"], 0, df["Q_anom"],
                      where=df["Q_anom"] >= 0, color="steelblue", alpha=0.4, label="Surplus")
axes[1].set_ylabel("Standardized Q anomaly")
axes[1].set_xlabel("Year")
axes[1].legend(frameon=False, loc="upper right")
plt.tight_layout()
plt.savefig(f"{FIG}/fig15_SPI_and_discharge_anomaly.png", bbox_inches="tight")
plt.close()

# Identify top drought & flood years for discussion
worst = df.nsmallest(5, "SPI12")[["Date", "SPI12", "ENSO", "IOD", "Q_obs"]]
wettest = df.nlargest(5, "SPI12")[["Date", "SPI12", "ENSO", "IOD", "Q_obs"]]
worst.to_csv(f"{TAB}/table14_top5_drought_months.csv", index=False)
wettest.to_csv(f"{TAB}/table15_top5_wet_months.csv", index=False)
print("\nTop 5 drought months:\n", worst)
print("\nTop 5 wet months:\n", wettest)

print("\n=== 07_spi_drought_analysis.py complete ===")
