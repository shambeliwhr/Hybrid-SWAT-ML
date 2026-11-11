"""
09 - Residual Diagnostics: ACF, PACF, and Q-Q Plots (All Models)
====================================================================
For each of the 5 models (SWAT, Random Forest, XGBoost, Hybrid SWAT-RF,
Hybrid SWAT-XGBoost), residuals (Q_obs - Q_predicted) are computed over
the same Train/Validation/Test windows used throughout the study, then:

  1. Autocorrelation Function (ACF)
  2. Partial Autocorrelation Function (PACF)
  3. Q-Q plot vs. the Normal distribution

...are plotted for each model, with Train+Validation residuals shown in
one column and Test residuals in a second column (2-panel comparison per
model, stacked into one grid figure per diagnostic). Train and Validation
are concatenated in chronological order (1986-2000 then 2001-2009) into a
single continuous series before computing ACF/PACF, since they are
contiguous in time.

A Ljung-Box test (lag=12, i.e. one annual cycle) is also tabulated for
each model/period as a formal check for residual autocorrelation.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import joblib
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_hydro import set_plot_style, time_split, ML_VAL_END

set_plot_style()
DATA = "D:/Flow/Discharge_lag"
FIG = "D:/Flow/Discharge_lag/figs"
TAB = "D:/Flow/Discharge_lag/tables"
MOD = "D:/Flow/Discharge_lag/models"
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

features = ["P", "Tmax", "Tmin", "RH", "SR", "WS",
            "Q_lag1", "Q_lag2", "Q_lag3", "ENSO", "IOD", "NAO"]

# ---------------------------------------------------------------
# 1. Rebuild the exact feature sets used in Experiments A and B
# ---------------------------------------------------------------
dfA = pd.read_csv(f"{DATA}/ElDiem_ML_ALL.csv")
dfA["Date"] = pd.to_datetime(dfA["Date"], format="%Y-%m")
dfA = dfA.sort_values("Date").reset_index(drop=True)

base = pd.read_csv(f"{DATA}/ElDiem_ML_BASE.csv")
qvar = pd.read_csv(f"{DATA}/ElDiem_Qvar.csv")[["Date", "Q_t-1", "Q_t-2", "Q_t-3"]]
base["Date"] = pd.to_datetime(base["Date"], format="%Y-%m")
qvar["Date"] = pd.to_datetime(qvar["Date"], format="%Y-%m")
dfB = base.merge(qvar, on="Date", how="inner").sort_values("Date").reset_index(drop=True)
dfB = dfB.rename(columns={"Q_t-1": "Q_lag1", "Q_t-2": "Q_lag2", "Q_t-3": "Q_lag3"})

trainA, valA, testA = time_split(dfA)
trainB, valB, testB = time_split(dfB)

# ---------------------------------------------------------------
# 2. Load trained models
# ---------------------------------------------------------------
rf_pure = joblib.load(f"{MOD}/expA_random_forest.joblib")
xgb_pure = joblib.load(f"{MOD}/expA_xgboost.joblib")
rf_hyb = joblib.load(f"{MOD}/expB_hybrid_rf.joblib")
xgb_hyb = joblib.load(f"{MOD}/expB_hybrid_xgb.joblib")


def residuals_for(period_df_A, period_df_B):
    """Return a dict of {model_name: residual_array} for one period."""
    res = {}
    res["SWAT"] = (period_df_B["Q_obs"] - period_df_B["Q_SWAT"]).values
    res["Random Forest"] = (period_df_A["Q_target"] - rf_pure.predict(period_df_A[features])).values
    res["XGBoost"] = (period_df_A["Q_target"] - xgb_pure.predict(period_df_A[features])).values
    res["Hybrid RF"] = (period_df_B["Q_obs"] - (period_df_B["Q_SWAT"] + rf_hyb.predict(period_df_B[features]))).values
    res["Hybrid XGBoost"] = (period_df_B["Q_obs"] - (period_df_B["Q_SWAT"] + xgb_hyb.predict(period_df_B[features]))).values
    return res


res_train = residuals_for(trainA, trainB)
res_val = residuals_for(valA, valB)
res_test = residuals_for(testA, testB)

# Train+Validation concatenated in chronological order (contiguous in time)
res_trainval = {m: np.concatenate([res_train[m], res_val[m]]) for m in res_train}

models = ["SWAT", "Random Forest", "XGBoost", "Hybrid RF", "Hybrid XGBoost"]
model_colors = {"SWAT": "#666666", "Random Forest": "#1f77b4", "XGBoost": "#d62728",
                "Hybrid RF": "#9467bd", "Hybrid XGBoost": "#2ca02c"}

periods = {f"Train+Validation (1986-{pd.Timestamp(ML_VAL_END).year})": res_trainval,
           f"Test ({pd.Timestamp(ML_VAL_END).year + 1}-2014)": res_test}

# ---------------------------------------------------------------
# 3. Ljung-Box test table (lag=12) — formal residual-autocorrelation check
# ---------------------------------------------------------------
lb_rows = []
for period_name, res_dict in periods.items():
    for m in models:
        lb = acorr_ljungbox(res_dict[m], lags=[12], return_df=True)
        lb_rows.append({"Period": period_name, "Model": m,
                         "LjungBox_stat(lag12)": round(lb["lb_stat"].iloc[0], 2),
                         "p_value": round(lb["lb_pvalue"].iloc[0], 4),
                         "Residual_autocorrelation": "Yes (p<0.05)" if lb["lb_pvalue"].iloc[0] < 0.05 else "No"})
lb_table = pd.DataFrame(lb_rows)
lb_table.to_csv(f"{TAB}/table19_ljungbox_residual_autocorrelation.csv", index=False)
print(lb_table.to_string(index=False))

# ---------------------------------------------------------------
# 4. ACF grid (rows = models, cols = Train+Val / Test)
# ---------------------------------------------------------------
nlags = 24
fig, axes = plt.subplots(len(models), 2, figsize=(13, 3.1 * len(models)))
for i, m in enumerate(models):
    for j, (period_name, res_dict) in enumerate(periods.items()):
        ax = axes[i, j]
        r = res_dict[m]
        acf_vals, confint = acf(r, nlags=nlags, alpha=0.05, fft=True)
        lags = np.arange(len(acf_vals))
        ax.vlines(lags, 0, acf_vals, color=model_colors[m], linewidth=2.2)
        ax.axhline(0, color="black", linewidth=1.0)
        ci_lo = confint[:, 0] - acf_vals
        ci_hi = confint[:, 1] - acf_vals
        ax.fill_between(lags, ci_lo, ci_hi, color="grey", alpha=0.25, linewidth=0)
        ax.set_ylim(-1.05, 1.05)
        if i == 0:
            ax.set_title(period_name, fontsize=16)
        if j == 0:
            ax.set_ylabel(f"{m}\nACF", fontsize=13)
        if i == len(models) - 1:
            ax.set_xlabel("Lag (months)")
fig.suptitle("Autocorrelation Function (ACF) of Residuals — All Models", fontsize=19, y=1.005)
plt.tight_layout()
plt.savefig(f"{FIG}/fig18_ACF_residuals_all_models.png", bbox_inches="tight")
plt.close()
print("Saved: fig18_ACF_residuals_all_models.png")

# ---------------------------------------------------------------
# 5. PACF grid (rows = models, cols = Train+Val / Test)
# ---------------------------------------------------------------
fig, axes = plt.subplots(len(models), 2, figsize=(13, 3.1 * len(models)))
for i, m in enumerate(models):
    for j, (period_name, res_dict) in enumerate(periods.items()):
        ax = axes[i, j]
        r = res_dict[m]
        pacf_vals, confint = pacf(r, nlags=nlags, alpha=0.05, method="ywm")
        lags = np.arange(len(pacf_vals))
        ax.vlines(lags, 0, pacf_vals, color=model_colors[m], linewidth=2.2)
        ax.axhline(0, color="black", linewidth=1.0)
        ci_lo = confint[:, 0] - pacf_vals
        ci_hi = confint[:, 1] - pacf_vals
        ax.fill_between(lags, ci_lo, ci_hi, color="grey", alpha=0.25, linewidth=0)
        ax.set_ylim(-1.05, 1.05)
        if i == 0:
            ax.set_title(period_name, fontsize=16)
        if j == 0:
            ax.set_ylabel(f"{m}\nPACF", fontsize=13)
        if i == len(models) - 1:
            ax.set_xlabel("Lag (months)")
fig.suptitle("Partial Autocorrelation Function (PACF) of Residuals — All Models", fontsize=19, y=1.005)
plt.tight_layout()
plt.savefig(f"{FIG}/fig19_PACF_residuals_all_models.png", bbox_inches="tight")
plt.close()
print("Saved: fig19_PACF_residuals_all_models.png")

# ---------------------------------------------------------------
# 6. Q-Q plot grid (rows = models, cols = Train+Val / Test)
# ---------------------------------------------------------------
fig, axes = plt.subplots(len(models), 2, figsize=(11, 3.1 * len(models)))
for i, m in enumerate(models):
    for j, (period_name, res_dict) in enumerate(periods.items()):
        ax = axes[i, j]
        r = res_dict[m]
        (osm, osr), (slope, intercept, r_val) = stats.probplot(r, dist="norm")
        ax.scatter(osm, osr, s=16, color=model_colors[m], alpha=0.7, edgecolors="none")
        ax.plot(osm, slope * osm + intercept, color="black", linewidth=1.6, linestyle="--")
        ax.text(0.05, 0.92, f"R² = {r_val**2:.3f}", transform=ax.transAxes, fontsize=11.5, va="top")
        if i == 0:
            ax.set_title(period_name, fontsize=16)
        if j == 0:
            ax.set_ylabel(f"{m}\nSample quantiles", fontsize=12.5)
        if i == len(models) - 1:
            ax.set_xlabel("Theoretical quantiles (Normal)")
fig.suptitle("Q-Q Plot of Residuals vs. Normal Distribution — All Models", fontsize=19, y=1.005)
plt.tight_layout()
plt.savefig(f"{FIG}/fig20_QQplot_residuals_all_models.png", bbox_inches="tight")
plt.close()
print("Saved: fig20_QQplot_residuals_all_models.png")

# ---------------------------------------------------------------
# 7. Residual time-series (Observed - Simulated), all models overlaid,
#    styled like the reference figure: dashed zero-residual line plus
#    one colored line per model. Train+Validation in one plot, Test in
#    a separate plot.
# ---------------------------------------------------------------
dates_trainval = pd.concat([trainB["Date"], valB["Date"]]).reset_index(drop=True)
dates_test = testB["Date"].reset_index(drop=True)
period_dates = {list(periods.keys())[0]: dates_trainval, list(periods.keys())[1]: dates_test}
fname_map = {list(periods.keys())[0]: "fig21_residuals_timeseries_trainval_all_models.png",
             list(periods.keys())[1]: "fig22_residuals_timeseries_test_all_models.png"}

for period_name, res_dict in periods.items():
    dates = period_dates[period_name]
    fig, ax = plt.subplots(figsize=(15, 6.5))
    ax.axhline(0, color="black", linestyle="--", linewidth=2.2, label="zero-residual", zorder=5)
    for m in models:
        ax.plot(dates, res_dict[m], color=model_colors[m], linewidth=1.8, label=m, alpha=0.9)
    ax.set_ylabel("Residual (monthly streamflow, m$^3$/s)")
    ax.set_xlabel("Date")
    ax.set_title(f"Residuals (Observed − Simulated) — All Models ({period_name})", pad=55, fontsize=19)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False, fontsize=14)
    fig.subplots_adjust(top=0.82)
    plt.savefig(f"{FIG}/{fname_map[period_name]}", bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname_map[period_name]}")

print("\n=== 09_residual_diagnostics.py complete ===")
