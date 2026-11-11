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
# 4. ACF -- single overlay plot per period (all 5 models on one axes,
#    line+marker style), matching the reference figure convention.
# ---------------------------------------------------------------
nlags = 24
period_tags = {list(periods.keys())[0]: ("trainval", "Train+Validation"),
               list(periods.keys())[1]: ("test", "Test")}

for period_name, res_dict in periods.items():
    tag, label = period_tags[period_name]
    fig, ax = plt.subplots(figsize=(11, 7))
    ci_band = None
    for m in models:
        r = res_dict[m]
        acf_vals, confint = acf(r, nlags=nlags, alpha=0.05, fft=True)
        lags = np.arange(len(acf_vals))
        ax.plot(lags, acf_vals, color=model_colors[m], marker="o", markersize=5,
                 linewidth=1.8, label=m)
        if ci_band is None:
            ci_band = (confint[:, 0] - acf_vals, confint[:, 1] - acf_vals)
    ax.axhline(0, color="black", linewidth=1.4)
    ax.axhline(ci_band[0][2], color="grey", linewidth=1.4, linestyle="--")
    ax.axhline(ci_band[1][2], color="grey", linewidth=1.4, linestyle="--")
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Lag (months)")
    ax.set_ylabel("ACF")
    ax.set_title(f"Autocorrelation (ACF) of Residuals — All Models ({period_name})")
    ax.legend(loc="upper right", ncol=2, frameon=True, framealpha=0.9, fontsize=13)
    plt.tight_layout()
    fname = f"fig23_ACF_residuals_{tag}_overlay.png" if tag == "trainval" else "fig24_ACF_residuals_test_overlay.png"
    plt.savefig(f"{FIG}/{fname}", bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")

# ---------------------------------------------------------------
# 5. PACF -- single overlay plot per period
# ---------------------------------------------------------------
for period_name, res_dict in periods.items():
    tag, label = period_tags[period_name]
    fig, ax = plt.subplots(figsize=(11, 7))
    ci_band = None
    for m in models:
        r = res_dict[m]
        pacf_vals, confint = pacf(r, nlags=nlags, alpha=0.05, method="ywm")
        lags = np.arange(len(pacf_vals))
        ax.plot(lags, pacf_vals, color=model_colors[m], marker="o", markersize=5,
                 linewidth=1.8, label=m)
        if ci_band is None:
            ci_band = (confint[:, 0] - pacf_vals, confint[:, 1] - pacf_vals)
    ax.axhline(0, color="black", linewidth=1.4)
    ax.axhline(ci_band[0][2], color="grey", linewidth=1.4, linestyle="--")
    ax.axhline(ci_band[1][2], color="grey", linewidth=1.4, linestyle="--")
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Lag (months)")
    ax.set_ylabel("PACF")
    ax.set_title(f"Partial Autocorrelation (PACF) of Residuals — All Models ({period_name})")
    ax.legend(loc="upper right", ncol=2, frameon=True, framealpha=0.9, fontsize=13)
    plt.tight_layout()
    fname = "fig25_PACF_residuals_trainval_overlay.png" if tag == "trainval" else "fig26_PACF_residuals_test_overlay.png"
    plt.savefig(f"{FIG}/{fname}", bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")

# ---------------------------------------------------------------
# 6. Q-Q plot -- single overlay plot per period (all 5 models on one
#    axes as colored scatter, one shared theoretical reference line)
# ---------------------------------------------------------------
for period_name, res_dict in periods.items():
    tag, label = period_tags[period_name]
    fig, ax = plt.subplots(figsize=(9.5, 9.5))
    all_osm = []
    for m in models:
        r = res_dict[m]
        (osm, osr), (slope, intercept, r_val) = stats.probplot(r, dist="norm")
        ax.scatter(osm, osr, s=28, color=model_colors[m], alpha=0.85, edgecolors="none", label=m)
        all_osm.append(osm)
    # single shared theoretical (45-degree-style) reference line spanning all models' range
    all_osm_flat = np.concatenate(all_osm)
    all_osr_flat = np.concatenate([res_dict[m] for m in models])
    lo, hi = np.percentile(all_osm_flat, [0.5, 99.5])
    slope_ref, intercept_ref = np.polyfit(np.sort(all_osm_flat), np.sort(all_osr_flat), 1)
    xline = np.array([all_osm_flat.min(), all_osm_flat.max()])
    ax.plot(xline, slope_ref * xline + intercept_ref, color="black", linewidth=2.2,
             linestyle="--", label="Theoretical", zorder=0)
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Sample quantiles")
    ax.set_title(f"Q-Q Plot of Residuals — All Models ({period_name})")
    handles, labels_ = ax.get_legend_handles_labels()
    order = [labels_.index("Theoretical")] + [i for i, l in enumerate(labels_) if l != "Theoretical"]
    ax.legend([handles[i] for i in order], [labels_[i] for i in order],
               loc="upper left", ncol=2, frameon=True, framealpha=0.9, fontsize=13)
    plt.tight_layout()
    fname = "fig27_QQplot_residuals_trainval_overlay.png" if tag == "trainval" else "fig28_QQplot_residuals_test_overlay.png"
    plt.savefig(f"{FIG}/{fname}", bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")

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
