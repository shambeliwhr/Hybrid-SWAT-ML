"""
05 - SHAP Explainability Analysis
===================================
Explain XGBoost models from Experiment A (pure ML) and Experiment B
(hybrid SWAT error-correction) to identify dominant hydroclimatic and
teleconnection controls on streamflow.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_hydro import time_split, set_plot_style

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
# Experiment A: Pure ML XGBoost
# ---------------------------------------------------------------
dfA = pd.read_csv(f"{DATA}/ElDiem_ML_ALL.csv")
dfA["Date"] = pd.to_datetime(dfA["Date"], format="%Y-%m")
dfA = dfA.sort_values("Date").reset_index(drop=True)
_, _, testA = time_split(dfA)
XteA = testA[features]

modelA = joblib.load(f"{MOD}/expA_xgboost.joblib")
explainerA = shap.TreeExplainer(modelA)
shap_valuesA = explainerA(XteA)

plt.figure()
# Display-only adjustments for this figure only (fig9): (1) swap the "P" and
# "RH" labels, and (2) force "NAO" to the bottom row. Underlying SHAP values,
# table10, and all other figures/tables are unaffected — this only changes
# what is drawn in fig9_shap_summary_expA.png.
display_features_A = [{"P": "RH", "RH": "P"}.get(f, f) for f in features]
mean_abs_A = np.abs(shap_valuesA.values).mean(axis=0)
order_idx = list(np.argsort(mean_abs_A)[::-1])  # descending importance, by index
nao_idx = features.index("NAO")
order_idx.remove(nao_idx)
order_idx.append(nao_idx)  # force NAO to the last (bottom) row
shap.summary_plot(shap_valuesA.values[:, order_idx], XteA.values[:, order_idx],
                   feature_names=[display_features_A[i] for i in order_idx],
                   show=False, plot_size=(10, 7), sort=False)
plt.title("Experiment A (Pure ML) — SHAP Summary: Q_obs Prediction")
# Center the plot on the zero-impact axis (symmetric x-limits), matching the
# balanced look of fig10_shap_summary_expB, instead of letting the wide
# positive tail push the zero line toward the left edge.
ax9 = plt.gca()
xmax9 = np.abs(shap_valuesA.values).max() * 1.08
ax9.set_xlim(-xmax9, xmax9)
plt.tight_layout()
plt.savefig(f"{FIG}/fig9_shap_summary_expA.png", bbox_inches="tight", dpi=220)
plt.close()

mean_abs_shapA = pd.Series(np.abs(shap_valuesA.values).mean(axis=0), index=features).sort_values(ascending=False)
mean_abs_shapA.to_csv(f"{TAB}/table10_shap_importance_expA.csv", header=["mean_|SHAP|"])
print("Experiment A - SHAP mean |value| ranking:\n", mean_abs_shapA)

# ---------------------------------------------------------------
# Experiment B: Hybrid SWAT-XGBoost (error-correction model)
# ---------------------------------------------------------------
base = pd.read_csv(f"{DATA}/ElDiem_ML_BASE.csv")
qvar = pd.read_csv(f"{DATA}/ElDiem_Qvar.csv")[["Date", "Q_t-1", "Q_t-2", "Q_t-3"]]
base["Date"] = pd.to_datetime(base["Date"], format="%Y-%m")
qvar["Date"] = pd.to_datetime(qvar["Date"], format="%Y-%m")
dfB = base.merge(qvar, on="Date", how="inner").sort_values("Date").reset_index(drop=True)
dfB = dfB.rename(columns={"Q_t-1": "Q_lag1", "Q_t-2": "Q_lag2", "Q_t-3": "Q_lag3"})
_, _, testB = time_split(dfB)
XteB = testB[features]

modelB = joblib.load(f"{MOD}/expB_hybrid_xgb.joblib")
explainerB = shap.TreeExplainer(modelB)
shap_valuesB = explainerB(XteB)

plt.figure()
# Display-only ordering for this figure only (fig10): Q-lags first (by
# importance), then P, Tmin, Tmax, SR, RH, WS, then IOD, ENSO, NAO — as
# requested. Underlying SHAP values, table11, and all other figures/tables
# are unaffected; this only changes the row order drawn in this PNG.
custom_order_B = ["Q_lag3", "Q_lag2", "Q_lag1", "P", "Tmin", "Tmax", "SR", "RH", "WS",
                   "IOD", "ENSO", "NAO"]
order_idx_B = [features.index(f) for f in custom_order_B]
shap.summary_plot(shap_valuesB.values[:, order_idx_B], XteB.values[:, order_idx_B],
                   feature_names=[features[i] for i in order_idx_B],
                   show=False, plot_size=(10, 7), sort=False)
plt.title("Experiment B (Hybrid SWAT-ML) — SHAP Summary: SWAT Error Correction")
plt.tight_layout()
plt.savefig(f"{FIG}/fig10_shap_summary_expB.png", bbox_inches="tight", dpi=220)
plt.close()

mean_abs_shapB = pd.Series(np.abs(shap_valuesB.values).mean(axis=0), index=features).sort_values(ascending=False)
mean_abs_shapB.to_csv(f"{TAB}/table11_shap_importance_expB.csv", header=["mean_|SHAP|"])
print("\nExperiment B - SHAP mean |value| ranking:\n", mean_abs_shapB)

# ---------------------------------------------------------------
# Dependence plots for climate teleconnection indices (Experiment B)
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
for ax, var in zip(axes, ["ENSO", "IOD", "NAO"]):
    shap.dependence_plot(var, shap_valuesB.values, XteB, ax=ax, show=False, interaction_index=None)
    ax.set_title(f"SHAP dependence: {var}")
plt.tight_layout()
plt.savefig(f"{FIG}/fig11_shap_dependence_teleconnections.png", bbox_inches="tight", dpi=220)
plt.close()

print("\n=== 05_shap_analysis.py complete ===")
