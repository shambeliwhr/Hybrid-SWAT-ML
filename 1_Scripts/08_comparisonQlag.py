"""
08 - Final All-Model Comparison
==================================
Combines Experiment A (pure ML) and Experiment B (hybrid SWAT-ML) test-period
results with the raw SWAT baseline into one ranked comparison table and one
overlay hydrograph figure.
"""
import pandas as pd
import matplotlib.pyplot as plt
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_hydro import set_plot_style

set_plot_style()
FIG = "D:/Flow/Discharge_lag/figs"
TAB = "D:/Flow/Discharge_lag/tables"
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

# ---------------------------------------------------------------
# Combined metrics table (ranked by NSE)
# ---------------------------------------------------------------
a = pd.read_csv(f"{TAB}/table5_expA_pureML_test_metrics.csv", index_col=0)
b = pd.read_csv(f"{TAB}/table7_expB_hybrid_test_metrics.csv", index_col=0)
combo = pd.concat([a, b.drop(index=[i for i in b.index if "SWAT" in i])])
combo = combo.sort_values("NSE", ascending=False)
combo.to_csv(f"{TAB}/table16_ALL_MODELS_final_comparison.csv")
print(combo)

# ---------------------------------------------------------------
# Combined hydrograph — all models, test period
# ---------------------------------------------------------------
predA = pd.read_csv(f"{TAB}/expA_test_predictions.csv", parse_dates=["Date"])
predB = pd.read_csv(f"{TAB}/expB_test_predictions.csv", parse_dates=["Date"])

fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(predA["Date"], predA["Q_obs"], color="black", lw=3.2, label="Q_obs")
ax.plot(predA["Date"], predA["Q_SWAT"], color="grey", lw=2.2, ls="--", label="Q_SWAT")
ax.plot(predA["Date"], predA["RF_pred"], color="#1f77b4", lw=2.4, label="Pure ML - Random Forest")
ax.plot(predA["Date"], predA["XGB_pred"], color="#2ca02c", lw=2.4, label="Pure ML - XGBoost")
ax.plot(predB["Date"], predB["Hybrid_RF"], color="#9467bd", lw=2.4, label="Hybrid SWAT-RF")
ax.plot(predB["Date"], predB["Hybrid_XGB"], color="#d62728", lw=2.6, label="Hybrid SWAT-XGBoost")
ax.set_ylabel("Discharge (m$^3$/s)")
ax.set_xlabel("Year")
ax.set_title("All Models — Test Period (2006–2014), El Diem Station")
ax.legend(frameon=False, ncol=3, loc="upper right")
plt.tight_layout()
plt.savefig(f"{FIG}/fig16_ALL_models_comparison_hydrograph.png", bbox_inches="tight")
plt.close()

print("Saved: fig16_ALL_models_comparison_hydrograph.png")
print("\n=== 08_final_comparison.py complete ===")
