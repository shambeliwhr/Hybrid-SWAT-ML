# -*- coding: utf-8 -*-
"""
05_summary.py
=============
Pulls together the outputs of scripts 01-04 into one publication-ready
summary table and a combined "graphical abstract" style figure.

Outputs
-------
- tables/05_FINAL_SUMMARY_TABLE.csv   (SWAT + M1-M5, calibration/train/val/test)
- figures/05_graphical_abstract.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import FIG_DIR, TAB_DIR, set_publication_style, COLORS, MODEL_LABELS
from data_prep import build_master, split_by_date
from config import (SWAT_CAL_START, SWAT_CAL_END, SWAT_VAL_START, SWAT_VAL_END,
                     ML_TEST_START, ML_TEST_END)
from metrics_utils import all_metrics

set_publication_style()

swat_perf = pd.read_csv(os.path.join(TAB_DIR, "01_SWAT_performance.csv"))
ml_perf = pd.read_csv(os.path.join(TAB_DIR, "02_model_performance.csv"))

swat_perf.insert(0, "Model", "SWAT")
final = pd.concat([
    swat_perf.rename(columns={"Label": "Split"}),
    ml_perf
], ignore_index=True)
final.to_csv(os.path.join(TAB_DIR, "05_FINAL_SUMMARY_TABLE.csv"), index=False)
print("=== FINAL SUMMARY TABLE ===")
print(final.to_string(index=False))

# ---------------------------------------------------------------------
# Graphical abstract: 4-panel figure summarizing the whole pipeline
# ---------------------------------------------------------------------
master = build_master()
pred = pd.read_csv(os.path.join(TAB_DIR, "master_dataset_with_predictions.csv"), parse_dates=["Date"])
test = split_by_date(pred, ML_TEST_START, ML_TEST_END)

fig = plt.figure(figsize=(16, 11), constrained_layout=True)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)

# Panel A: SWAT full period hydrograph (small)
axA = fig.add_subplot(gs[0, 0])
axA.plot(master["Date"], master["Q_obs"], color=COLORS["Q_obs"], lw=2.5, label="Observed")
axA.plot(master["Date"], master["Q_SWAT"], color=COLORS["Q_SWAT"], lw=2.0, ls="--", label="SWAT")
axA.set_title("(a) SWAT Stand-Alone (1986\u20132014)")
axA.set_ylabel(r"Q (m$^3$/s)")
axA.legend(fontsize=14, frameon=True)
for s in axA.spines.values():
    s.set_linewidth(2.5)

# Panel B: Test-period best pure-ML (M3) vs Hybrid (M5) vs Observed
axB = fig.add_subplot(gs[0, 1])
axB.plot(test["Date"], test["Q_obs"], color=COLORS["Q_obs"], lw=2.5, label="Observed")
axB.plot(test["Date"], test["Q_M3"], color=COLORS["M3"], lw=2.0, ls="--", label="M3 (Pure ML)")
axB.plot(test["Date"], test["Q_M5"], color=COLORS["M5"], lw=2.0, ls="--", label="M5 (Hybrid)")
axB.set_title("(b) Test Period (2005\u20132014)")
axB.legend(fontsize=14, frameon=True)
for s in axB.spines.values():
    s.set_linewidth(2.5)

# Panel C: NSE bar comparison
axC = fig.add_subplot(gs[1, 0])
test_perf = final[final["Split"] == "Test"].set_index("Model")
labels_order = ["SWAT", "M1", "M2", "M3", "M4", "M5"]
swat_test_metrics = all_metrics(split_by_date(master, ML_TEST_START, ML_TEST_END)["Q_obs"],
                                  split_by_date(master, ML_TEST_START, ML_TEST_END)["Q_SWAT"])
nse_vals = [swat_test_metrics["NSE"]] + [test_perf.loc[m, "NSE"] for m in ["M1", "M2", "M3", "M4", "M5"]]
bar_colors = [COLORS["Q_SWAT"]] + [COLORS[m] for m in ["M1", "M2", "M3", "M4", "M5"]]
axC.bar(labels_order, nse_vals, color=bar_colors, edgecolor="black", linewidth=2.5)
axC.axhline(0.7, color="gray", ls=":", lw=2.0)
axC.set_ylabel("NSE (Test period)")
axC.set_title("(c) Test-Period NSE by Model")
for s in axC.spines.values():
    s.set_linewidth(2.5)

# Panel D: SHAP importance (M3 vs M5), top 6 features
axD = fig.add_subplot(gs[1, 1])
shap_m3 = pd.read_csv(os.path.join(TAB_DIR, "03_SHAP_importance_M3.csv")).head(6)
shap_m5 = pd.read_csv(os.path.join(TAB_DIR, "03_SHAP_importance_M5.csv")).head(6)
y = np.arange(6)
axD.barh(y + 0.2, shap_m3["Importance_%"], height=0.35, color=COLORS["M3"],
          edgecolor="black", linewidth=2.0, label="M3 (Pure ML)")
axD.barh(y - 0.2, shap_m5["Importance_%"], height=0.35, color=COLORS["M5"],
          edgecolor="black", linewidth=2.0, label="M5 (Hybrid)")
for yi, (fm3, im3) in enumerate(zip(shap_m3["Feature"], shap_m3["Importance_%"])):
    axD.text(im3 + 0.6, yi + 0.2, fm3, va="center", fontsize=12, color=COLORS["M3"])
for yi, (fm5, im5) in enumerate(zip(shap_m5["Feature"], shap_m5["Importance_%"])):
    axD.text(im5 + 0.6, yi - 0.2, fm5, va="center", fontsize=12, color=COLORS["M5"])
axD.set_yticks(y)
axD.set_yticklabels([f"Rank {i+1}" for i in range(6)], fontsize=14)
axD.set_xlim(0, 66)
axD.set_xlabel("SHAP importance (%)")
axD.set_title("(d) Top-6 Drivers: M3 vs M5 (by rank)")
axD.legend(fontsize=13, frameon=True, loc="upper right", bbox_to_anchor=(1.0, 0.85))
for s in axD.spines.values():
    s.set_linewidth(2.5)

fig.suptitle("Hybrid SWAT-ML-SHAP Streamflow Framework \u2013 El Diem, Upper Blue Nile Basin",
             fontsize=22)
fig.savefig(os.path.join(FIG_DIR, "05_graphical_abstract.png"), bbox_inches="tight")
plt.close(fig)

print("\nSummary complete. See tables/05_FINAL_SUMMARY_TABLE.csv and figures/05_graphical_abstract.png")
