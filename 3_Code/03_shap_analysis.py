# -*- coding: utf-8 -*-
"""
03_shap_analysis.py
====================
SHAP (SHapley Additive exPlanations) analysis for the two headline
models: M3 (full pure-ML) and M5 (full hybrid SWAT-ML). Uses TreeSHAP
on the XGBoost models saved by 02_ml_models.py, evaluated on the full
dataset (1986-2014) for a stable, publication-ready attribution.

Outputs
-------
- tables/03_SHAP_importance_M3.csv / _M5.csv
- figures/03_SHAP_summary_bar_M3.png / _M5.png     (mean |SHAP|)
- figures/03_SHAP_beeswarm_M3.png / _M5.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from xgboost import XGBRegressor

from config import (FIG_DIR, TAB_DIR, MODEL_DIR, set_publication_style, COLORS,
                     MET_VARS, LAG_VARS, CLIM_VARS)
from data_prep import build_master

font_family = set_publication_style()
shap.plots._utils.plt = plt  # ensure shap uses the configured rcParams

master = build_master()

FEATURES = {
    "M3": MET_VARS + LAG_VARS + CLIM_VARS,
    "M5": MET_VARS + LAG_VARS + CLIM_VARS,
}

PRETTY = {
    "P": r"Rainfall $P_t$", "Tmax": "Tmax", "Tmin": "Tmin", "RH": "Rel. Humidity",
    "SR": "Solar Radiation", "WS": "Wind Speed", "P_lag1": r"Rainfall $P_{t-1}$ (antecedent)",
    "Q_lag1": r"$Q_{t-1}$", "Q_lag2": r"$Q_{t-2}$", "Q_lag3": r"$Q_{t-3}$",
    "ENSO": "ENSO (Ni\u00f1o 3.4)", "IOD": "IOD (DMI)", "NAO": "NAO",
}

summary_all = []

for name in ["M3", "M5"]:
    feats = FEATURES[name]
    model = XGBRegressor()
    model.load_model(os.path.join(MODEL_DIR, f"{name}.json"))

    X = master[feats].rename(columns=PRETTY)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    mean_abs = np.abs(shap_values.values).mean(axis=0)
    imp = pd.DataFrame({"Feature": X.columns, "Mean_abs_SHAP": mean_abs})
    imp["Importance_%"] = 100 * imp["Mean_abs_SHAP"] / imp["Mean_abs_SHAP"].sum()
    imp = imp.sort_values("Importance_%", ascending=False).reset_index(drop=True)
    imp.to_csv(os.path.join(TAB_DIR, f"03_SHAP_importance_{name}.csv"), index=False)
    imp["Model"] = name
    summary_all.append(imp)

    print(f"\n=== SHAP feature importance ({name}) ===")
    print(imp[["Feature", "Importance_%"]].to_string(index=False))

    # --- Bar plot (mean |SHAP|) ---
    fig, ax = plt.subplots(figsize=(9, 7))
    order = imp.sort_values("Importance_%")
    ax.barh(order["Feature"], order["Importance_%"], color=COLORS[name],
            edgecolor="black", linewidth=2.5)
    ax.set_xlabel("Mean |SHAP value| contribution (%)")
    target_lbl = "Q_obs" if name == "M3" else "SWAT error correction"
    ax.set_title(f"{name}: SHAP Feature Importance ({target_lbl})")
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"03_SHAP_summary_bar_{name}.png"))
    plt.close(fig)

    # --- Beeswarm plot ---
    fig = plt.figure(figsize=(10, 7))
    shap.plots.beeswarm(shap_values, show=False, max_display=len(feats))
    fig = plt.gcf()
    fig.set_size_inches(10, 7)
    for ax in fig.axes:
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)
    plt.title(f"{name}: SHAP Beeswarm ({target_lbl})", fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"03_SHAP_beeswarm_{name}.png"), dpi=600, bbox_inches="tight")
    plt.close(fig)

pd.concat(summary_all).to_csv(os.path.join(TAB_DIR, "03_SHAP_importance_combined.csv"), index=False)
print("\nSHAP analysis complete. Tables and figures written to OUTPUT_DIR.")
