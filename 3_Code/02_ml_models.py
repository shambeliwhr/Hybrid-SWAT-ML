# -*- coding: utf-8 -*-
"""
02_ml_models.py
================
Progressive nested ML / Hybrid SWAT-ML model comparison (M1-M5) for
monthly streamflow prediction at El Diem, Upper Blue Nile Basin.

    M1  Pure ML   : Meteorology only                          -> Q_obs
    M2  Pure ML   : Meteorology + discharge memory (lags)      -> Q_obs
    M3  Pure ML   : Meteorology + lags + teleconnections       -> Q_obs   (full pure-ML)
    M4  Hybrid    : Meteorology only, corrects SWAT error      -> Error -> Q_hybrid
    M5  Hybrid    : Meteorology + lags + teleconnections,
                    corrects SWAT error                        -> Error -> Q_hybrid (full hybrid)

TRAIN / VALIDATION / TEST split (user-specified):
    Train       1986-01 to 1997-12  (144 months)
    Validation  1998-01 to 2004-12  (84 months)
    Test        2005-01 to 2014-12  (120 months)

Hyperparameters are grid-searched on TRAIN, scored on VALIDATION
(strict chronological split, no shuffling), then the winning
configuration is refit on TRAIN+VALIDATION before a single, one-shot
evaluation on the held-out TEST period.

Outputs
-------
- tables/02_model_performance.csv        (NSE/KGE/R2/RMSE/MAE/PBIAS/RSR, train/val/test, all models)
- tables/02_best_hyperparameters.csv
- figures/02_hydrograph_test_all_models.png
- figures/02_scatter_best_models.png
- figures/02_metric_comparison_bars.png
- models/*.json                          (saved XGBoost models for use in 03_shap_analysis.py)
- master_dataset_with_predictions.csv    (full dataframe incl. all model predictions, for reuse)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from config import (FIG_DIR, TAB_DIR, MODEL_DIR, set_publication_style, COLORS, MODEL_LABELS,
                     MET_VARS, LAG_VARS, CLIM_VARS,
                     ML_TRAIN_START, ML_TRAIN_END, ML_VAL_START, ML_VAL_END,
                     ML_TEST_START, ML_TEST_END)
from metrics_utils import metrics_table, all_metrics
from data_prep import build_master, split_by_date
from ml_core import tune_and_fit

set_publication_style()

# ---------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------
master = build_master()

train = split_by_date(master, ML_TRAIN_START, ML_TRAIN_END)
val = split_by_date(master, ML_VAL_START, ML_VAL_END)
test = split_by_date(master, ML_TEST_START, ML_TEST_END)
trainval = pd.concat([train, val]).reset_index(drop=True)

print(f"Train n={len(train)}  Val n={len(val)}  Test n={len(test)}  (Total={len(master)})")
COLORS = {
    "Q_obs": "#000000",   # dark  — Observed
    "Q_SWAT": "#FF0000",  # red — SWAT
    "M1": "#2E27F5",      # blue
    "M2": "#F628C2",      # light purple
    "M3": "#008000",      # green
    "M4": "#00FFFF",      # cyan
    "M5": "#800080",      # purple
}
FEATURES = {
    "M1": MET_VARS,
    "M2": MET_VARS + LAG_VARS,
    "M3": MET_VARS + LAG_VARS + CLIM_VARS,
    "M4": MET_VARS,
    "M5": MET_VARS + LAG_VARS + CLIM_VARS,
}
IS_HYBRID = {"M1": False, "M2": False, "M3": False, "M4": True, "M5": True}

results_rows = []
best_params_rows = []
predictions = {}   # model_name -> {'train':..., 'test':...} arrays of Q
fitted_models = {}

for name, feats in FEATURES.items():
    target_col = "Error_SWAT" if IS_HYBRID[name] else "Q_obs"

    Xtr, ytr = train[feats], train[target_col]
    Xva, yva = val[feats], val[target_col]
    Xte, yte = test[feats], test[target_col]
    Xtv, ytv = trainval[feats], trainval[target_col]

    model, best_params, val_score = tune_and_fit(Xtr, ytr, Xva, yva, Xtv, ytv)
    fitted_models[name] = model
    best_params_rows.append({"Model": name, **best_params, "Val_NSE_target": round(val_score, 3)})
    model.save_model(os.path.join(MODEL_DIR, f"{name}.json"))

    # Predict on all three splits
    pred = {
        "train": model.predict(train[feats]),
        "val": model.predict(val[feats]),
        "test": model.predict(test[feats]),
    }

    # Convert to discharge space (Q_hybrid = Q_SWAT + predicted error, for hybrid models)
    Q_pred = {}
    for split_name, df_split in zip(["train", "val", "test"], [train, val, test]):
        if IS_HYBRID[name]:
            Q_pred[split_name] = df_split["Q_SWAT"].values + pred[split_name]
        else:
            Q_pred[split_name] = pred[split_name]
    predictions[name] = Q_pred

    # Metrics (always evaluated in DISCHARGE space against Q_obs, for fair cross-model comparison)
    for split_name, split_key, df_split in zip(["Train", "Validation", "Test"], ["train", "val", "test"], [train, val, test]):
        m = all_metrics(df_split["Q_obs"], Q_pred[split_key], label=f"{name} - {split_name}")
        m["Model"] = name
        m["Split"] = split_name
        results_rows.append(m)

    print(f"{name} done. Best params: {best_params}")

perf = pd.DataFrame(results_rows)
perf = perf[["Model", "Split", "N", "NSE", "KGE", "r", "alpha", "beta", "R2", "RMSE", "MAE", "PBIAS_%", "RSR", "Rating"]]
perf.to_csv(os.path.join(TAB_DIR, "02_model_performance.csv"), index=False)

pd.DataFrame(best_params_rows).to_csv(os.path.join(TAB_DIR, "02_best_hyperparameters.csv"), index=False)

print("\n=== Test-period (2006-2014) performance, all models ===")
print(perf[perf["Split"] == "Test"].sort_values("NSE", ascending=False).to_string(index=False))

# ---------------------------------------------------------------------
# 2. Save master dataset with all predictions attached (for SHAP / teleconnection scripts)
# ---------------------------------------------------------------------
full_pred = master.copy()
for name in FEATURES:
    feats = FEATURES[name]
    if IS_HYBRID[name]:
        err_pred_full = fitted_models[name].predict(master[feats])
        full_pred[f"Q_{name}"] = master["Q_SWAT"].values + err_pred_full
    else:
        full_pred[f"Q_{name}"] = fitted_models[name].predict(master[feats])
full_pred.to_csv(os.path.join(TAB_DIR, "master_dataset_with_predictions.csv"), index=False)

# ---------------------------------------------------------------------
# 3. Hydrograph: Test period (2006-2014), all models vs Observed & SWAT
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(16, 6))
ax.plot(test["Date"], test["Q_obs"], color=COLORS["Q_obs"], lw=3.0, label="Observed", zorder=10)
ax.plot(test["Date"], test["Q_SWAT"], color=COLORS["Q_SWAT"], lw=2.5, ls="-", label="SWAT stand-alone")
for name in ["M1", "M2", "M3", "M4", "M5"]:
    ax.plot(test["Date"], predictions[name]["test"], color=COLORS[name], lw=2.5,
             ls="-", alpha=0.9, label=MODEL_LABELS[name].split(":")[0])

#ax.set_xlabel("Year")
#ax.set_ylabel(r"Discharge (m$^3$/s)")
ax.set_xlabel("Year", fontsize=18, fontweight="bold")
ax.set_ylabel(r"Discharge (m$^3$/s)", fontsize=18, fontweight="bold")

plt.setp(ax.get_xticklabels(), fontsize=16, fontweight="bold")
plt.setp(ax.get_yticklabels(), fontsize=16, fontweight="bold")

ax.set_title("Test-Period (2006\u20132014) Discharge Prediction \u2013 El Diem", fontsize=22,fontweight="bold")
ax.xaxis.set_major_locator(mdates.YearLocator(1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

ax.legend(loc="upper left", frameon=False, ncol=7, fontsize=18.8)
plt.tight_layout()
for spine in ax.spines.values():
    spine.set_linewidth(2.5)
plt.setp(ax.get_xticklabels(), rotation=90, ha="center")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "02_hydrograph_test_all_models.png"))
plt.close(fig)

# ---------------------------------------------------------------------
# 4. Scatter: best pure-ML (M3) vs best hybrid (M5) on test set
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)
lims = [0, test["Q_obs"].max() * 1.05]

for ax, name, ttl in zip(axes, ["M3", "M5"], ["M3: Full Pure-ML", "M5: Full Hybrid SWAT-ML"]):
    ax.scatter(test["Q_obs"], predictions[name]["test"], s=60, color=COLORS[name],
               edgecolor="black", linewidth=1.0, alpha=0.8)
    ax.plot(lims, lims, color="black", lw=2.5, label="1:1 line")
    coef = np.polyfit(test["Q_obs"], predictions[name]["test"], 1)
    xr = np.array(lims)
    ax.plot(xr, coef[0] * xr + coef[1], color="red", lw=2.5, ls="--",
             label=f"y={coef[0]:.2f}x+{coef[1]:.1f}")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(r"Observed discharge (m$^3$/s)")
    ax.set_title(ttl)
    ax.legend(loc="upper left", frameon=False, fontsize=14)
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
axes[0].set_ylabel(r"Predicted discharge (m$^3$/s)")
fig.suptitle("Observed vs. Predicted \u2013 Test Period (2006\u20132014)")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "02_scatter_best_models.png"))
plt.close(fig)

# ---------------------------------------------------------------------
# 5. Bar chart: NSE and KGE across M1-M5 (SWAT included) for the Test period
# ---------------------------------------------------------------------
test_perf = perf[perf["Split"] == "Test"].set_index("Model")
swat_test = split_by_date(master, ML_TEST_START, ML_TEST_END)
swat_metrics_test = all_metrics(swat_test["Q_obs"], swat_test["Q_SWAT"], "SWAT")

labels_order = ["SWAT", "M1", "M2", "M3", "M4", "M5"]
nse_vals = [swat_metrics_test["NSE"]] + [test_perf.loc[m, "NSE"] for m in ["M1", "M2", "M3", "M4", "M5"]]
kge_vals = [swat_metrics_test["KGE"]] + [test_perf.loc[m, "KGE"] for m in ["M1", "M2", "M3", "M4", "M5"]]
bar_colors = [COLORS["Q_SWAT"]] + [COLORS[m] for m in ["M1", "M2", "M3", "M4", "M5"]]

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(labels_order))
w = 0.35
ax.bar(x - w / 2, nse_vals, width=w, color=bar_colors, edgecolor="black", linewidth=2.5, label="NSE")
ax.bar(x + w / 2, kge_vals, width=w, color=bar_colors, edgecolor="black", linewidth=2.5,
       alpha=0.55, hatch="//", label="KGE")
ax.set_xticks(x)
ax.set_xticklabels(labels_order)
ax.axhline(0.7, color="gray", lw=2.0, ls=":")
ax.set_ylabel("Efficiency")
ax.set_title("Model Performance Comparison \u2013 Test Period (2006\u20132014)")
ax.legend(loc="upper right", frameon=False,  ncol=2)
for spine in ax.spines.values():
    spine.set_linewidth(2.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "02_metric_comparison_bars.png"))
plt.close(fig)

print("\nML / Hybrid model comparison complete. Tables, figures, and models written to OUTPUT_DIR.")
