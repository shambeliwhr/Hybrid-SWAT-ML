"""
03 - Experiment A: Pure Machine-Learning Streamflow Prediction
================================================================
Q_obs(t) = f(P, Tmax, Tmin, RH, SR, WS, Q_lag1, Q_lag2, Q_lag3, ENSO, IOD, NAO)

Models: Random Forest (benchmark) and XGBoost (primary), compared
against SWAT process-based simulation. Time-ordered split (no shuffling)
to avoid information leakage:
    Train      1986-01 to 1997-12  (12 yr, 144 mo)
    Validation 1998-01 to 2005-12  ( 8 yr,  96 mo)
    Test       2006-01 to 2014-12  ( 9 yr, 108 mo)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
import xgboost as xgb
import joblib
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_hydro import hydro_metrics, time_split, set_plot_style

set_plot_style()
np.random.seed(42)
# NOTE ON REPRODUCIBILITY: RandomForestRegressor and XGBRegressor below are
# pinned to n_jobs=1 (single-threaded), and RandomizedSearchCV is likewise
# pinned to n_jobs=1. This is deliberate: random_state alone does NOT
# guarantee bit-identical results across machines when multi-threading is
# enabled, because XGBoost's histogram-based split-finding sums floating-
# point values across threads in an order that depends on core count and OS
# thread scheduling. Different machines (or even different runs on a
# multi-core machine) can therefore select a different "best" hyperparameter
# set and produce a materially different trained model, even with the same
# random_state. Running single-threaded removes that source of instability
# at the cost of longer training time; do not change n_jobs back to -1 if
# exact, cross-machine-reproducible figures/tables matter.
DATA = "D:/Flow/Discharge_lag"
FIG = "D:/Flow/Discharge_lag/figs"
TAB = "D:/Flow/Discharge_lag/tables"
MOD = "D:/Flow/Discharge_lag/models"
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)
os.makedirs(MOD, exist_ok=True)

df = pd.read_csv(f"{DATA}/ElDiem_ML_ALL.csv")
df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m")
df = df.sort_values("Date").reset_index(drop=True)

features = ["P", "Tmax", "Tmin", "RH", "SR", "WS",
            "Q_lag1", "Q_lag2", "Q_lag3", "ENSO", "IOD", "NAO"]
target = "Q_target"

train, val, test = time_split(df)
print(f"Train: {train.shape[0]} obs ({train.Date.min().date()}-{train.Date.max().date()})")
print(f"Val:   {val.shape[0]} obs ({val.Date.min().date()}-{val.Date.max().date()})")
print(f"Test:  {test.shape[0]} obs ({test.Date.min().date()}-{test.Date.max().date()})")

Xtr, ytr = train[features], train[target]
Xval, yval = val[features], val[target]
Xte, yte = test[features], test[target]

# combine train+val for CV with a PredefinedSplit (val used as internal validation fold)
X_trval = pd.concat([Xtr, Xval])
y_trval = pd.concat([ytr, yval])
test_fold = np.r_[np.full(len(Xtr), -1), np.zeros(len(Xval))]
ps = PredefinedSplit(test_fold)

results = {}
importances = {}
predictions = {"Date": test["Date"].values, "Q_obs": yte.values}

# ---------------------------------------------------------------
# Random Forest (benchmark)
# ---------------------------------------------------------------
rf_grid = {
    "n_estimators": [200, 400, 600],
    "max_depth": [4, 6, 8, None],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", 0.6, 1.0],
}
rf_search = RandomizedSearchCV(RandomForestRegressor(random_state=42, n_jobs=1), rf_grid,
                                n_iter=15, cv=ps, scoring="neg_root_mean_squared_error",
                                random_state=42, n_jobs=1)
rf_search.fit(X_trval, y_trval)
rf_best = rf_search.best_estimator_
rf_best.fit(pd.concat([Xtr, Xval]), pd.concat([ytr, yval]))  # refit train+val
rf_pred = rf_best.predict(Xte)
results["Random Forest"] = hydro_metrics(yte, rf_pred)
importances["Random Forest"] = pd.Series(rf_best.feature_importances_, index=features)
predictions["RF_pred"] = rf_pred
joblib.dump(rf_best, f"{MOD}/expA_random_forest.joblib")
print("RF best params:", rf_search.best_params_)

# ---------------------------------------------------------------
# XGBoost (primary model)
# ---------------------------------------------------------------
xgb_grid = {
    "n_estimators": [200, 400, 600],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_lambda": [0.5, 1, 2],
}
xgb_search = RandomizedSearchCV(
    xgb.XGBRegressor(random_state=42, objective="reg:squarederror", n_jobs=1),
    xgb_grid, n_iter=25, cv=ps, scoring="neg_root_mean_squared_error",
    random_state=42, n_jobs=1)
xgb_search.fit(X_trval, y_trval)
xgb_best = xgb_search.best_estimator_
xgb_best.fit(pd.concat([Xtr, Xval]), pd.concat([ytr, yval]))
xgb_pred = xgb_best.predict(Xte)
results["XGBoost"] = hydro_metrics(yte, xgb_pred)
importances["XGBoost"] = pd.Series(xgb_best.feature_importances_, index=features)
predictions["XGB_pred"] = xgb_pred
joblib.dump(xgb_best, f"{MOD}/expA_xgboost.joblib")
print("XGB best params:", xgb_search.best_params_)

# ---------------------------------------------------------------
# SWAT for the same test window (for direct comparison)
# ---------------------------------------------------------------
base = pd.read_csv(f"{DATA}/ElDiem_ML_BASE.csv")
base["Date"] = pd.to_datetime(base["Date"], format="%Y-%m")
swat_test = base.merge(test[["Date"]], on="Date", how="inner").sort_values("Date")
results["SWAT"] = hydro_metrics(swat_test["Q_obs"], swat_test["Q_SWAT"])
predictions["Q_SWAT"] = swat_test["Q_SWAT"].values

# ---------------------------------------------------------------
# Save results
# ---------------------------------------------------------------
res_df = pd.DataFrame(results).T.round(3)
res_df.to_csv(f"{TAB}/table5_expA_pureML_test_metrics.csv")
print("\nExperiment A — Test-period performance:")
print(res_df)

pred_df = pd.DataFrame(predictions)
pred_df.to_csv(f"{TAB}/expA_test_predictions.csv", index=False)

imp_df = pd.DataFrame(importances).round(4)
imp_df.to_csv(f"{TAB}/table6_expA_feature_importance.csv")

# ---------------------------------------------------------------
# Plots
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 5.5))
ax.plot(pred_df["Date"], pred_df["Q_obs"], color="black", lw=3.0, label="Q_obs")
ax.plot(pred_df["Date"], pred_df["Q_SWAT"], color="grey", lw=2.2, ls="--", label="Q_SWAT")
ax.plot(pred_df["Date"], pred_df["RF_pred"], color="#1f77b4", lw=2.4, label="RF (pure ML)")
ax.plot(pred_df["Date"], pred_df["XGB_pred"], color="#d62728", lw=2.4, label="XGBoost (pure ML)")
ax.set_ylabel("Discharge (m$^3$/s)")
ax.set_title("Experiment A — Pure-ML Streamflow Prediction (Test period 2006–2014)")
ax.legend(frameon=False, ncol=4, fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIG}/fig5_expA_test_hydrograph.png", bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(6, 4.5))
imp_df["XGBoost"].sort_values().plot.barh(ax=axes[1], color="#d62728")
axes[1].set_title("XGBoost")
imp_df["Random Forest"].sort_values().plot.barh(ax=axes[0], color="#1f77b4")
axes[0].set_title("Random Forest")
for a in axes: a.set_xlabel("Feature importance")
plt.suptitle("Experiment A — Feature Importance")
plt.tight_layout()
plt.savefig(f"{FIG}/fig6_expA_feature_importance.png", bbox_inches="tight")
plt.close()

print("\n=== 03_experimentA_pure_ML.py complete ===")
