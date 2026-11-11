"""
04 - Experiment B: Hybrid SWAT-ML Error-Correction Framework
================================================================
E(t) = Q_obs(t) - Q_SWAT(t)
E(t) = f(P, Tmax, Tmin, RH, SR, WS, Q_lag1, Q_lag2, Q_lag3, ENSO, IOD, NAO)
Q_hybrid(t) = Q_SWAT(t) + E_ML(t)

Same time-ordered split as Experiment A:
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
DATA = "D:/Flow/Discharge_lag"
FIG = "D:/Flow/Discharge_lag/figs"
TAB = "D:/Flow/Discharge_lag/tables"
MOD = "D:/Flow/Discharge_lag/models"
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)
os.makedirs(MOD, exist_ok=True)

base = pd.read_csv(f"{DATA}/ElDiem_ML_BASE.csv")
qvar = pd.read_csv(f"{DATA}/ElDiem_Qvar.csv")[["Date", "Q_t-1", "Q_t-2", "Q_t-3"]]
base["Date"] = pd.to_datetime(base["Date"], format="%Y-%m")
qvar["Date"] = pd.to_datetime(qvar["Date"], format="%Y-%m")

df = base.merge(qvar, on="Date", how="inner").sort_values("Date").reset_index(drop=True)
df = df.rename(columns={"Q_t-1": "Q_lag1", "Q_t-2": "Q_lag2", "Q_t-3": "Q_lag3"})
df["Error"] = df["Q_obs"] - df["Q_SWAT"]

features = ["P", "Tmax", "Tmin", "RH", "SR", "WS",
            "Q_lag1", "Q_lag2", "Q_lag3", "ENSO", "IOD", "NAO"]
target = "Error"

train, val, test = time_split(df)
print(f"Train: {train.shape[0]} obs ({train.Date.min().date()}-{train.Date.max().date()})")
print(f"Val:   {val.shape[0]} obs ({val.Date.min().date()}-{val.Date.max().date()})")
print(f"Test:  {test.shape[0]} obs ({test.Date.min().date()}-{test.Date.max().date()})")
Xtr, ytr = train[features], train[target]
Xval, yval = val[features], val[target]
Xte, yte = test[features], test[target]

X_trval = pd.concat([Xtr, Xval]); y_trval = pd.concat([ytr, yval])
test_fold = np.r_[np.full(len(Xtr), -1), np.zeros(len(Xval))]
ps = PredefinedSplit(test_fold)

results = {}
predictions = {"Date": test["Date"].values, "Q_obs": test["Q_obs"].values,
               "Q_SWAT": test["Q_SWAT"].values}
importances = {}

# ---- Random Forest error-correction ----
rf_grid = {"n_estimators": [200, 400, 600], "max_depth": [3, 4, 6, None],
           "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", 0.6, 1.0]}
rf_search = RandomizedSearchCV(RandomForestRegressor(random_state=42), rf_grid, n_iter=15,
                                cv=ps, scoring="neg_root_mean_squared_error", random_state=42, n_jobs=-1)
rf_search.fit(X_trval, y_trval)
rf_best = rf_search.best_estimator_
rf_best.fit(X_trval, y_trval)
rf_err_pred = rf_best.predict(Xte)
rf_hybrid = test["Q_SWAT"].values + rf_err_pred
results["Hybrid RF"] = hydro_metrics(test["Q_obs"], rf_hybrid)
predictions["Hybrid_RF"] = rf_hybrid
importances["Hybrid RF"] = pd.Series(rf_best.feature_importances_, index=features)
joblib.dump(rf_best, f"{MOD}/expB_hybrid_rf.joblib")

# ---- XGBoost error-correction (primary) ----
xgb_grid = {"n_estimators": [200, 400, 600], "max_depth": [2, 3, 4, 5],
            "learning_rate": [0.01, 0.03, 0.05, 0.1], "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0], "reg_lambda": [0.5, 1, 2]}
xgb_search = RandomizedSearchCV(xgb.XGBRegressor(random_state=42, objective="reg:squarederror"),
                                 xgb_grid, n_iter=25, cv=ps, scoring="neg_root_mean_squared_error",
                                 random_state=42, n_jobs=-1)
xgb_search.fit(X_trval, y_trval)
xgb_best = xgb_search.best_estimator_
xgb_best.fit(X_trval, y_trval)
xgb_err_pred = xgb_best.predict(Xte)
xgb_hybrid = test["Q_SWAT"].values + xgb_err_pred
results["Hybrid XGBoost"] = hydro_metrics(test["Q_obs"], xgb_hybrid)
predictions["Hybrid_XGB"] = xgb_hybrid
importances["Hybrid XGBoost"] = pd.Series(xgb_best.feature_importances_, index=features)
joblib.dump(xgb_best, f"{MOD}/expB_hybrid_xgb.joblib")

# ---- Baseline SWAT (no correction) for reference ----
results["SWAT (no correction)"] = hydro_metrics(test["Q_obs"], test["Q_SWAT"])

res_df = pd.DataFrame(results).T.round(3)
res_df.to_csv(f"{TAB}/table7_expB_hybrid_test_metrics.csv")
print("Experiment B — Test-period performance (Hybrid SWAT-ML):")
print(res_df)

pred_df = pd.DataFrame(predictions)
pred_df.to_csv(f"{TAB}/expB_test_predictions.csv", index=False)
imp_df = pd.DataFrame(importances).round(4)
imp_df.to_csv(f"{TAB}/table8_expB_feature_importance.csv")

improve = pd.DataFrame({
    "NSE_gain_vs_SWAT_%": [
        100 * (results["Hybrid RF"]["NSE"] - results["SWAT (no correction)"]["NSE"]) / abs(results["SWAT (no correction)"]["NSE"]),
        100 * (results["Hybrid XGBoost"]["NSE"] - results["SWAT (no correction)"]["NSE"]) / abs(results["SWAT (no correction)"]["NSE"]),
    ],
    "RMSE_reduction_%": [
        100 * (results["SWAT (no correction)"]["RMSE"] - results["Hybrid RF"]["RMSE"]) / results["SWAT (no correction)"]["RMSE"],
        100 * (results["SWAT (no correction)"]["RMSE"] - results["Hybrid XGBoost"]["RMSE"]) / results["SWAT (no correction)"]["RMSE"],
    ]}, index=["Hybrid RF", "Hybrid XGBoost"]).round(1)
improve.to_csv(f"{TAB}/table9_hybrid_improvement_over_SWAT.csv")
print("\nImprovement over raw SWAT:\n", improve)

# ---- Plots ----
fig, ax = plt.subplots(figsize=(12, 5.5))
ax.plot(pred_df["Date"], pred_df["Q_obs"], color="black", lw=3.0, label="Q_obs")
ax.plot(pred_df["Date"], pred_df["Q_SWAT"], color="grey", lw=2.2, ls="--", label="Q_SWAT (raw)")
ax.plot(pred_df["Date"], pred_df["Hybrid_RF"], color="#1f77b4", lw=2.4, label="Hybrid SWAT-RF")
ax.plot(pred_df["Date"], pred_df["Hybrid_XGB"], color="#d62728", lw=2.4, label="Hybrid SWAT-XGBoost")
ax.set_ylabel("Discharge (m$^3$/s)")
ax.set_title("Experiment B — Hybrid SWAT-ML Streamflow Prediction (Test period 2006–2014)")
ax.legend(frameon=False, ncol=4, fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIG}/fig7_expB_test_hydrograph.png", bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(6, 4.5))
imp_df["Hybrid XGBoost"].sort_values().plot.barh(ax=ax, color="#d62728")
ax.set_xlabel("Feature importance")
ax.set_title("Experiment B — XGBoost Error-Correction Feature Importance")
plt.tight_layout()
plt.savefig(f"{FIG}/fig8_expB_feature_importance.png", bbox_inches="tight")
plt.close()

print("\n=== 04_experimentB_hybrid.py complete ===")
