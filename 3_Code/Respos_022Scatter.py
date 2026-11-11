# -*- coding: utf-8 -*-
"""
02_ml_models.py  (full version)
================================
Progressive nested ML / Hybrid SWAT-ML model comparison (M1-M5) for
monthly streamflow prediction at El Diem, Upper Blue Nile Basin.

    M1  Pure ML   : Meteorology only                          -> Q_obs
    M2  Pure ML   : Meteorology + discharge memory (lags)      -> Q_obs
    M3  Pure ML   : Meteorology + lags + teleconnections       -> Q_obs   (full pure-ML)
    M4  Hybrid    : Meteorology only, corrects SWAT error      -> Error -> Q_hybrid
    M5  Hybrid    : Meteorology + lags + teleconnections,
                    corrects SWAT error                        -> Error -> Q_hybrid (full hybrid)

Split (set in config.py; must match the manuscript):
    Train       1986-01 to 1997-12  (144 months)
    Validation  1998-01 to 2005-12  (96 months)
    Test        2006-01 to 2014-12  (108 months)

Outputs
-------
tables/02_model_performance.csv, 02_best_hyperparameters.csv
tables/02_accuracy_bootstrap_CI.csv, 02_diebold_mariano.csv
tables/master_dataset_with_predictions.csv
figures/02_hydrograph_test_all_models.png
figures/02_scatter_best_models.png          (M3 vs M5, metrics on panel, no equation)
figures/02_taylor_diagram.png
figures/02_error_box_violin.png
figures/02_metric_comparison_bars.png
models/*.json
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
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

# --- Larger text for Q1-journal standard (figures are ~14-16 in wide here and get scaled down in print) ---
plt.rcParams.update({
    "font.size": 18,
    "axes.labelsize": 22,
    "axes.labelweight": "bold",
    "axes.titlesize": 24,
    "axes.titleweight": "bold",
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 18,
    "figure.titlesize": 26,
    "figure.titleweight": "bold",
})

# ---------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------
master = build_master()

train = split_by_date(master, ML_TRAIN_START, ML_TRAIN_END)
val = split_by_date(master, ML_VAL_START, ML_VAL_END)
test = split_by_date(master, ML_TEST_START, ML_TEST_END)
trainval = pd.concat([train, val]).reset_index(drop=True)

print(f"Train n={len(train)}  Val n={len(val)}  Test n={len(test)}  (Total={len(master)})")
if len(test) != 108:
    print("WARNING: test set has", len(test), "months; the manuscript uses 108 (2006-01 to 2014-12). "
          "Check ML_VAL_END / ML_TEST_START in config.py.")

COLORS = {
    "Q_obs": "#000000",   # Observed
    "Q_SWAT": "#FF0000",  # SWAT
    "M1": "#2E27F5",
    "M2": "#F628C2",
    "M3": "#008000",
    "M4": "#00FFFF",
    "M5": "#800080",
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
predictions = {}
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

    pred = {
        "train": model.predict(train[feats]),
        "val": model.predict(val[feats]),
        "test": model.predict(test[feats]),
    }

    Q_pred = {}
    for split_name, df_split in zip(["train", "val", "test"], [train, val, test]):
        if IS_HYBRID[name]:
            Q_pred[split_name] = df_split["Q_SWAT"].values + pred[split_name]
        else:
            Q_pred[split_name] = pred[split_name]
    predictions[name] = Q_pred

    for split_name, split_key, df_split in zip(["Train", "Validation", "Test"],
                                               ["train", "val", "test"], [train, val, test]):
        m = all_metrics(df_split["Q_obs"], Q_pred[split_key], label=f"{name} - {split_name}")
        m["Model"] = name
        m["Split"] = split_name
        results_rows.append(m)

    print(f"{name} done. Best params: {best_params}")

perf = pd.DataFrame(results_rows)
perf = perf[["Model", "Split", "N", "NSE", "KGE", "r", "alpha", "beta", "R2", "RMSE", "MAE", "PBIAS_%", "RSR", "Rating"]]
perf.to_csv(os.path.join(TAB_DIR, "02_model_performance.csv"), index=False)
pd.DataFrame(best_params_rows).to_csv(os.path.join(TAB_DIR, "02_best_hyperparameters.csv"), index=False)

print("\n=== Test-period performance, all models ===")
print(perf[perf["Split"] == "Test"].sort_values("NSE", ascending=False).to_string(index=False))

# ---------------------------------------------------------------------
# 2. Save master dataset with all predictions
# ---------------------------------------------------------------------
full_pred = master.copy()
for name in FEATURES:
    feats = FEATURES[name]
    if IS_HYBRID[name]:
        full_pred[f"Q_{name}"] = master["Q_SWAT"].values + fitted_models[name].predict(master[feats])
    else:
        full_pred[f"Q_{name}"] = fitted_models[name].predict(master[feats])
full_pred.to_csv(os.path.join(TAB_DIR, "master_dataset_with_predictions.csv"), index=False)

# ---------------------------------------------------------------------
# 3. Hydrograph: test period, all models vs Observed & SWAT
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(16, 6))
ax.plot(test["Date"], test["Q_obs"], color=COLORS["Q_obs"], lw=3.0, label="Observed", zorder=10)
ax.plot(test["Date"], test["Q_SWAT"], color=COLORS["Q_SWAT"], lw=2.5, ls="-", label="SWAT stand-alone")
for name in ["M1", "M2", "M3", "M4", "M5"]:
    ax.plot(test["Date"], predictions[name]["test"], color=COLORS[name], lw=2.5,
            ls="-", alpha=0.9, label=MODEL_LABELS[name].split(":")[0])

ax.set_xlabel("Year", fontsize=18, fontweight="bold")
ax.set_ylabel(r"Discharge (m$^3$/s)", fontsize=18, fontweight="bold")
plt.setp(ax.get_xticklabels(), fontsize=16, fontweight="bold")
plt.setp(ax.get_yticklabels(), fontsize=16, fontweight="bold")
ax.set_title("Test-Period (2006\u20132014) Discharge Prediction \u2013 El Diem", fontsize=22, fontweight="bold")
ax.xaxis.set_major_locator(mdates.YearLocator(1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.legend(loc="upper left", frameon=False, ncol=7, fontsize=20)
for spine in ax.spines.values():
    spine.set_linewidth(2.5)
plt.setp(ax.get_xticklabels(), rotation=90, ha="center")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "02_hydrograph_test_all_models.png"))
plt.close(fig)

# ---------------------------------------------------------------------
# 4. Scatter: M3 (full pure-ML) vs M5 (full hybrid) on test set
#    Panel text shows only NSE, R2, KGE, RMSE (as reported in the manuscript). No y = ax + b.
# ---------------------------------------------------------------------
STATS = {
    "M3": {"NSE": 0.889, "R2": 0.889, "KGE": 0.906, "RMSE": 702.1},
    "M5": {"NSE": 0.918, "R2": 0.918, "KGE": 0.936, "RMSE": 564.65},
}

fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
lims = [0, test["Q_obs"].max() * 1.05]

for ax, name, ttl in zip(axes, ["M5", "M3"], ["M5: Full Hybrid SWAT-ML", "M3: Full Pure-ML"]):
    x = test["Q_obs"].values
    y = np.asarray(predictions[name]["test"])
    n = len(x)

    # OLS regression with 95% confidence and prediction intervals
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(lims[0], lims[1], 300)
    yfit = intercept + slope * xs
    s_err = np.sqrt(np.sum((y - (intercept + slope * x)) ** 2) / (n - 2))
    tcrit = stats.t.ppf(0.975, n - 2)
    sxx = np.sum((x - x.mean()) ** 2)
    lev = 1.0 / n + (xs - x.mean()) ** 2 / sxx
    ci = tcrit * s_err * np.sqrt(lev)          # confidence band of the mean
    pi = tcrit * s_err * np.sqrt(1.0 + lev)    # prediction band of a new observation

    # shaded bands (prediction interval light cyan, confidence interval sky blue)
    ax.fill_between(xs, yfit - pi, yfit + pi, color="#DDFFFF", zorder=1, label="95% prediction interval")
    ax.fill_between(xs, yfit - ci, yfit + ci, color="#87CEEB", zorder=2, label="95% confidence interval")

    ax.scatter(x, y, s=80, color=COLORS[name], edgecolor="black", linewidth=1.0, alpha=0.85, zorder=4)
    ax.plot(xs, yfit, color="red", lw=2.5, zorder=3, label="Regression line")
    ax.plot(lims, lims, color="black", lw=2.5, ls="--", zorder=3, label="1:1 line")

    st = STATS[name]
    txt = (f"NSE = {st['NSE']:.3f}\n"
           f"R$^2$ = {st['R2']:.3f}\n"
           f"KGE = {st['KGE']:.3f}\n"
           f"RMSE = {st['RMSE']:.2f} m$^3$/s")
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, ha="left", va="top",
            fontsize=22, fontweight="bold", zorder=6,
            bbox=dict(boxstyle="round", fc="white", ec="black", lw=1.5, alpha=0.9))

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(r"Observed discharge (m$^3$/s)")
    ax.set_title(ttl)

    # legend order: regression, 1:1, prediction interval, confidence interval
    h, l = ax.get_legend_handles_labels()
    want = ["Regression line", "1:1 line", "95% prediction interval", "95% confidence interval"]
    order_idx = [l.index(w) for w in want]
    ax.legend([h[i] for i in order_idx], [l[i] for i in order_idx],
              loc="lower right", frameon=True, edgecolor="black", fontsize=18).set_zorder(7)
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
axes[0].set_ylabel(r"Predicted discharge (m$^3$/s)")
fig.suptitle("Observed vs. Predicted \u2013 Test Period (2006\u20132014)")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "02_scatter_best_models.png"))
plt.close(fig)

# ---------------------------------------------------------------------
# 5. Bar chart: NSE and KGE (SWAT + M1-M5), test period
# ---------------------------------------------------------------------
test_perf = perf[perf["Split"] == "Test"].set_index("Model")
swat_metrics_test = all_metrics(test["Q_obs"], test["Q_SWAT"], "SWAT")

labels_order = ["SWAT", "M1", "M2", "M3", "M4", "M5"]
nse_vals = [swat_metrics_test["NSE"]] + [test_perf.loc[m, "NSE"] for m in labels_order[1:]]
kge_vals = [swat_metrics_test["KGE"]] + [test_perf.loc[m, "KGE"] for m in labels_order[1:]]
bar_colors = [COLORS["Q_SWAT"]] + [COLORS[m] for m in labels_order[1:]]

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
ax.legend(loc="upper right", frameon=False, ncol=2, fontsize=20)
for spine in ax.spines.values():
    spine.set_linewidth(2.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "02_metric_comparison_bars.png"))
plt.close(fig)

# ---------------------------------------------------------------------
# 6. Taylor diagram (test period; SWAT + M1-M5; observed = reference)
# ---------------------------------------------------------------------
obs = test["Q_obs"].values
sim = {"SWAT": test["Q_SWAT"].values}
for m in ["M1", "M2", "M3", "M4", "M5"]:
    sim[m] = np.asarray(predictions[m]["test"])
plot_colors = {"SWAT": COLORS["Q_SWAT"], **{m: COLORS[m] for m in ["M1", "M2", "M3", "M4", "M5"]}}

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, polar=True)
ax.set_thetamin(0)
ax.set_thetamax(90)

# centred-RMS-difference contours (normalised by observed std)
T, R = np.meshgrid(np.linspace(0, np.pi / 2, 300), np.linspace(0, 1.6, 300))
crmsd = np.sqrt(1 + R ** 2 - 2 * R * np.cos(T))
cs = ax.contour(T, R, crmsd, levels=[0.25, 0.5, 0.75, 1.0, 1.25], colors="grey", linestyles=":", linewidths=1.5)
ax.clabel(cs, fmt="%.2f", fontsize=14)

rt = [0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0]
ax.set_xticks(np.arccos(rt))
ax.set_xticklabels([str(v) for v in rt], fontsize=18)
ax.set_ylim(0, 1.6)
ax.set_xlabel("Normalised standard deviation", fontsize=20, fontweight="bold", labelpad=14)
ax.tick_params(axis="y", labelsize=18)
ax.text(np.pi / 4, 1.78, "Correlation coefficient", rotation=-45, ha="center", va="center", fontsize=20, fontweight="bold")

ax.plot(0, 1, marker="*", color="black", ms=22, ls="", label="Observed", zorder=10)
for m, s_ in sim.items():
    r = np.corrcoef(obs, s_)[0, 1]
    sd = s_.std(ddof=1) / obs.std(ddof=1)
    ax.plot(np.arccos(r), sd, "o", ms=14, color=plot_colors[m], mec="black", mew=1.2,
            ls="", label=m)
ax.set_title("Taylor diagram \u2013 Test Period (2006\u20132014)", fontsize=22, fontweight="bold", pad=32)
ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.0), frameon=False, fontsize=18)
fig.savefig(os.path.join(FIG_DIR, "02_taylor_diagram.png"), bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------
# 7. Error boxplot and violin (residual = Q_obs - Q_sim, test period)
# ---------------------------------------------------------------------
order = ["SWAT", "M1", "M2", "M3", "M4", "M5"]
resid = {m: obs - sim[m] for m in order}

fig, axs = plt.subplots(1, 2, figsize=(18, 8), sharey=True)

# (a) boxplot
bp = axs[0].boxplot([resid[m] for m in order], patch_artist=True, showmeans=True, widths=0.6,
                    meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=8),
                    medianprops=dict(color="black", lw=2.5),
                    flierprops=dict(marker="o", markersize=5, alpha=0.6))
for patch, m in zip(bp["boxes"], order):
    patch.set_facecolor(plot_colors[m])
    patch.set_alpha(0.75)
    patch.set_linewidth(2)
axs[0].set_xticklabels(order)
axs[0].set_title("(a) Boxplot of residuals")

# (b) violin
vp = axs[1].violinplot([resid[m] for m in order], showmeans=True, showmedians=True, widths=0.8)
for body, m in zip(vp["bodies"], order):
    body.set_facecolor(plot_colors[m])
    body.set_edgecolor("black")
    body.set_alpha(0.75)
for key in ["cmeans", "cmedians", "cbars", "cmins", "cmaxes"]:
    vp[key].set_color("black")
    vp[key].set_linewidth(1.8)
axs[1].set_xticks(range(1, len(order) + 1))
axs[1].set_xticklabels(order)
axs[1].set_title("(b) Violin plot of residuals")

for a in axs:
    a.axhline(0, color="red", ls="--", lw=2.0)
    for spine in a.spines.values():
        spine.set_linewidth(2.5)
axs[0].set_ylabel(r"Residual, $Q_{obs}-Q_{sim}$ (m$^3$/s)")
fig.suptitle("Prediction errors \u2013 Test Period (2006\u20132014)")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "02_error_box_violin.png"))
plt.close(fig)

# ---------------------------------------------------------------------
# 8. Accuracy tests: block-bootstrap CIs and Diebold-Mariano tests
#    (moving-block bootstrap / HAC variance because residuals are autocorrelated)
# ---------------------------------------------------------------------
def nse_f(o, s):
    return 1 - np.sum((o - s) ** 2) / np.sum((o - o.mean()) ** 2)


def rmse_f(o, s):
    return np.sqrt(np.mean((o - s) ** 2))


def _block_indices(n, L, rng):
    nb = int(np.ceil(n / L))
    starts = rng.integers(0, n - L + 1, nb)
    return np.concatenate([np.arange(a, a + L) for a in starts])[:n]


def block_boot_ci(o, s, fun, L=12, B=2000, seed=1):
    rng = np.random.default_rng(seed)
    vals = [fun(o[i], s[i]) for i in (_block_indices(len(o), L, rng) for _ in range(B))]
    return np.percentile(vals, [2.5, 97.5])


def block_boot_diff_ci(o, s_a, s_b, fun, L=12, B=2000, seed=1):
    """95% CI of fun(o,s_a) - fun(o,s_b) using the same resampled blocks for both models."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        i = _block_indices(len(o), L, rng)
        vals.append(fun(o[i], s_a[i]) - fun(o[i], s_b[i]))
    return np.percentile(vals, [2.5, 97.5])


def dm_test(e1, e2, lag=6):
    """Diebold-Mariano (squared-error loss, Newey-West variance). H0: equal accuracy.
    Negative statistic -> model 1 has smaller squared error than model 2."""
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    dc = d - d.mean()
    lrv = np.sum(dc ** 2) / n
    for k in range(1, lag + 1):
        lrv += 2 * (1 - k / (lag + 1)) * np.sum(dc[k:] * dc[:-k]) / n
    stat = d.mean() / np.sqrt(lrv / n)
    p = 2 * stats.t.sf(abs(stat), n - 1)
    return stat, p


rows = []
for m in order:
    lo_n, hi_n = block_boot_ci(obs, sim[m], nse_f)
    lo_r, hi_r = block_boot_ci(obs, sim[m], rmse_f)
    rows.append({"Model": m, "NSE": nse_f(obs, sim[m]), "NSE_CI_low": lo_n, "NSE_CI_high": hi_n,
                 "RMSE": rmse_f(obs, sim[m]), "RMSE_CI_low": lo_r, "RMSE_CI_high": hi_r})
ci_df = pd.DataFrame(rows).round(3)
ci_df.to_csv(os.path.join(TAB_DIR, "02_accuracy_bootstrap_CI.csv"), index=False)
print("\n=== Block-bootstrap 95% CIs (test period) ===")
print(ci_df.to_string(index=False))

pairs = [("SWAT", "M1"), ("SWAT", "M2"), ("SWAT", "M3"), ("SWAT", "M4"), ("SWAT", "M5"),
         ("M3", "M5"), ("M4", "M5")]
dm_rows = []
for a, b in pairs:
    stat, p = dm_test(resid[a], resid[b])
    lo, hi = block_boot_diff_ci(obs, sim[a], sim[b], rmse_f)
    dm_rows.append({"Model_A": a, "Model_B": b, "DM_stat": stat, "p_value": p,
                    "Significant_0.05": "Yes" if p < 0.05 else "No",
                    "dRMSE_A_minus_B": rmse_f(obs, sim[a]) - rmse_f(obs, sim[b]),
                    "dRMSE_CI_low": lo, "dRMSE_CI_high": hi})
dm_df = pd.DataFrame(dm_rows).round(4)
dm_df.to_csv(os.path.join(TAB_DIR, "02_diebold_mariano.csv"), index=False)
print("\n=== Diebold-Mariano tests (squared error, test period) ===")
print(dm_df.to_string(index=False))

print("\nML / Hybrid model comparison complete. Tables, figures, and models written to OUTPUT_DIR.")