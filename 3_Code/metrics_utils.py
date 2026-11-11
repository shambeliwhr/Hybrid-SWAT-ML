# -*- coding: utf-8 -*-
"""
metrics_utils.py
=================
Standard hydrological model-performance metrics used throughout the
pipeline (SWAT stand-alone, ML models, hybrid models). Formulations
follow common hydrological modelling practice:

    NSE   - Nash-Sutcliffe Efficiency (Nash & Sutcliffe, 1970)
    KGE   - Kling-Gupta Efficiency (Gupta et al., 2009 formulation)
    R2    - Coefficient of determination
    RMSE  - Root Mean Square Error
    MAE   - Mean Absolute Error
    PBIAS - Percent Bias
    RSR   - RMSE-observations standard deviation Ratio (Moriasi et al., 2007)
"""

import numpy as np
import pandas as pd


def nse(obs, sim):
    obs, sim = np.asarray(obs, dtype=float), np.asarray(sim, dtype=float)
    denom = np.sum((obs - np.mean(obs)) ** 2)
    if denom == 0:
        return np.nan
    return 1.0 - np.sum((obs - sim) ** 2) / denom


def kge(obs, sim):
    """Gupta et al. (2009) KGE: KGE = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)"""
    obs, sim = np.asarray(obs, dtype=float), np.asarray(sim, dtype=float)
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim) / np.std(obs)          # variability ratio
    beta = np.mean(sim) / np.mean(obs)          # bias ratio
    return 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2), r, alpha, beta


def r_squared(obs, sim):
    obs, sim = np.asarray(obs, dtype=float), np.asarray(sim, dtype=float)
    r = np.corrcoef(obs, sim)[0, 1]
    return r ** 2


def rmse(obs, sim):
    obs, sim = np.asarray(obs, dtype=float), np.asarray(sim, dtype=float)
    return float(np.sqrt(np.mean((obs - sim) ** 2)))


def mae(obs, sim):
    obs, sim = np.asarray(obs, dtype=float), np.asarray(sim, dtype=float)
    return float(np.mean(np.abs(obs - sim)))


def pbias(obs, sim):
    obs, sim = np.asarray(obs, dtype=float), np.asarray(sim, dtype=float)
    return float(100.0 * np.sum(sim - obs) / np.sum(obs))


def rsr(obs, sim):
    obs, sim = np.asarray(obs, dtype=float), np.asarray(sim, dtype=float)
    return rmse(obs, sim) / np.std(obs)


def performance_rating(nse_val, pbias_val, rsr_val):
    """Moriasi et al. (2007, 2015) monthly-step performance rating."""
    if nse_val > 0.80 and abs(pbias_val) < 10 and rsr_val <= 0.50:
        return "Very good"
    elif nse_val > 0.70 and abs(pbias_val) < 15 and rsr_val <= 0.60:
        return "Good"
    elif nse_val > 0.50 and abs(pbias_val) < 25 and rsr_val <= 0.70:
        return "Satisfactory"
    else:
        return "Unsatisfactory"


def all_metrics(obs, sim, label=""):
    """Return a dict of all metrics for one obs/sim pair."""
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    kge_val, r, alpha, beta = kge(obs, sim)
    nse_val = nse(obs, sim)
    pbias_val = pbias(obs, sim)
    rsr_val = rsr(obs, sim)
    return {
        "Label": label,
        "N": len(obs),
        "NSE": round(nse_val, 3),
        "KGE": round(kge_val, 3),
        "r": round(r, 3),
        "alpha": round(alpha, 3),
        "beta": round(beta, 3),
        "R2": round(r_squared(obs, sim), 3),
        "RMSE": round(rmse(obs, sim), 2),
        "MAE": round(mae(obs, sim), 2),
        "PBIAS_%": round(pbias_val, 2),
        "RSR": round(rsr_val, 3),
        "Rating": performance_rating(nse_val, pbias_val, rsr_val),
    }


def metrics_table(pairs):
    """pairs: list of (label, obs, sim) -> tidy DataFrame."""
    rows = [all_metrics(obs, sim, label) for label, obs, sim in pairs]
    return pd.DataFrame(rows)
