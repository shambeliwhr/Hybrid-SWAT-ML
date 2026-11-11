# -*- coding: utf-8 -*-
"""
ml_core.py
==========
Shared XGBoost training / hyperparameter-tuning routine used for every
nested model (M1-M5). Uses a strict chronological train -> validation
split for hyperparameter selection (NO shuffling / k-fold, to avoid
leakage in a time series), then refits the selected model on
train+validation before scoring on the untouched test period.
"""

import itertools
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from metrics_utils import nse

PARAM_GRID = {
    "n_estimators": [200, 400],
    "max_depth": [2, 3, 4],
    "learning_rate": [0.03, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

FIXED_PARAMS = dict(
    objective="reg:squarederror",
    random_state=42,
    n_jobs=4,
)


def _grid_iter(grid):
    keys = list(grid.keys())
    for values in itertools.product(*[grid[k] for k in keys]):
        yield dict(zip(keys, values))


def tune_and_fit(X_train, y_train, X_val, y_val, X_trainval, y_trainval,
                  param_grid=None, verbose=False):
    """
    1) Grid-search hyperparameters by fitting on TRAIN and scoring NSE on VAL
       (chronological, single split -- appropriate for a short monthly series).
    2) Refit the best hyperparameters on TRAIN+VAL (more data for the final model).
    Returns: fitted_model, best_params, best_val_nse
    """
    grid = param_grid or PARAM_GRID
    best_score = -np.inf
    best_params = None

    for params in _grid_iter(grid):
        model = XGBRegressor(**FIXED_PARAMS, **params)
        model.fit(X_train, y_train)
        pred_val = model.predict(X_val)
        score = nse(y_val, pred_val)
        if score > best_score:
            best_score = score
            best_params = params
        if verbose:
            print(params, "val NSE=", round(score, 3))

    final_model = XGBRegressor(**FIXED_PARAMS, **best_params)
    final_model.fit(X_trainval, y_trainval)
    return final_model, best_params, best_score
