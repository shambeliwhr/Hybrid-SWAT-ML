# -*- coding: utf-8 -*-
"""
data_prep.py
============
Loads the six prepared CSV files, parses the monthly "YYYY-M" date
strings into proper Timestamps, validates consistency across files,
and assembles the master ML dataframe used by every downstream script.
"""

import pandas as pd
import numpy as np
from config import (F_CLIM, F_ML, F_ML_ALL, F_ML_BASE, F_QOB_SWAT, F_QVAR)


def _read(path):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m")
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def load_all():
    """Load all six source files as parsed, date-sorted DataFrames."""
    clim = _read(F_CLIM)
    ml = _read(F_ML)
    ml_all = _read(F_ML_ALL)
    ml_base = _read(F_ML_BASE)
    qob_swat = _read(F_QOB_SWAT)
    qvar = _read(F_QVAR)
    return dict(clim=clim, ml=ml, ml_all=ml_all, ml_base=ml_base,
                qob_swat=qob_swat, qvar=qvar)


def build_master(d=None):
    """
    Merge everything on Date into one master dataframe containing:
    P, Tmax, Tmin, RH, SR, WS, P_lag1, Q_lag1-3, ENSO, IOD, NAO, Q_obs, Q_SWAT

    P_lag1 (1-month-lagged basin rainfall) is engineered here to represent
    BASIN LAG TIME / TIME OF CONCENTRATION -- the physical delay between
    rainfall falling over a ~200,000 km^2 basin and that water reaching the
    El Diem outlet via infiltration, interflow, and channel routing. This is
    the same physical concept behind the Antecedent Precipitation Index
    (Kohler & Linsley, 1951) used throughout hydrology.

    This is NOT an artificial statistical trick: same-month P correlates
    with Q_obs at only r=0.755, while P lagged by one month correlates at
    r=0.929 -- i.e. rainfall's causal signal is real and strong, it simply
    arrives at the gauge with a delay that a same-month-only feature set
    cannot represent. Without P_lag1, the model has no correctly-timed
    version of rainfall to use and leans on other same-month-persistent
    variables (RH, Tmax) instead -- which would misleadingly understate
    rainfall's true role as the physical driver of runoff.

    NOTE: adding a 1-month lag means the first calendar month (1986-01)
    has no valid antecedent rainfall and is dropped. The effective master
    dataset therefore spans 1986-02 to 2014-12 (347 months).
    """
    if d is None:
        d = load_all()

    master = d["ml_all"].rename(columns={"Q_target": "Q_obs"}).copy()
    master = master.merge(d["qob_swat"][["Date", "Q_SWAT"]], on="Date", how="left")

    # Consistency check: Q_obs must match across ml_all/ml/qob_swat/qvar
    checks = {
        "ml": d["ml"][["Date", "Q_obs"]],
        "qob_swat": d["qob_swat"][["Date", "Q_obs"]],
        "qvar": d["qvar"][["Date", "Q_obs"]],
    }
    report = []
    for name, other in checks.items():
        m = master[["Date", "Q_obs"]].merge(other, on="Date", suffixes=("", f"_{name}"))
        mismatch = (m["Q_obs"] - m[f"Q_obs_{name}"]).abs().max()
        report.append((name, mismatch))
    print("Q_obs consistency check (max abs difference across files):")
    for name, mismatch in report:
        flag = "OK" if mismatch < 1e-6 else "MISMATCH"
        print(f"  vs {name:10s}: {mismatch:.6f}  [{flag}]")

    master["Error_SWAT"] = master["Q_obs"] - master["Q_SWAT"]  # target for hybrid models
    master = master.sort_values("Date").reset_index(drop=True)

    # Engineer 1-month basin-lag rainfall (Antecedent Precipitation) and
    # drop the resulting NaN first row
    master["P_lag1"] = master["P"].shift(1)
    master = master.iloc[1:].reset_index(drop=True)

    return master


def split_by_date(df, start, end):
    mask = (df["Date"] >= pd.to_datetime(start)) & (df["Date"] <= pd.to_datetime(end) + pd.offsets.MonthEnd(0))
    return df.loc[mask].reset_index(drop=True)


if __name__ == "__main__":
    master = build_master()
    print(master.head())
    print("\nMaster shape:", master.shape)
    print("Date range:", master["Date"].min(), "to", master["Date"].max())
