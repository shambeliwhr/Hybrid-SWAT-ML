# -*- coding: utf-8 -*-
"""
config.py
=========
Central configuration for the Hybrid SWAT-ML-SHAP streamflow pipeline
(El Diem gauge station, Upper Blue Nile Basin).

Set DATA_DIR / OUTPUT_DIR to match your machine before running.
On the user's Windows/Spyder environment this should be:

    DATA_DIR   = "D:/Flow/Discharge_lag"
    OUTPUT_DIR = "D:/Flow/Discharge_lag/Outputs"

All other scripts import from this module so the paths and plotting
style only need to be changed in ONE place.
"""

import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ---------------------------------------------------------------------
# 1. PATHS  -- EDIT THESE TWO LINES FOR YOUR MACHINE
#    (env-var override only used for automated testing; on your machine
#    these simply resolve to the D:/Flow/Discharge_lag defaults below)
# ---------------------------------------------------------------------
DATA_DIR = os.environ.get("SWAT_ML_DATA_DIR", "D:/Flow/Discharge_lag")
OUTPUT_DIR = os.environ.get("SWAT_ML_OUTPUT_DIR", "D:/Flow/Discharge_lag/Outputs")

FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
TAB_DIR = os.path.join(OUTPUT_DIR, "tables")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")

for _d in (OUTPUT_DIR, FIG_DIR, TAB_DIR, MODEL_DIR):
    os.makedirs(_d, exist_ok=True)

# Input file names (must exist inside DATA_DIR)
F_CLIM = os.path.join(DATA_DIR, "ElDiem_Clim.csv")
F_ML = os.path.join(DATA_DIR, "ElDiem_ML.csv")
F_ML_ALL = os.path.join(DATA_DIR, "ElDiem_ML_ALL.csv")
F_ML_BASE = os.path.join(DATA_DIR, "ElDiem_ML_BASE.csv")
F_QOB_SWAT = os.path.join(DATA_DIR, "ElDiem_Qob_SWAT.csv")
F_QVAR = os.path.join(DATA_DIR, "ElDiem_Qvar.csv")

# ---------------------------------------------------------------------
# 2. TEMPORAL SPLITS  (monthly time step, 1986-01 to 2014-12)
# ---------------------------------------------------------------------
# SWAT stand-alone (physically-based model) calibration / validation
SWAT_CAL_START, SWAT_CAL_END = "1986-01", "2005-12"
SWAT_VAL_START, SWAT_VAL_END = "2006-01", "2014-12"

# Machine-learning train / validation / test (chronological, no shuffling).
# NOTE: master dataset effectively starts 1986-02 (first month dropped when
# engineering the P_lag1 antecedent-rainfall feature -- see data_prep.py).
# User-specified: train 1986-1997, validation 1998-2005, test 2005-2014.
# The stated validation/test periods both include 2005; to avoid overlap this
# is resolved as validation ending 2004-12 and test starting 2005-01.
ML_TRAIN_START, ML_TRAIN_END = "1986-01", "1997-12"
ML_VAL_START, ML_VAL_END = "1998-01", "2004-12"
ML_TEST_START, ML_TEST_END = "2005-01", "2014-12"

RANDOM_STATE = 42

# ---------------------------------------------------------------------
# 3. FEATURE GROUPS (progressive nested model design, M1-M5)
# ---------------------------------------------------------------------
# P_lag1 (1-month basin-lag rainfall / Antecedent Precipitation) represents
# basin routing time for this ~200,000 km^2 catchment: same-month P vs Q_obs
# r=0.755; 1-month-lag P vs Q_obs r=0.929. See data_prep.build_master().
MET_VARS = ["P", "Tmax", "Tmin", "RH", "SR", "WS", "P_lag1"]
LAG_VARS = ["Q_lag1", "Q_lag2", "Q_lag3"]
CLIM_VARS = ["ENSO", "IOD", "NAO"]

# ---------------------------------------------------------------------
# 4. PUBLICATION-QUALITY PLOT STYLE
#    Times New Roman, >=18 pt general text, 20 pt axis labels/titles,
#    >=2.5 pt line widths, 600 DPI raster export.
# ---------------------------------------------------------------------
def set_publication_style():
    """Apply a consistent Times New Roman / thick-line publication style."""

    # Try Times New Roman first (native on Windows/Office installs).
    # Fall back to a metric-compatible serif so the script never crashes
    # on machines where TNR isn't installed (e.g. Linux/macOS/CI).
    preferred = ["Times New Roman", "Liberation Serif", "Nimbus Roman", "serif"]
    available = {f.name for f in fm.fontManager.ttflist}
    font_family = next((f for f in preferred if f in available or f == "serif"), "serif")

    plt.rcParams.update({
        "font.family": font_family,
        "font.serif": [font_family],
        "mathtext.fontset": "custom",
        "mathtext.rm": font_family,
        "mathtext.it": font_family,
        "mathtext.bf": font_family,

        # Font sizes
        "font.size": 18,            # minimum general text
        "axes.titlesize": 20,       # titles
        "axes.labelsize": 20,       # axis labels
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "figure.titlesize": 20,

        # Line / marker weights (>= 2.5 pt)
        "lines.linewidth": 2.5,
        "lines.markersize": 8,
        "axes.linewidth": 2.5,
        "xtick.major.width": 2.5,
        "ytick.major.width": 2.5,
        "xtick.minor.width": 2.0,
        "ytick.minor.width": 2.0,
        "grid.linewidth": 1.2,
        "patch.linewidth": 2.5,
        "legend.edgecolor": "black",

        # Ticks visible/outward for clean publication look
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 7,
        "ytick.major.size": 7,

        # Export quality
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.grid": False,
        "axes.spines.top": True,
        "axes.spines.right": True,
    })
    return font_family


# Distinct, colorblind-safe, print-friendly color palette used throughout
COLORS = {
    "Q_obs": "#1a1a1a",       # near-black observed
    "Q_SWAT": "#1f77b4",      # blue
    "M1": "#8c564b",
    "M2": "#9467bd",
    "M3": "#2ca02c",
    "M4": "#ff7f0e",
    "M5": "#d62728",
    "ENSO": "#d62728",
    "IOD": "#1f77b4",
    "NAO": "#2ca02c",
    "cal": "#c7e9c0",
    "val": "#fdd0a2",
}

MODEL_LABELS = {
    "M1": "M1: Meteorology only",
    "M2": "M2: Meteorology + Q memory",
    "M3": "M3: Meteorology + Q memory + Teleconnections (Pure ML)",
    "M4": "M4: Hybrid SWAT-ML (Meteorology error correction)",
    "M5": "M5: Hybrid SWAT-ML (Full error correction)",
}
