import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Study period definitions (El Diem, Upper Blue Nile, 1986-2014)
# ---------------------------------------------------------------------
# SWAT stand-alone model: calibration / validation only (no separate test).
SWAT_CAL_END = "2005-12-31"   # Calibration: 1986-01 to 2005-12
SWAT_VAL_END = "2014-12-31"   # Validation:  2006-01 to 2014-12

# Machine-learning experiments: strict, time-ordered train/val/test split.
ML_TRAIN_END = "1997-12-31"   # Train:      1986-01 to 1997-12  (12 yr, 144 mo)
ML_VAL_END = "2005-12-31"     # Validation: 1998-01 to 2005-12  ( 8 yr,  96 mo)
# Test:       2006-01 to 2014-12  ( 9 yr, 108 mo)


def set_plot_style(base_size=16):
    """Consistent, publication-style figure formatting for ALL scripts:
    Times New Roman (falls back to Liberation/DejaVu Serif if TNR is not
    installed on the machine) both INSIDE the plot axes (titles, labels,
    ticks, legends, annotations, colorbars) and OUTSIDE the plot
    (figure-level captions/suptitles), larger fonts throughout for
    readability, and a higher DPI for crisp on-screen and print display.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",  # serif-matching math/greek symbols
        "font.size": base_size,
        "axes.titlesize": base_size + 4,
        "axes.labelsize": base_size + 2,
        "xtick.labelsize": base_size,
        "ytick.labelsize": base_size,
        "legend.fontsize": base_size,
        "legend.title_fontsize": base_size + 1,
        "figure.titlesize": base_size + 5,
        "axes.linewidth": 1.4,
        "grid.alpha": 0.3,
        "figure.dpi": 140,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
    })


def hydro_metrics(obs, sim):
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    nse = 1 - np.sum((obs - sim) ** 2) / np.sum((obs - obs.mean()) ** 2)
    rmse = np.sqrt(np.mean((obs - sim) ** 2))
    mae = np.mean(np.abs(obs - sim))
    pbias = 100 * np.sum(sim - obs) / np.sum(obs)
    r = np.corrcoef(obs, sim)[0, 1]
    r2 = r ** 2
    alpha = sim.std() / obs.std()
    beta = sim.mean() / obs.mean()
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return dict(NSE=nse, R2=r2, RMSE=rmse, MAE=mae, PBIAS=pbias, KGE=kge)


def time_split(df, date_col="Date", train_end=ML_TRAIN_END, val_end=ML_VAL_END):
    """Strict, non-shuffled, time-ordered split for the ML experiments.
    Default periods: Train 1986-2000, Validation 2001-2009, Test 2010-2014.
    """
    train = df[df[date_col] <= train_end].copy()
    val = df[(df[date_col] > train_end) & (df[date_col] <= val_end)].copy()
    test = df[df[date_col] > val_end].copy()
    return train, val, test


def cal_val_split(df, date_col="Date", cal_end=SWAT_CAL_END):
    """Calibration / validation split for the SWAT stand-alone model.
    Default periods: Calibration 1986-2005, Validation 2006-2014.
    """
    cal = df[df[date_col] <= cal_end].copy()
    val = df[df[date_col] > cal_end].copy()
    return cal, val


def shade_periods(ax, periods, y_frac=(0.0, 1.0), alpha=0.10, text_y_frac=0.96,
                   fontsize=None):
    """Shade vertical time-period bands on a datetime x-axis for visual
    clarity (e.g., Train / Validation / Test, or Calibration / Validation).

    periods: list of (label, start_date, end_date, color) tuples.
    """
    import pandas as pd
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    for label, start, end, color in periods:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color=color, alpha=alpha, lw=0, zorder=0)
        mid = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
        ax.text(mid, ymin + span * text_y_frac, label, ha="center", va="top",
                 fontsize=fontsize, color=color, fontweight="bold")
    ax.set_ylim(ymin, ymax)
