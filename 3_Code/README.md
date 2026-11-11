# Hybrid SWAT-ML-SHAP Streamflow Pipeline — El Diem, Upper Blue Nile Basin

This package implements the full analysis described in `Procedure.docx`:
SWAT stand-alone evaluation, a progressive nested ML / hybrid SWAT-ML
error-correction model comparison (M1–M5), SHAP explainability, and
ENSO/IOD/NAO teleconnection attribution — all with Times New Roman,
≥18 pt text, 20 pt axis labels, and ≥2.5 pt line widths throughout.

## 1. Setup

1. Copy this folder anywhere on your machine (e.g. `D:/Flow/Discharge_lag/scripts`).
2. Make sure the six input CSVs are in **`D:/Flow/Discharge_lag`**:
   `ElDiem_Clim.csv`, `ElDiem_ML.csv`, `ElDiem_ML_ALL.csv`,
   `ElDiem_ML_BASE.csv`, `ElDiem_Qob_SWAT.csv`, `ElDiem_Qvar.csv`.
3. Open `config.py` and confirm/edit:
   ```python
   DATA_DIR = "D:/Flow/Discharge_lag"
   OUTPUT_DIR = "D:/Flow/Discharge_lag/Outputs"
   ```
4. In Spyder, make sure the working directory is this script folder
   (or add it to the Python path) so the modules can `import config`.

Required packages (install once): `pip install xgboost shap scikit-learn
pandas numpy matplotlib scipy pycwt`. All other listed dependencies
(pandas, numpy, matplotlib, scipy, scikit-learn) are typically already
present in an Anaconda/Spyder install; `xgboost`, `shap`, and `pycwt`
usually are not and must be installed with pip as shown above.

## 2. Run

Either run `run_all.py` once (runs all 5 stages in order), or run the
numbered scripts individually in Spyder's IPython console:

```
01_swat_standalone.py        SWAT calibration (1986-2005) / validation (2006-2014)
02_ml_models.py               M1-M5 nested ML / hybrid model training + comparison
03_shap_analysis.py           SHAP explainability for M3 (pure ML) and M5 (hybrid)
04_teleconnection_analysis.py ENSO/IOD/NAO correlation + composite analysis
05_summary.py                 Final summary table + graphical-abstract figure
06_wavelet_coherence.py       Wavelet coherence: Rainfall-ENSO, Discharge-IOD, Discharge-ENSO
```

## 3. Model design (M1–M5)

| Model | Features | Target | Type |
|---|---|---|---|
| M1 | Meteorology (P, Tmax, Tmin, RH, SR, WS, P_lag1) | Q_obs | Pure ML |
| M2 | M1 + discharge memory (Q_lag1‑3) | Q_obs | Pure ML |
| M3 | M2 + teleconnections (ENSO, IOD, NAO) | Q_obs | Pure ML (full) |
| M4 | Meteorology only (incl. P_lag1) | Q_obs − Q_SWAT (error) | Hybrid SWAT-ML |
| M5 | Meteorology + lags + teleconnections | Q_obs − Q_SWAT (error) | Hybrid SWAT-ML (full) |

`P_lag1` (1-month basin-lag / antecedent rainfall) is included in the
meteorology group to represent basin routing time for this
~200,000 km² catchment — see Section 6 for the physical justification.


For M4/M5, the final discharge prediction is reconstructed as
`Q_hybrid = Q_SWAT + predicted_error`, and **all** models (M1–M5,
including hybrids) are scored in discharge space against `Q_obs` so
NSE/KGE/etc. are directly comparable across the whole table.

Hyperparameters are grid-searched on TRAIN → scored on VALIDATION
(strict chronological split, no shuffling), then the winning
configuration is refit on TRAIN+VALIDATION before scoring on the
untouched TEST period (2005–2014).

## 4. Data splits

| Purpose | Period | n (months) |
|---|---|---|
| SWAT calibration | 1986-01 – 2005-12 | 240 |
| SWAT validation | 2006-01 – 2014-12 | 108 |
| ML train | 1986-02 – 1997-12 | 143 |
| ML validation | 1998-01 – 2004-12 | 84 |
| ML test | 2005-01 – 2014-12 | 120 |

(User-specified periods were train 1986–1997, validation 1998–2005,
test 2005–2014; since validation and test both named 2005, this is
resolved as validation ending 2004-12 and test starting 2005-01 to
avoid overlap.)

(ML data starts one month later than SWAT calibration because the
first month, 1986-01, has no valid antecedent-rainfall value and is
dropped when engineering `P_lag1` — see Section 6.)

## 5. Outputs (written to `OUTPUT_DIR`)

- `tables/` — all metric tables, SHAP importances, teleconnection
  correlations, and `master_dataset_with_predictions.csv` (full
  348-month dataset with every model's predictions attached).
- `figures/` — all PNGs at 600 DPI, Times New Roman, ≥2.5 pt lines.
- `models/` — saved XGBoost models (`M1.json`…`M5.json`) reusable
  without retraining (e.g. for further SHAP dependence plots).

## 6. Why rainfall is represented as a 1-month basin-lag feature (P_lag1)

Precipitation is the physical driver of runoff — this is not in
dispute. What a *same-month* correlation coefficient measures is
something narrower: how well this month's rainfall total lines up
with this month's discharge reading at the outlet. For a small,
fast-responding catchment those can coincide closely. For a
~200,000 km² basin like the Upper Blue Nile, they generally do not,
because rainfall has to infiltrate, move through interflow and
groundwater storage, and route through the channel network before it
is measured as discharge at El Diem — a delay hydrologists call
**basin lag time / time of concentration**, and the reason the
**Antecedent Precipitation Index** (Kohler & Linsley, 1951) has been
standard practice in rainfall-runoff modelling for over 70 years.

The data confirms this directly:

| | correlation with Q_obs |
|---|---|
| Same-month rainfall (P) | r = 0.755 |
| **1-month-lagged rainfall (P_lag1)** | **r = 0.929** |

An earlier version of this pipeline used only same-month P, which
denied the model the correctly-timed version of the one variable that
physically causes runoff — the model then leaned on relative humidity
and temperature (same-month-persistent variables, r≈0.79 with Q_obs
either way) as imperfect substitutes. An ablation test confirmed this:
removing RH from the feature set does *not* hand its importance to P
— it goes to Tmax (RH's near-mirror, r=‑0.935) instead, showing that
same-month P genuinely lacked the signal to compete, not that the
model was ignoring rainfall out of some artifact.

With `P_lag1` included, lagged rainfall becomes the clear #1 SHAP
driver in M3 (~55% of attribution) — consistent with rainfall's known
physical role once its causal delay is represented correctly. This
is the version delivered in this package.



## 7. Wavelet coherence (Fig. 12-14)

As specified in the procedure document ("Very Q1-friendly... commonly
published in Journal of Hydrology"), `06_wavelet_coherence.py` computes
Continuous Wavelet Transform-based Wavelet Coherence (Grinsted et al.,
2004, via the `pycwt` package) for:

- **Fig. 12** — Rainfall (P) ↔ ENSO
- **Fig. 13** — Discharge (Q) ↔ IOD
- **Fig. 14** — Discharge (Q) ↔ ENSO

Each plot shows the coherence power surface (period in years on a log2
y-axis vs. time on the x-axis), a black contour marking 90% significance
against a red-noise (AR1) background, the cone of influence (hatched
region subject to edge effects), and phase arrows indicating lead/lag
relationships (arrows pointing right = in phase; left = anti-phase;
down = X leads Y by 90°; up = Y leads X by 90°). The x-axis shows every
calendar year, rotated 90°, as requested. Band-averaged coherence
(annual, interannual, and the 2–7 year ENSO band) is summarized in
`tables/06_wavelet_coherence_summary.csv`.

Note: wavelet coherence significance is computed via Monte Carlo (150
AR1 surrogate realizations per pair) and can take several minutes to
run the first time for each unique pair; `pycwt` caches significance
levels internally so repeat runs are faster.

## 8. Other notes for your manuscript

- **Q_obs consistency check**: `data_prep.py` cross-validates `Q_obs`
  across all source files on load and prints any mismatch. A ~0.0002
  m³/s floating-point rounding difference exists between
  `ElDiem_Qvar.csv` and the other files — negligible, but shown so you
  can verify against your original master spreadsheet.
- **M2/M3 train & validation NSE ≈ 1.00**: this reflects the very
  high month-to-month autocorrelation of discharge (Q_lag1 alone is
  an extremely strong predictor), not a coding error. Test-period NSE
  is the honest out-of-sample number and should be the one
  reported/emphasized in the paper; consider noting the train–test gap
  as an autocorrelation-driven optimism gap in your Discussion.
- **Teleconnection correlations use deseasonalized anomalies**
  (monthly value minus its 1986–2014 monthly climatology), not raw
  discharge/rainfall, because the strong Kiremt seasonal cycle
  otherwise swamps any interannual ENSO/IOD signal. This follows
  standard climate-teleconnection practice and is what should be
  cited/reported, not a raw-series correlation.
