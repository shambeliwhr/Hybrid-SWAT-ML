# -*- coding: utf-8 -*-
"""
run_all.py
==========
Runs the entire Hybrid SWAT-ML-SHAP pipeline end-to-end, in order.
Just run this file in Spyder (or `python run_all.py`) after setting
DATA_DIR in config.py to D:/Flow/Discharge_lag.
"""
import subprocess
import sys
import os

SCRIPTS = [
    "01_swat_standalone.py",
    "02_ml_models.py",
    "03_shap_analysis.py",
    "04_teleconnection_analysis.py",
    "05_summary.py",
    "06_wavelet_coherence.py",
]

here = os.path.dirname(os.path.abspath(__file__))

for script in SCRIPTS:
    print("\n" + "=" * 70)
    print(f"RUNNING {script}")
    print("=" * 70)
    result = subprocess.run([sys.executable, os.path.join(here, script)], cwd=here)
    if result.returncode != 0:
        print(f"\n*** {script} FAILED (exit code {result.returncode}) - stopping. ***")
        sys.exit(result.returncode)

print("\nAll scripts completed successfully. Check the OUTPUT_DIR (config.py) "
      "for figures/, tables/, and models/.")
