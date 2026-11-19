"""Re-run the whole analysis from the raw files, then the tests.

    python run_all.py              # prepare -> scorecard -> models -> ... -> threshold + tests
    python run_all.py --download   # first download the raw files (1.7 GB Lending Club + FRED)
"""

import subprocess
import sys

STEPS = ["scripts/02_prepare.py", "scripts/03_scorecard.py", "scripts/04_models.py",
         "scripts/05_calibration.py", "scripts/06_shap.py", "scripts/07_stability_fairness.py",
         "scripts/08_ifrs9_ecl.py", "scripts/09_threshold.py"]

if __name__ == "__main__":
    if "--download" in sys.argv:
        STEPS.insert(0, "scripts/01_download_data.py")
    for step in STEPS:
        print(f"\n=== {step} ===", flush=True)
        subprocess.run([sys.executable, step], check=True)
    subprocess.run([sys.executable, "-m", "pytest", "-q"], check=True)
