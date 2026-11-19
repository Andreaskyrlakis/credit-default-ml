"""Download the Lending Club loan file (Hugging Face mirror, CC0, no account) and two FRED series.

    python scripts/01_download_data.py
The Lending Club file is 1.7 GB; it is not stored in git.
"""

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

config.RAW.mkdir(parents=True, exist_ok=True)
if not config.LC_FILE.exists():
    print("downloading Lending Club 2007-2018Q4 (1.7 GB) ...", flush=True)
    urllib.request.urlretrieve(config.LC_URL, config.LC_FILE)
for series in ["UNRATE", "CORCACBS"]:
    out = config.RAW / f"fred_{series.lower()}.csv"
    urllib.request.urlretrieve(config.FRED_URL.format(series), out)
    print("saved", out.name)
