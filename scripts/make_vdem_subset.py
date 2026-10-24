"""
Build data/raw/vdem_ecowas_subset.csv from the full V-Dem dataset.

Source: the official V-Dem release on GitHub (github.com/vdeminstitute/vdemdata),
file data/vdem.RData. The version must match sources.vdem.version in config.yaml.

Usage, from the repository root:
  git clone --depth 1 https://github.com/vdeminstitute/vdemdata /tmp/vdemdata
  pip install pyreadr
  python scripts/make_vdem_subset.py /tmp/vdemdata/data/vdem.RData
"""

import csv
import sys
from pathlib import Path

import pandas as pd
import pyreadr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.config import load_config

FIRST_YEAR = 1980      # early enough to count years since the last coup (capped at 20)
COLUMNS = ["country_name", "country_text_id", "year",
           "e_pt_coup", "e_pt_coup_attempts", "v2cagenmob", "v2x_ex_military"]

cfg = load_config()
members = [c["iso3"] for c in cfg["countries"]]

full = list(pyreadr.read_r(sys.argv[1]).values())[0]     # the .RData file holds one table
subset = full[full.country_text_id.isin(members) & (full.year >= FIRST_YEAR)][COLUMNS].copy()
subset["year"] = subset["year"].astype(int)
for col in ["e_pt_coup", "e_pt_coup_attempts"]:
    subset[col] = subset[col].astype("Int64")             # whole numbers; blank stays blank
subset = subset.sort_values(["country_text_id", "year"])

path = cfg["sources"]["vdem"]["file"]
subset.to_csv(path, index=False, quoting=csv.QUOTE_NONNUMERIC, na_rep="NA")
# pandas quotes the missing-value marker; write it unquoted, as V-Dem does
Path(path).write_text(Path(path).read_text().replace('"NA"', "NA"))
print(f"Saved {len(subset)} rows ({subset.year.min()} to {subset.year.max()}, "
      f"{subset.country_text_id.nunique()} countries) to {path}")
