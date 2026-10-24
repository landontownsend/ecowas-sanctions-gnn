"""
Rebuild the saved World Bank snapshot (data/raw/world_bank.csv) from the API.

The pipeline reads this snapshot by default, so results stay the same from run
to run. Run this script only to take a new copy of the data (a new "vintage").
The years come from config.yaml (sources.world_bank.first_year and last_year).

Usage, from the repository root:  python scripts/refresh_world_bank.py
GitHub Actions runs it in .github/workflows/refresh-data.yml.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # so "pipeline" can be imported

from pipeline.config import load_config
from pipeline.ingest import fetch_world_bank

cfg = load_config()
long = fetch_world_bank(cfg)                       # iso3, year, variable, value

# Same layout as the saved file: one row per country and year, one column per indicator
wide = long.pivot_table(index=["iso3", "year"], columns="variable", values="value").reset_index()
wide.columns.name = None
wide = wide[["iso3", "year", "gdp_growth", "imports_pct_gdp", "imports_real", "inflation"]]
wide = wide.sort_values(["iso3", "year"])

path = cfg["sources"]["world_bank"]["snapshot"]
wide.to_csv(path, index=False)
print(f"Saved {len(wide)} rows ({wide.year.min()} to {wide.year.max()}) to {path}")
