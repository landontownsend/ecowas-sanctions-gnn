"""
Run the whole pipeline: ingest -> transform -> checks -> export.
Export writes two files: the study-period features and the forecast-year inputs.

From the repository root:
  python -m pipeline.run          # uses the saved World Bank snapshot (reproducible)
  python -m pipeline.run --live   # calls the World Bank API for the latest data
"""

import argparse
import os

import duckdb

from pipeline.checks import run_checks
from pipeline.config import load_config
from pipeline.ingest import ingest
from pipeline.transform import features_table, forecast_table, transform


def export(df, path):
    """Write a feature table to CSV for the model scripts, rounded to 3 decimals."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.round(3).to_csv(path, index=False)


def run(live=False, config_path="config.yaml"):
    cfg = load_config(config_path)
    db_path = cfg["paths"]["warehouse"]
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    with duckdb.connect(db_path) as con:
        print(f"1. Ingest ({'World Bank API' if live else 'saved World Bank snapshot'})")
        ingest(con, cfg, live=live)

        print("2. Transform (raw -> staging -> features)")
        transform(con, cfg)

        print("3. Data quality checks")
        run_checks(con, cfg)

        print("4. Export")
        df = features_table(con)
        export(df, cfg["paths"]["features_csv"])
        print(f"   Saved {len(df)} rows and {len(df.columns)} columns to "
              f"{cfg['paths']['features_csv']}")
        fc = forecast_table(con)
        export(fc, cfg["paths"]["forecast_csv"])
        print(f"   Saved {len(fc)} rows and {len(fc.columns)} columns to "
              f"{cfg['paths']['forecast_csv']}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECOWAS sanctions risk data pipeline")
    parser.add_argument("--live", action="store_true",
                        help="download World Bank data from the API instead of the snapshot")
    args = parser.parse_args()
    run(live=args.live)
