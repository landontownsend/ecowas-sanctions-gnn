"""Load config.yaml, the single settings file for the project."""

import os
import shutil
from pathlib import Path

import yaml


def load_config(path="config.yaml"):
    """
    Read config.yaml and return it as a Python dictionary.

    If the environment variable ECOWAS_DATA_DIR is set (the Airflow and
    dashboard containers set it), the files that change from run to run go in
    that folder instead of the repository:
      warehouse      ECOWAS_DATA_DIR/ecowas.duckdb
      GDELT snapshot ECOWAS_DATA_DIR/gdelt_country_month.csv (copied from the
                     repository the first time, then updated each month)
      dated runs     ECOWAS_DATA_DIR/runs
    Example: ECOWAS_DATA_DIR=/opt/airflow-data puts the warehouse at
    /opt/airflow-data/ecowas.duckdb.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["paths"].setdefault("runs", "runs")

    data_dir = os.environ.get("ECOWAS_DATA_DIR")
    if data_dir:
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        cfg["paths"]["warehouse"] = str(data_dir / "ecowas.duckdb")
        cfg["paths"]["runs"] = str(data_dir / "runs")
        gdelt = data_dir / "gdelt_country_month.csv"
        if not gdelt.exists():
            repo_copy = Path(cfg["sources"]["gdelt"]["snapshot"])
            shutil.copy(repo_copy, gdelt)
            if repo_copy.with_suffix(".json").exists():
                shutil.copy(repo_copy.with_suffix(".json"), gdelt.with_suffix(".json"))
        cfg["sources"]["gdelt"]["snapshot"] = str(gdelt)
    return cfg
