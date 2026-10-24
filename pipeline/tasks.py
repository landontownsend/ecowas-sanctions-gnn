"""
One function per Airflow task (dags/ecowas_monthly.py and dags/ecowas_models.py).

The functions are plain Python, so they can be tested without Airflow. Each one
opens its own DuckDB connection, because Airflow runs every task in a separate
process. The tasks run one after another, so only one task writes to the
warehouse at a time.

Every function works from the repository root, whatever folder Airflow starts in.
"""

import json
import os
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]      # the repository root


def _setup():
    """Move to the repository root and load the settings."""
    os.chdir(ROOT)
    from pipeline.config import load_config
    return load_config()


def _connect(cfg):
    """Open the warehouse (create its folder the first time)."""
    path = cfg["paths"]["warehouse"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return duckdb.connect(path)


def _run(script, *args):
    """Run one of the project's scripts with the same Python, and stop on an error."""
    subprocess.run([sys.executable, script, *args], cwd=ROOT, check=True)


# -----------------------------------------------------------------------------
# Monthly pipeline
# -----------------------------------------------------------------------------

def update_gdelt():
    """Add the complete months that are missing from the GDELT snapshot."""
    _setup()
    _run("scripts/refresh_gdelt.py", "--update")


def ingest(live=True):
    """Step 1: load every source into the raw tables (live World Bank data by default)."""
    from pipeline.ingest import ingest as ingest_all
    cfg = _setup()
    with _connect(cfg) as con:
        ingest_all(con, cfg, live=live)


def transform():
    """Step 2: run the SQL models (raw -> staging -> features)."""
    from pipeline.transform import transform as run_sql
    cfg = _setup()
    with _connect(cfg) as con:
        run_sql(con, cfg)


def checks():
    """Step 3: the data quality checks. A failure stops the DAG here."""
    from pipeline.checks import run_checks
    cfg = _setup()
    with _connect(cfg) as con:
        run_checks(con, cfg)


def export():
    """Step 4: write the feature tables to data/processed/ for the model scripts."""
    from pipeline.run import export as write_csv
    from pipeline.transform import features_table, forecast_table
    cfg = _setup()
    with _connect(cfg) as con:
        write_csv(features_table(con), cfg["paths"]["features_csv"])
        write_csv(forecast_table(con), cfg["paths"]["forecast_csv"])


def baselines():
    """Baselines A to C, descriptive results, and the economic impact measure."""
    _setup()
    _run("src/analysis.py")


def watch_list():
    """Refit the main model (Baseline C v2) and rewrite the watch list."""
    _setup()
    _run("src/gnn.py", "--watch-list")


def publish(run_date=None):
    """
    Save a dated copy of the watch lists and the summary, and write a short note
    on what changed since the last saved run.
    Example: run_date "2026-11-02" writes runs/2026-11-02/ with
    risk_next12_v2.csv, risk_next12.csv, summary.json, and changes.md.
    Returns the folder it wrote.
    """
    cfg = _setup()
    run_date = run_date or date.today().isoformat()
    runs = Path(cfg["paths"]["runs"])
    folder = runs / run_date
    folder.mkdir(parents=True, exist_ok=True)

    results = Path(cfg["paths"]["results"])
    for name in ["risk_next12_v2.csv", "risk_next12.csv", "summary.json"]:
        if (results / name).exists():
            shutil.copy(results / name, folder / name)

    # Compare with the newest earlier run, if there is one
    earlier = sorted(p for p in runs.iterdir() if p.is_dir() and p.name < run_date)
    if earlier:
        old = pd.read_csv(earlier[-1] / "risk_next12_v2.csv")
        new = pd.read_csv(folder / "risk_next12_v2.csv")
        note = compare_watch_lists(old, new, earlier[-1].name, run_date)
    else:
        note = f"# Watch list, {run_date}\n\nFirst saved run; nothing to compare with.\n"
    (folder / "changes.md").write_text(note)
    print(note)
    return str(folder)


def compare_watch_lists(old, new, old_date, new_date):
    """
    A short Markdown note on how the watch list changed.
    Example line: "- GIN: score 74 -> 79, tier Elevated -> Elevated, member rank 2 -> 2"
    """
    lines = [f"# Watch list, {new_date}", "", f"Compared with {old_date}.", ""]
    both = old.merge(new, on="iso3", suffixes=("_old", "_new"))
    changed = both[(both.risk_score_0_100_old != both.risk_score_0_100_new)
                   | (both.tier_old != both.tier_new)
                   | (both.member_rank_old.fillna(0) != both.member_rank_new.fillna(0))]
    if changed.empty:
        lines.append("No changes in scores, tiers, or ranks.")
    for r in changed.itertuples():
        rank_old = "-" if pd.isna(r.member_rank_old) else int(r.member_rank_old)
        rank_new = "-" if pd.isna(r.member_rank_new) else int(r.member_rank_new)
        lines.append(f"- {r.iso3}: score {r.risk_score_0_100_old} -> {r.risk_score_0_100_new}, "
                     f"tier {r.tier_old} -> {r.tier_new}, member rank {rank_old} -> {rank_new}")
    return "\n".join(lines) + "\n"


# -----------------------------------------------------------------------------
# Full model run (manual)
# -----------------------------------------------------------------------------

def gnn_full():
    """Baseline C v2 and GNN v2 with nested tuning (20 to 40 minutes on 4 cores)."""
    _setup()
    _run("src/gnn.py")


def figures():
    """Figures 1 to 4."""
    _setup()
    _run("src/charts.py")
    _run("src/gnn_chart.py")


def publish_models(run_date=None):
    """Save a dated copy of the full model results in runs/models-YYYY-MM-DD/."""
    cfg = _setup()
    run_date = run_date or date.today().isoformat()
    folder = Path(cfg["paths"]["runs"]) / f"models-{run_date}"
    folder.mkdir(parents=True, exist_ok=True)
    results = Path(cfg["paths"]["results"])
    for name in ["gnn_summary.json", "gnn_loyo_predictions.csv", "gnn_tuning.csv",
                 "risk_next12_v2.csv"]:
        shutil.copy(results / name, folder / name)
    decision = json.load(open(results / "gnn_summary.json"))["decision"]["main_model"]
    print(f"Saved the model run to {folder}; main model: {decision}")
    return str(folder)
