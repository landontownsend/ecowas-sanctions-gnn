"""
Step 2: transform.
Runs the SQL models in sql/ inside DuckDB: raw -> staging -> features.
"""

from pathlib import Path

SQL_DIR = Path("sql")
MODELS = ["staging.sql", "features.sql"]   # run in this order


def transform(con, cfg):
    """Run each SQL file. Year settings are filled in from config.yaml."""
    settings = {"start_year": cfg["years"]["start"],
                "end_year": cfg["years"]["end"],
                "forecast_year": cfg["years"]["forecast"],
                "gdelt_min_events": cfg["sources"]["gdelt"]["min_events"]}
    for name in MODELS:
        sql = (SQL_DIR / name).read_text().format(**settings)
        con.execute(sql)


def features_table(con):
    """Return the study-period feature table as a pandas DataFrame, sorted by country and year."""
    return con.execute("SELECT * FROM features.country_year ORDER BY iso3, year").df()


def forecast_table(con):
    """Return the model inputs for the years after the study period, sorted by country and year."""
    return con.execute("SELECT * FROM features.forecast_inputs ORDER BY iso3, year").df()
