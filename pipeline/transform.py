"""
Step 2: transform.
Runs the SQL models in sql/ inside DuckDB: raw -> staging -> features.
"""

from pathlib import Path

SQL_DIR = Path("sql")
MODELS = ["staging.sql", "features.sql"]   # run in this order


def transform(con, cfg):
    """Run each SQL file. Year settings are filled in from config.yaml."""
    settings = {"start_year": cfg["years"]["start"], "end_year": cfg["years"]["end"]}
    for name in MODELS:
        sql = (SQL_DIR / name).read_text().format(**settings)
        con.execute(sql)


def features_table(con):
    """Return the finished feature table as a pandas DataFrame, sorted by country and year."""
    return con.execute("SELECT * FROM features.country_year ORDER BY iso3, year").df()
