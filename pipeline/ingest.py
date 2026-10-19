"""
Step 1: ingest.
Loads every source into the `raw` schema of the DuckDB warehouse, unchanged
except for one thing: World Bank data is stored in long format
(one row per country, year, and indicator), whatever its source.

Raw tables
  raw.world_bank      World Bank indicators (API or saved snapshot)
  raw.vdem            V-Dem release subset (coups, military rule, mobilization)
  raw.ecowas_events   hand-coded ECOWAS actions
  raw.countries       the 15 members, from config.yaml
  raw.borders         the border network, from config.yaml
Every load is recorded in meta.load_log.
"""

from datetime import datetime, timezone

import pandas as pd
import requests


# -----------------------------------------------------------------------------
# World Bank
# -----------------------------------------------------------------------------

def parse_world_bank_page(payload, variable):
    """
    Turn one page of a World Bank API response into a list of rows.

    The API returns a list with two parts: [page information, records].
    Example record:
      {"countryiso3code": "NER", "date": "2023", "value": 3372988000.0, ...}
    If the request was invalid, the API returns only a message, with no records.
    """
    if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
        raise ValueError(f"World Bank API returned no data: {payload}")
    rows = []
    for record in payload[1]:
        rows.append({
            "iso3": record["countryiso3code"],
            "year": int(record["date"]),
            "variable": variable,
            "value": record["value"],          # None when the value is missing
        })
    return rows


def fetch_world_bank(cfg, session=None):
    """
    Download every indicator in config.yaml from the World Bank API.
    Returns a long table: iso3, year, variable, value.
    """
    wb = cfg["sources"]["world_bank"]
    session = session or requests.Session()
    codes = ";".join(c["iso3"] for c in cfg["countries"])
    rows = []
    for indicator, variable in wb["indicators"].items():
        page, pages = 1, 1
        while page <= pages:                   # the API can split results into pages
            url = f"{wb['api_url']}/country/{codes}/indicator/{indicator}"
            params = {"format": "json", "date": f"{wb['first_year']}:{wb['last_year']}",
                      "per_page": 1000, "page": page}
            response = session.get(url, params=params, timeout=60)
            response.raise_for_status()
            payload = response.json()
            rows += parse_world_bank_page(payload, variable)
            pages = payload[0].get("pages", 1)
            page += 1
    long = pd.DataFrame(rows)
    return long.dropna(subset=["value"]).reset_index(drop=True)


def read_world_bank_snapshot(cfg):
    """
    Read the saved World Bank file (one column per indicator) and convert it to
    the same long format the API produces.
    """
    wide = pd.read_csv(cfg["sources"]["world_bank"]["snapshot"])
    variables = list(cfg["sources"]["world_bank"]["indicators"].values())
    long = wide.melt(id_vars=["iso3", "year"], value_vars=variables,
                     var_name="variable", value_name="value")
    return long.dropna(subset=["value"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Load everything into DuckDB
# -----------------------------------------------------------------------------

def _write(con, table, df, source):
    """Replace a raw table with a DataFrame and record the load."""
    con.register("incoming", df)
    con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM incoming")
    con.unregister("incoming")
    con.execute("INSERT INTO meta.load_log VALUES (?, ?, ?, ?)",
                [table, source, len(df), datetime.now(timezone.utc)])


def ingest(con, cfg, live=False):
    """Load all raw sources into the warehouse. live=True calls the World Bank API."""
    con.execute("CREATE SCHEMA IF NOT EXISTS raw")
    con.execute("CREATE SCHEMA IF NOT EXISTS meta")
    con.execute("""CREATE TABLE IF NOT EXISTS meta.load_log (
                       table_name VARCHAR, source VARCHAR, n_rows INTEGER,
                       loaded_at TIMESTAMPTZ)""")

    # World Bank: live API or the saved snapshot
    if live:
        wb = fetch_world_bank(cfg)
        wb.to_csv("data/raw/world_bank_live.csv", index=False)   # keep a copy of what came in
        source = cfg["sources"]["world_bank"]["api_url"]
    else:
        wb = read_world_bank_snapshot(cfg)
        source = cfg["sources"]["world_bank"]["snapshot"]
    _write(con, "raw.world_bank", wb, source)

    # V-Dem: the release subset (V-Dem has no live API)
    vdem_file = cfg["sources"]["vdem"]["file"]
    vdem = pd.read_csv(vdem_file)
    _write(con, "raw.vdem", vdem, f"{vdem_file} (V-Dem {cfg['sources']['vdem']['version']})")

    # Hand-coded ECOWAS actions
    events_file = cfg["sources"]["ecowas_events"]["file"]
    _write(con, "raw.ecowas_events", pd.read_csv(events_file), events_file)

    # Reference data from config.yaml
    _write(con, "raw.countries", pd.DataFrame(cfg["countries"]), "config.yaml")
    borders = pd.DataFrame([(a, b) for a, nbrs in cfg["borders"].items() for b in nbrs],
                           columns=["iso3", "neighbor"])
    _write(con, "raw.borders", borders, "config.yaml")
