"""
Build data/raw/gdelt_country_month.csv from the GDELT 1.0 event files.

For each ECOWAS member and each month, the script counts:
  events_total     all events located in the country (ActionGeo_CountryCode)
  events_protest   CAMEO root code 14 (protest)
  events_violence  CAMEO root codes 18, 19, 20 (assault, fight, mass violence)
The settings are in config.yaml under sources.gdelt.

GDELT 1.0 publishes three kinds of files (all listed on the index page):
  YYYY.zip                  one file per year, up to 2005
  YYYYMM.zip                one file per month, 2006 to March 2013
  YYYYMMDD.export.CSV.zip   one file per day, from April 1, 2013
Every file is tab-separated with no header. The columns used here are in the
same place in all three kinds:
  column 1  SQLDATE (YYYYMMDD)
  column 28 EventRootCode ("01" to "20")
  column 51 ActionGeo_CountryCode (FIPS code, for example "NI" for Nigeria)

The full download is tens of gigabytes, so it runs in GitHub Actions
(.github/workflows/refresh-data.yml). Each file is downloaded, counted, and
deleted, several files at a time.

Usage, from the repository root:  python scripts/refresh_gdelt.py
"""

import csv
import io
import json
import re
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.config import load_config

CFG = load_config()
G = CFG["sources"]["gdelt"]
FIPS_TO_ISO = {c["fips"]: c["iso3"] for c in CFG["countries"]}
BASE_URL = G["index_url"].rsplit("/", 1)[0]          # http://data.gdeltproject.org/events
COLS = [1, 28, 51]                                    # SQLDATE, EventRootCode, ActionGeo_CountryCode
NAMES = ["date", "root", "fips"]


def list_files():
    """Read the index page and return the file names that cover first_year to last_year."""
    html = requests.get(G["index_url"], timeout=120).text
    # File names start with 4, 6, or 8 digits. The look-behind skips names such as
    # "GDELT.MASTERREDUCEDV2.1979-2013.zip", which is a different, reduced format.
    pattern = r"(?<![\w.\-])((?:\d{4}|\d{6}|\d{8}\.export\.CSV)\.zip)"
    names = sorted(set(re.findall(pattern, html, flags=re.IGNORECASE)))
    keep = []
    for name in names:
        stamp = name.split(".")[0]                    # "2005", "200601", or "20130401"
        year = int(stamp[:4])
        if G["first_year"] <= year <= G["last_year"]:
            keep.append(name)
    return keep


def download(name):
    """Download one file, with up to 4 tries. Returns the bytes."""
    for attempt in range(4):
        try:
            r = requests.get(f"{BASE_URL}/{name}", timeout=600)
            r.raise_for_status()
            return r.content
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(10 * (attempt + 1))


def count_file(name):
    """Count events by member, year, and month in one GDELT file."""
    data = download(name)
    counts = []
    rows_read = 0
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        inner = z.namelist()[0]
        with z.open(inner) as f:
            chunks = pd.read_csv(f, sep="\t", header=None, usecols=COLS, names=None,
                                 dtype=str, quoting=csv.QUOTE_NONE, chunksize=1_000_000,
                                 on_bad_lines="skip", encoding_errors="replace")
            for chunk in chunks:
                chunk.columns = NAMES
                rows_read += len(chunk)
                # Check that the columns are what we expect, so a format change cannot pass silently
                roots = chunk["root"].dropna()
                good = roots.str.fullmatch(r"\d{2}") & roots.astype(str).between("01", "20")
                if len(roots) > 1000 and good.mean() < 0.95:
                    raise ValueError(f"{name}: column 28 does not look like EventRootCode")
                chunk = chunk[chunk["fips"].isin(FIPS_TO_ISO)]
                chunk = chunk[chunk["date"].str.fullmatch(r"\d{8}", na=False)]
                chunk = chunk.assign(
                    iso3=chunk["fips"].map(FIPS_TO_ISO),
                    year=chunk["date"].str[:4].astype(int),
                    month=chunk["date"].str[4:6].astype(int),
                    protest=chunk["root"].isin(G["protest_roots"]).astype(int),
                    violence=chunk["root"].isin(G["violence_roots"]).astype(int),
                )
                counts.append(chunk.groupby(["iso3", "year", "month"])
                              .agg(events_total=("root", "size"),
                                   events_protest=("protest", "sum"),
                                   events_violence=("violence", "sum"))
                              .reset_index())
    out = pd.concat(counts) if counts else pd.DataFrame()
    return name, out, rows_read


def main():
    files = list_files()
    print(f"{len(files)} GDELT files cover {G['first_year']} to {G['last_year']}: "
          f"{files[0]} ... {files[-1]}", flush=True)
    if len(files) < 100:
        raise SystemExit("Too few files found on the index page; stopping.")

    parts, total_rows = [], 0
    with ProcessPoolExecutor(max_workers=6) as pool:
        for i, (name, counts, rows) in enumerate(pool.map(count_file, files), start=1):
            parts.append(counts)
            total_rows += rows
            if i % 200 == 0 or i == len(files):
                print(f"  {i} of {len(files)} files, {total_rows:,} rows read", flush=True)

    # Add up across files (a month can appear in several daily files)
    allc = pd.concat(parts).groupby(["iso3", "year", "month"]).sum().reset_index()
    allc = allc[allc.year.between(G["first_year"], G["last_year"])]

    # One row for every member and month, with 0 where GDELT has no events
    grid = pd.MultiIndex.from_product(
        [sorted(FIPS_TO_ISO.values()), range(G["first_year"], G["last_year"] + 1), range(1, 13)],
        names=["iso3", "year", "month"]).to_frame(index=False)
    out = grid.merge(allc, on=["iso3", "year", "month"], how="left").fillna(0)
    for col in ["events_total", "events_protest", "events_violence"]:
        out[col] = out[col].astype(int)

    out.to_csv(G["snapshot"], index=False)
    manifest = {"files": len(files), "first_file": files[0], "last_file": files[-1],
                "rows_read": total_rows, "member_events": int(out.events_total.sum()),
                "built": time.strftime("%Y-%m-%d")}
    Path(G["snapshot"]).with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
