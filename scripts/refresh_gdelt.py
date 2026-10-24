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

Usage, from the repository root:
  python scripts/refresh_gdelt.py            full rebuild, first_year to last_year
  python scripts/refresh_gdelt.py --update   add only the complete months that are
                                             not in the snapshot yet (the monthly
                                             Airflow task; about 30 files a month)
"""

import csv
import io
import json
import re
import sys
import time
import traceback
import zipfile
from concurrent.futures import ProcessPoolExecutor
from datetime import date
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


def all_files():
    """Read the index page and return every event file name, sorted."""
    html = requests.get(G["index_url"], timeout=120).text
    # File names start with 4, 6, or 8 digits. The look-behind skips names such as
    # "GDELT.MASTERREDUCEDV2.1979-2013.zip", which is a different, reduced format.
    pattern = r"(?<![\w.\-])((?:\d{4}|\d{6}|\d{8}\.export\.CSV)\.zip)"
    return sorted(set(re.findall(pattern, html, flags=re.IGNORECASE)))


def list_files():
    """The file names that cover first_year to last_year."""
    keep = []
    for name in all_files():
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


def annotate(level, message):
    """Print a GitHub Actions annotation (readable through the API) and a normal line."""
    print(f"::{level}::{message}", flush=True)


def count_file(name):
    """Count one file. Returns (name, counts, rows read, error message or None)."""
    try:
        counts, rows = count_file_inner(name)
        return name, counts, rows, None
    except Exception as e:                      # record the problem; main() decides
        return name, pd.DataFrame(), 0, f"{type(e).__name__}: {e}"[:300]


def count_file_inner(name):
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
                    raise ValueError(f"column 28 does not look like EventRootCode "
                                     f"(only {good.mean():.0%} valid; first values {list(roots[:5])})")
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
    return out, rows_read


def count_files(files, workers=6):
    """
    Count every file, several at a time.
    Returns (counts by member, year, and month; rows read; list of skipped files).
    """
    parts, total_rows, errors = [], 0, []
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    results = pool.map(count_file, files) if pool else map(count_file, files)   # map: one at a time
    for i, (name, counts, rows, error) in enumerate(results, start=1):
        if error:
            errors.append(f"{name}: {error}")
            annotate("warning", f"skipped {name}: {error}")
        parts.append(counts)
        total_rows += rows
        if i % 500 == 0 or i == len(files):
            print(f"  {i} of {len(files)} files, {total_rows:,} rows read", flush=True)
    if pool:
        pool.shutdown()

    # A few bad files are recorded and skipped; more than 1% stops the run
    if len(errors) > max(5, len(files) // 100):
        annotate("error", f"{len(errors)} of {len(files)} files failed. First: {errors[:5]}")
        raise SystemExit(1)

    # Add up across files (a month can appear in several daily files)
    allc = pd.concat(parts).groupby(["iso3", "year", "month"]).sum().reset_index()
    return allc, total_rows, errors


def month_grid(months):
    """One row for every member and every (year, month) in months."""
    rows = [(iso, y, m) for iso in sorted(FIPS_TO_ISO.values()) for (y, m) in months]
    return pd.DataFrame(rows, columns=["iso3", "year", "month"])


def fill_grid(grid, counts):
    """Put the counts on the grid, with 0 where GDELT has no events."""
    out = grid.merge(counts, on=["iso3", "year", "month"], how="left").fillna(0)
    for col in ["events_total", "events_protest", "events_violence"]:
        out[col] = out[col].astype(int)
    return out


def main():
    """Full rebuild: every file from first_year to last_year."""
    files = list_files()
    annotate("notice", f"{len(files)} GDELT files cover {G['first_year']} to {G['last_year']}: "
                       f"{files[:3]} ... {files[-1]}")
    if len(files) < 100:
        raise SystemExit("Too few files found on the index page; stopping.")

    allc, total_rows, errors = count_files(files)
    allc = allc[allc.year.between(G["first_year"], G["last_year"])]

    months = [(y, m) for y in range(G["first_year"], G["last_year"] + 1) for m in range(1, 13)]
    out = fill_grid(month_grid(months), allc)

    out.to_csv(G["snapshot"], index=False)
    manifest = {"files": len(files), "first_file": files[0], "last_file": files[-1],
                "files_skipped": errors, "rows_read": total_rows,
                "member_events": int(out.events_total.sum()),
                "built": time.strftime("%Y-%m-%d")}
    annotate("notice", f"GDELT done: {len(files) - len(errors)} files, {total_rows:,} rows, "
                       f"{manifest['member_events']:,} events in members")
    Path(G["snapshot"]).with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def update(today=None, workers=6):
    """
    Add the complete months that are not in the snapshot yet.
    Example: the snapshot ends in December 2025 and today is November 2, 2026.
    The complete months are January to October 2026, so the daily files
    20260101 to 20261031 are counted and those 10 months are added.
    Events in those files that are dated before the new months are left out,
    because their months are already in the snapshot.
    Returns the number of months added.
    """
    today = today or date.today()
    path = Path(G["snapshot"])
    old = pd.read_csv(path)
    last = int((old.year * 100 + old.month).max())          # for example 202512

    # The complete months after the last one in the snapshot
    months = []
    y, m = last // 100, last % 100
    while True:
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
        if (y, m) >= (today.year, today.month):              # this month is not complete yet
            break
        months.append((y, m))
    if not months:
        print(f"GDELT snapshot is up to date (last month {last})")
        return 0

    wanted = {f"{y}{m:02d}" for (y, m) in months}
    files = [n for n in all_files() if n.endswith(".export.CSV.zip") and n[:6] in wanted]
    print(f"Adding {len(months)} months ({months[0]} to {months[-1]}) from {len(files)} daily files",
          flush=True)
    if not files:
        raise SystemExit("GDELT lists no daily files for the new months; stopping.")

    counts, total_rows, errors = count_files(files, workers=workers)
    keys = counts.year * 100 + counts.month
    counts = counts[keys.isin([y * 100 + m for (y, m) in months])]
    new = fill_grid(month_grid(months), counts)
    out = pd.concat([old, new], ignore_index=True).sort_values(["iso3", "year", "month"])
    out.to_csv(path, index=False)

    # Record the update next to the snapshot
    manifest_path = path.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.setdefault("updates", []).append(
        {"date": today.isoformat(), "months": [f"{y}-{m:02d}" for (y, m) in months],
         "files": len(files), "files_skipped": errors, "rows_read": total_rows})
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Added {len(months)} months; the snapshot now ends in {months[-1]}")
    return len(months)


if __name__ == "__main__":
    try:
        if "--update" in sys.argv:
            update()
        else:
            main()
    except SystemExit:
        raise
    except Exception:
        annotate("error", traceback.format_exc().strip().replace("\n", " | ")[-900:])
        raise
