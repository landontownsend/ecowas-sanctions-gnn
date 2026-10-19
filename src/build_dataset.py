"""
build_dataset.py
ECOWAS sanctions risk: dataset build

Builds one row per ECOWAS member per year, 2010 to 2024 (15 x 15 = 225 rows).

FORECAST VARIABLES (does ECOWAS act, and how severely?)
  Outcome:    ECOWAS action and its severity          -> data/raw/ecowas_events.csv
  Predictors: coups, neighboring coups, mass mobilization, military rule
                                                       -> V-Dem (data/raw/vdem_ecowas_subset.csv)

ECONOMIC VARIABLES (how exposed is the country, and how much does it hurt?)
  Predictors: GDP growth, inflation, imports as a share of GDP   -> World Bank
  Outcome:    change in real imports around the action year      -> World Bank

How to run:
  pip install pandas requests
  python src/build_dataset.py   (from the repository root)

Output: data/processed/ecowas_country_year.csv
"""

import os
import requests
import pandas as pd

START_YEAR = 2010
END_YEAR = 2024
RAW = "data/raw"
OUT = "data/processed"


# =============================================================================
# SECTION 1: COUNTRY LIST
# vdem_name is how each country is spelled in the V-Dem file.
# Mali, Burkina Faso, and Niger are the three landlocked members.
# =============================================================================

COUNTRIES = pd.DataFrame([
    # iso3,  name,            vdem_name,       landlocked
    ["BEN", "Benin",          "Benin",          0],
    ["BFA", "Burkina Faso",   "Burkina Faso",   1],
    ["CPV", "Cabo Verde",     "Cape Verde",     0],
    ["CIV", "Cote d'Ivoire",  "Ivory Coast",    0],
    ["GMB", "Gambia",         "The Gambia",     0],
    ["GHA", "Ghana",          "Ghana",          0],
    ["GIN", "Guinea",         "Guinea",         0],
    ["GNB", "Guinea-Bissau",  "Guinea-Bissau",  0],
    ["LBR", "Liberia",        "Liberia",        0],
    ["MLI", "Mali",           "Mali",           1],
    ["NER", "Niger",          "Niger",          1],
    ["NGA", "Nigeria",        "Nigeria",        0],
    ["SEN", "Senegal",        "Senegal",        0],
    ["SLE", "Sierra Leone",   "Sierra Leone",   0],
    ["TGO", "Togo",           "Togo",           0],
], columns=["iso3", "country", "vdem_name", "landlocked"])


def build_skeleton():
    """Every country paired with every year: 15 x 15 = 225 rows."""
    # Start one year early so 2010 can have prior-year values; 2009 is dropped at the end
    years = pd.DataFrame({"year": range(START_YEAR - 1, END_YEAR + 1)})
    return COUNTRIES[["iso3", "country", "landlocked"]].merge(years, how="cross")


# =============================================================================
# SECTION 2: FORECAST OUTCOME (ECOWAS actions)
# Severity scale, using the strongest action taken that year:
#   0 = no new action
#   1 = suspension only
#   2 = sanctions (travel bans, asset freezes, or economic sanctions)
#   3 = border closure or cutoff of trade and financial flows
# =============================================================================

SEVERITY = {"suspension": 1, "sanctions": 2, "border_closure": 3}


def load_actions():
    events = pd.read_csv(os.path.join(RAW, "ecowas_events.csv"))
    events["year"] = pd.to_datetime(events["event_date"]).dt.year
    events["severity"] = events["event_type"].map(SEVERITY)

    # One yes/no column per action type
    events["suspension"] = (events["event_type"] == "suspension").astype(int)
    events["sanctions"] = (events["event_type"] == "sanctions").astype(int)
    events["border_closure"] = (events["event_type"] == "border_closure").astype(int)

    # Collapse to one row per country-year. max() keeps the strongest action.
    yearly = events.groupby(["iso3", "year"]).agg(
        action_severity=("severity", "max"),
        suspension=("suspension", "max"),
        sanctions=("sanctions", "max"),
        border_closure=("border_closure", "max"),
    ).reset_index()
    yearly["ecowas_action"] = 1
    return yearly


# =============================================================================
# SECTION 3: FORECAST PREDICTORS (V-Dem)
# e_pt_coup           successful coups (Powell and Thyne, via V-Dem)
# e_pt_coup_attempts  all attempts, successful and failed
# v2cagenmob          mass mobilization: how frequent AND how large protests,
#                     strikes, and demonstrations were (higher = more and bigger)
# v2x_ex_military     how much the head of government's power rests on the
#                     military, 0 to 1 (higher = more military control)
# =============================================================================

def load_vdem():
    vdem = pd.read_csv(os.path.join(RAW, "vdem_ecowas_subset.csv"))
    vdem = vdem.merge(COUNTRIES[["vdem_name", "iso3"]],
                      left_on="country_name", right_on="vdem_name")

    vdem["coups_successful"] = vdem["e_pt_coup"]
    # Attempts include successful ones, so subtract to get failed only
    vdem["coups_failed"] = vdem["e_pt_coup_attempts"] - vdem["e_pt_coup"]
    vdem["mass_mobilization"] = vdem["v2cagenmob"]
    vdem["military_rule_index"] = vdem["v2x_ex_military"]

    return vdem[["iso3", "year", "coups_successful", "coups_failed",
                 "mass_mobilization", "military_rule_index"]]


def add_neighbor_coups(panel, vdem):
    """
    Successful coups in OTHER ECOWAS members during the previous three years.
    Example: for Niger in 2023, this adds up successful coups in every member
    except Niger in 2020, 2021, and 2022.
    """
    values = []
    for _, row in panel.iterrows():
        window = vdem[
            (vdem["iso3"] != row["iso3"]) &
            (vdem["year"] >= row["year"] - 3) &
            (vdem["year"] <= row["year"] - 1)
        ]
        values.append(window["coups_successful"].sum())
    panel["neighbor_coups_prev3"] = values
    return panel


# =============================================================================
# SECTION 4: ECONOMIC VARIABLES (World Bank)
# Downloaded once and saved to data/raw/world_bank.csv.
# NE.IMP.GNFS.KD is real imports (constant dollars), used for the damage measure.
# =============================================================================

WB_INDICATORS = {
    "NY.GDP.MKTP.KD.ZG": "gdp_growth",       # real GDP growth, annual %
    "FP.CPI.TOTL.ZG":    "inflation",        # consumer price inflation, annual %
    "NE.IMP.GNFS.ZS":    "imports_pct_gdp",  # imports as a share of GDP
    "NE.IMP.GNFS.KD":    "imports_real",     # imports in constant US dollars
}


def load_world_bank():
    path = os.path.join(RAW, "world_bank.csv")
    if os.path.exists(path):
        return pd.read_csv(path)

    codes = ";".join(COUNTRIES["iso3"])
    rows = []
    for code, name in WB_INDICATORS.items():
        url = (f"https://api.worldbank.org/v2/country/{codes}/indicator/{code}"
               f"?format=json&date=2009:2025&per_page=2000")
        try:
            data = requests.get(url, timeout=60).json()[1]
        except Exception as e:
            print(f"WARNING: could not download {code} from the World Bank: {e}")
            continue
        for record in data:
            rows.append({"iso3": record["countryiso3code"], "year": int(record["date"]),
                         "variable": name, "value": record["value"]})

    if not rows:
        return None   # nothing downloaded; the fallback below handles inflation

    wide = pd.DataFrame(rows).pivot_table(index=["iso3", "year"], columns="variable",
                                          values="value").reset_index()
    wide.to_csv(path, index=False)
    return wide


def load_inflation_backup():
    """World Bank inflation from the datahub mirror (covers through 2023)."""
    infl = pd.read_csv(os.path.join(RAW, "worldbank_inflation_datahub.csv"))
    infl = infl.rename(columns={"Country Code": "iso3", "Year": "year", "Inflation": "inflation"})
    return infl[["iso3", "year", "inflation"]]


def add_import_change(panel, wb):
    """
    Import damage measure: percent change in real imports from the year BEFORE to
    the average of the current year and the year AFTER.
    Using two years after captures actions that happen late in a year.
    Example: Niger's closure began July 30, 2023, so this compares the average
    of 2023 and 2024 imports with 2022 imports.
    """
    imp = wb[["iso3", "year", "imports_real"]].dropna()
    lookup = {(r.iso3, r.year): r.imports_real for r in imp.itertuples()}

    changes = []
    for _, row in panel.iterrows():
        before = lookup.get((row["iso3"], row["year"] - 1))
        now = lookup.get((row["iso3"], row["year"]))
        after = lookup.get((row["iso3"], row["year"] + 1))
        if before and now and after:
            changes.append(((now + after) / 2 / before - 1) * 100)
        else:
            changes.append(None)
    panel["import_change_2yr"] = changes
    return panel


# =============================================================================
# SECTION 5: PUT IT ALL TOGETHER
# =============================================================================

def build():
    os.makedirs(OUT, exist_ok=True)
    panel = build_skeleton()

    # Forecast outcome. No row in the events file means no action (0).
    panel = panel.merge(load_actions(), on=["iso3", "year"], how="left")
    for col in ["ecowas_action", "action_severity", "suspension", "sanctions", "border_closure"]:
        panel[col] = panel[col].fillna(0).astype(int)

    # Forecast predictors
    vdem = load_vdem()
    panel = panel.merge(vdem, on=["iso3", "year"], how="left")
    panel = add_neighbor_coups(panel, vdem)

    # Economic variables
    wb = load_world_bank()
    if wb is not None:
        panel = panel.merge(wb[["iso3", "year", "gdp_growth", "inflation", "imports_pct_gdp"]],
                            on=["iso3", "year"], how="left")
        panel = add_import_change(panel, wb)
    else:
        print("World Bank not reachable: using backup inflation and leaving "
              "gdp_growth, imports_pct_gdp, and import_change_2yr empty.")
        panel = panel.merge(load_inflation_backup(), on=["iso3", "year"], how="left")
        panel["gdp_growth"] = None
        panel["imports_pct_gdp"] = None
        panel["import_change_2yr"] = None

    # Prior-year values, for the early-warning version of the model
    panel = panel.sort_values(["iso3", "year"]).reset_index(drop=True)
    lag_cols = ["coups_successful", "coups_failed", "mass_mobilization",
                "military_rule_index", "gdp_growth", "inflation", "imports_pct_gdp"]
    for col in lag_cols:
        panel[f"{col}_lag1"] = panel.groupby("iso3")[col].shift(1)

    # Drop the extra 2009 rows now that 2010 has its prior-year values
    panel = panel[panel["year"] >= START_YEAR].reset_index(drop=True)

    order = ["iso3", "country", "year", "landlocked",
             "ecowas_action", "action_severity", "suspension", "sanctions", "border_closure",
             "coups_successful", "coups_failed", "neighbor_coups_prev3",
             "mass_mobilization", "military_rule_index",
             "gdp_growth", "inflation", "imports_pct_gdp", "import_change_2yr"] + \
            [f"{c}_lag1" for c in lag_cols]
    panel = panel[order]
    panel = panel.round(3)

    out_path = os.path.join(OUT, "ecowas_country_year.csv")
    panel.to_csv(out_path, index=False)
    print(f"Saved {len(panel)} rows and {len(panel.columns)} columns to {out_path}")
    return panel


if __name__ == "__main__":
    build()
