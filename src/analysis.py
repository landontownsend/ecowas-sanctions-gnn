"""
analysis.py
ECOWAS sanctions risk: descriptive results, import damage, and baseline forecasts

Run from the repository root:  python src/analysis.py
Reads:  data/processed/ecowas_country_year.csv
        data/raw/vdem_ecowas_subset.csv
        data/raw/world_bank.csv
Writes: results/*.csv and results/summary.json

Sections
  1. Load data and border connections
  2. Descriptive patterns (actions over time, coups vs actions)
  3. Economic impact: import change in action years
  4. Forecast models (Stage 1, Baselines A to C): leave-one-year-out
  5. Out-of-sample check: score 2025 from 2024 conditions
  6. Current risk: score the next 12 months from 2025 conditions
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

DATA = "data"
OUT = "results"
os.makedirs(OUT, exist_ok=True)
summary = {}

# =============================================================================
# SECTION 1: DATA AND BORDER CONNECTIONS
# The border list is the network: each pair of ECOWAS members that share a land
# border is connected. Cabo Verde is an island and has no connections.
# =============================================================================

df = pd.read_csv(f"{DATA}/processed/ecowas_country_year.csv")

BORDERS = {
    "BEN": ["NGA", "NER", "BFA", "TGO"],
    "BFA": ["MLI", "NER", "BEN", "TGO", "GHA", "CIV"],
    "CPV": [],
    "CIV": ["LBR", "GIN", "MLI", "BFA", "GHA"],
    "GMB": ["SEN"],
    "GHA": ["CIV", "BFA", "TGO"],
    "GIN": ["GNB", "SEN", "MLI", "CIV", "LBR", "SLE"],
    "GNB": ["SEN", "GIN"],
    "LBR": ["SLE", "GIN", "CIV"],
    "MLI": ["SEN", "GIN", "CIV", "BFA", "NER"],
    "NER": ["MLI", "BFA", "BEN", "NGA"],
    "NGA": ["BEN", "NER"],
    "SEN": ["GMB", "GNB", "GIN", "MLI"],
    "SLE": ["GIN", "LBR"],
    "TGO": ["GHA", "BFA", "BEN"],
}
pd.DataFrame([(a, b) for a, nbrs in BORDERS.items() for b in nbrs],
             columns=["country", "neighbor"]).to_csv(f"{OUT}/borders.csv", index=False)

# Mali, Burkina Faso and Niger left ECOWAS on January 29, 2025
EXITED_2025 = ["MLI", "BFA", "NER"]


def add_network_features(panel):
    """
    Network features: what is happening in the countries a member borders.
      border_military_max_lag1: highest military rule score among border
                                neighbors last year
      border_coups_prev3:       successful coups in border neighbors over the
                                previous three years
    """
    mil = {(r.iso3, r.year): r.military_rule_index for r in panel.itertuples()}
    coup = {(r.iso3, r.year): r.coups_successful for r in panel.itertuples()}
    mil_max, coups3 = [], []
    for r in panel.itertuples():
        nbrs = BORDERS[r.iso3]
        vals = [mil.get((n, r.year - 1)) for n in nbrs]
        vals = [v for v in vals if v is not None and not np.isnan(v)]
        mil_max.append(max(vals) if vals else 0.0)
        coups3.append(sum(coup.get((n, y), 0) or 0
                          for n in nbrs for y in range(r.year - 3, r.year)))
    panel["border_military_max_lag1"] = mil_max
    panel["border_coups_prev3"] = coups3
    return panel


# Extend the panel with 2009 and 2025 V-Dem values so neighbor features and the
# 2025-2026 scores have the years they need.
vdem = pd.read_csv(f"{DATA}/raw/vdem_ecowas_subset.csv")
name_to_iso = {"Ivory Coast": "CIV", "Cape Verde": "CPV", "The Gambia": "GMB"}
vdem["iso3"] = vdem["country_text_id"]
vdem = vdem.rename(columns={"e_pt_coup": "coups_successful",
                            "v2x_ex_military": "military_rule_index",
                            "v2cagenmob": "mass_mobilization"})
# V-Dem leaves 2025 coup counts blank where no coup occurred, so blank means 0
vdem["coups_successful"] = vdem["coups_successful"].fillna(0)
wide = vdem[["iso3", "year", "coups_successful", "military_rule_index", "mass_mobilization"]]
wide = wide[wide.year.between(2006, 2025)].copy()
wide = add_network_features(wide)
df = df.merge(wide[["iso3", "year", "border_military_max_lag1", "border_coups_prev3"]],
              on=["iso3", "year"], how="left")

# =============================================================================
# SECTION 2: DESCRIPTIVE PATTERNS
# =============================================================================

acts = df[df.ecowas_action == 1]
summary["n_rows"] = len(df)
summary["n_action_years"] = int(len(acts))
summary["actions_2020_2023"] = int(acts.year.between(2020, 2023).sum())

by_year = df.groupby("year").agg(actions=("ecowas_action", "sum"),
                                 coups=("coups_successful", "sum"),
                                 max_severity=("action_severity", "max")).reset_index()
by_year.to_csv(f"{OUT}/by_year.csv", index=False)

coup_years = df[df.coups_successful > 0]
failed_only = df[(df.coups_failed > 0) & (df.coups_successful == 0)]
summary["successful_coup_years"] = int(len(coup_years))
summary["successful_coup_years_with_action"] = int(coup_years.ecowas_action.sum())
summary["failed_only_years"] = int(len(failed_only))
summary["failed_only_years_with_action"] = int(failed_only.ecowas_action.sum())
summary["actions_without_coup"] = int(((df.ecowas_action == 1) & (df.coups_successful == 0)).sum())

# =============================================================================
# SECTION 3: ECONOMIC IMPACT
# import_change_2yr compares real imports before an action with the average of
# the action year and the year after. To remove region-wide shocks (such as
# COVID in 2020), each value is compared with the median change among
# members with no action that same year ("excess" change).
# =============================================================================

base = df[df.ecowas_action == 0].groupby("year")["import_change_2yr"].median()
df["regional_median_import_change"] = df["year"].map(base)
df["excess_import_change"] = df["import_change_2yr"] - df["regional_median_import_change"]

impact = df[df.ecowas_action == 1][["iso3", "country", "year", "action_severity",
                                    "import_change_2yr", "regional_median_import_change",
                                    "excess_import_change", "imports_pct_gdp", "landlocked"]]
impact = impact.sort_values("excess_import_change")
impact.to_csv(f"{OUT}/impact.csv", index=False)

summary["impact_median_excess_all_actions"] = round(float(impact.excess_import_change.median()), 1)
summary["impact_median_excess_closures"] = round(float(
    impact[impact.action_severity == 3].excess_import_change.median()), 1)
summary["impact_niger_2023"] = round(float(
    impact[(impact.iso3 == "NER") & (impact.year == 2023)].excess_import_change.iloc[0]), 1)
summary["impact_actions_below_region"] = int((impact.excess_import_change < 0).sum())
summary["impact_corr_exposure"] = round(float(
    impact[["imports_pct_gdp", "excess_import_change"]].corr().iloc[0, 1]), 2)

gdp_act = df[df.ecowas_action == 1].gdp_growth.mean()
gdp_none = df[df.ecowas_action == 0].gdp_growth.mean()
summary["gdp_growth_action_years"] = round(float(gdp_act), 2)
summary["gdp_growth_other_years"] = round(float(gdp_none), 2)

# =============================================================================
# SECTION 4: FORECAST MODELS (STAGE 1, BASELINES)
# Outcome: ecowas_action (any action) in year t.
# Inputs: only information available before year t (prior-year values and the
# three-year coup counts).
# The model is a penalized logistic regression. The penalty (C=1) shrinks the
# coefficients, which keeps the model stable with only ten actions.
# Testing: leave one year out. The model is trained on 14 years and scores the
# 15 countries in the held-out year; repeated for every year.
# =============================================================================

OWN = ["military_rule_index_lag1", "coups_successful_lag1"]
REGIONAL = OWN + ["neighbor_coups_prev3"]
NETWORK = OWN + ["border_military_max_lag1", "border_coups_prev3"]
MODELS = [("Own conditions only", OWN, "own"),
          ("Own + region-wide coups", REGIONAL, "regional"),
          ("Own + border neighbors (network)", NETWORK, "network")]


def loyo_scores(data, features):
    """Leave-one-year-out predicted probabilities for every row."""
    preds = pd.Series(index=data.index, dtype=float)
    for yr in sorted(data.year.unique()):
        train, test = data[data.year != yr], data[data.year == yr]
        scaler = StandardScaler().fit(train[features])
        model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
        model.fit(scaler.transform(train[features]), train.ecowas_action)
        preds.loc[test.index] = model.predict_proba(scaler.transform(test[features]))[:, 1]
    return preds


def top3_hits(data, preds):
    """Share of actions where the country ranked in the top 3 of its year."""
    hits = 0
    for idx in data[data.ecowas_action == 1].index:
        yr = data.loc[idx, "year"]
        ranks = preds[data.year == yr].rank(ascending=False)
        hits += int(ranks.loc[idx] <= 3)
    return hits


model_df = df.dropna(subset=NETWORK).copy()
results = []
for name, feats, key in MODELS:
    p = loyo_scores(model_df, feats)
    model_df[f"pred_{key}"] = p
    results.append({"model": name,
                    "auc": round(roc_auc_score(model_df.ecowas_action, p), 3),
                    "top3_hits": top3_hits(model_df, p),
                    "actions": int(model_df.ecowas_action.sum())})
res = pd.DataFrame(results)
res.to_csv(f"{OUT}/model_comparison.csv", index=False)
summary["model_comparison"] = results

# Which actions each model caught (for the speaker notes)
model_df[model_df.ecowas_action == 1][["iso3", "year", "action_severity",
                                       "pred_own", "pred_regional", "pred_network"]].round(3) \
    .to_csv(f"{OUT}/action_predictions.csv", index=False)

# Coefficients from the full-sample network model (standardized, so comparable)
scaler = StandardScaler().fit(model_df[NETWORK])
full = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
full.fit(scaler.transform(model_df[NETWORK]), model_df.ecowas_action)
coefs = pd.DataFrame({"feature": NETWORK, "coef": full.coef_[0]})
coefs["odds_multiplier_per_sd"] = np.exp(coefs.coef)
coefs.round(3).to_csv(f"{OUT}/coefficients.csv", index=False)


def fit_full(data, feats):
    sc = StandardScaler().fit(data[feats])
    m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
    m.fit(sc.transform(data[feats]), data.ecowas_action)
    return sc, m

# =============================================================================
# SECTION 5: OUT-OF-SAMPLE CHECK
# Train on 2010-2024, then score 2025 using 2024 conditions. Compare with what
# happened in 2025: Guinea-Bissau's coup (Nov 26) and ECOWAS suspension (Nov 27);
# Benin's failed coup attempt (Dec 7), which drew no suspension or sanctions.
# =============================================================================

v = wide.set_index(["iso3", "year"])


def score_year(target_year, feats, sc, m):
    """Build inputs for target_year from the prior years' V-Dem values and score every member."""
    prev = target_year - 1
    def coup(iso, y):
        return v.loc[(iso, y), "coups_successful"] if (iso, y) in v.index else 0
    rows = []
    for iso in BORDERS:
        nbrs = BORDERS[iso]
        rows.append({
            "iso3": iso,
            "military_rule_index_lag1": v.loc[(iso, prev), "military_rule_index"],
            "coups_successful_lag1": coup(iso, prev),
            "neighbor_coups_prev3": sum(coup(o, y) for o in BORDERS if o != iso
                                        for y in range(target_year - 3, target_year)),
            "border_military_max_lag1": max([v.loc[(n, prev), "military_rule_index"] for n in nbrs], default=0.0),
            "border_coups_prev3": sum(coup(n, y) for n in nbrs
                                      for y in range(target_year - 3, target_year)),
        })
    s = pd.DataFrame(rows)
    s["risk"] = m.predict_proba(sc.transform(s[feats]))[:, 1]
    s["rank"] = s["risk"].rank(ascending=False, method="first").astype(int)
    return s.sort_values("rank")

# Add 2025 rows to the V-Dem table so 2025 network features exist
sc_n, m_n = fit_full(model_df, NETWORK)
oos = score_year(2025, NETWORK, sc_n, m_n)
oos.round(3).to_csv(f"{OUT}/oos_2025.csv", index=False)
summary["oos_2025_top5"] = oos.head(5)[["iso3", "risk", "rank"]].round(3).to_dict("records")
summary["oos_2025_gnb_rank"] = int(oos[oos.iso3 == "GNB"]["rank"].iloc[0])
summary["oos_2025_ben_rank"] = int(oos[oos.iso3 == "BEN"]["rank"].iloc[0])

# =============================================================================
# SECTION 6: CURRENT RISK (NEXT 12 MONTHS)
# Train on all years through 2024 and score using 2025 conditions.
# Mali, Burkina Faso and Niger are scored but flagged, since they are no
# longer members and cannot be suspended by ECOWAS.
# =============================================================================

wide26 = wide.copy()
now = score_year(2026, NETWORK, sc_n, m_n)
now["status"] = np.where(now.iso3.isin(EXITED_2025), "Left ECOWAS (2025)",
                         np.where(now.iso3 == "GNB", "Suspended (Nov 2025)", "Member"))
members = now[now.status != "Left ECOWAS (2025)"].copy()
members["member_rank"] = members["risk"].rank(ascending=False, method="first").astype(int)
now = now.merge(members[["iso3", "member_rank"]], on="iso3", how="left")
# Risk tiers among current members
now["tier"] = pd.cut(now["risk"], bins=[-1, 0.25, 0.5, 2], labels=["Lower", "Elevated", "High"])
now.round(3).to_csv(f"{OUT}/risk_next12.csv", index=False)
summary["risk_next12"] = now[["iso3", "risk", "member_rank", "status", "tier"]] \
    .round(3).astype(str).to_dict("records")

with open(f"{OUT}/summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)
print(json.dumps(summary, indent=2, default=str))

# =============================================================================
# SECTION 7: EXTRA CHECKS FOR THE PRESENTATION
# =============================================================================

# How the network model scored the four junta-led members in held-out 2024,
# when ECOWAS took no new action (it lifted Niger, Mali and Burkina Faso
# sanctions in February 2024). This shows the policy shift after 2023.
fp24 = model_df[(model_df.year == 2024) & (model_df.iso3.isin(["MLI", "BFA", "NER", "GIN"]))]
summary["held_out_2024_junta_scores"] = fp24[["iso3", "pred_network"]].round(3).to_dict("records")

# Damage by geography (small groups, descriptive only)
summary["impact_median_excess_landlocked"] = round(float(impact[impact.landlocked == 1].excess_import_change.median()), 1)
summary["impact_median_excess_coastal"] = round(float(impact[impact.landlocked == 0].excess_import_change.median()), 1)

# Risk tiers for current members (relative scores from the network model)
def tier(r):
    return "High" if r >= 0.85 else ("Elevated" if r >= 0.70 else "Lower")
now["tier"] = [("Left ECOWAS" if st == "Left ECOWAS (2025)" else tier(r))
               for r, st in zip(now.risk, now.status)]
context = {
    "GNB": "Military council since Nov 2025; suspended; elections promised by Dec 6, 2026",
    "GIN": "Readmitted Jan 2026 after Doumbouya's election; model does not yet see this",
    "BEN": "Failed coup attempt Dec 7, 2025; borders Niger and Burkina Faso",
}
now["context"] = now.iso3.map(context).fillna("")
now["risk_score_0_100"] = (now.risk * 100).round(0).astype(int)
now.to_csv(f"{OUT}/risk_next12.csv", index=False)

with open(f"{OUT}/summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)
