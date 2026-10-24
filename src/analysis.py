"""
analysis.py
ECOWAS sanctions risk: descriptive results, import damage, and baseline forecasts

Run from the repository root:  python src/analysis.py
Reads:  data/processed/ecowas_country_year.csv   (study period, from the pipeline)
        data/processed/ecowas_forecast_inputs.csv (years after it, from the pipeline)
Writes: results/*.csv and results/summary.json
Years and watch list settings come from config.yaml.

Sections
  1. Load data and border connections
  2. Descriptive patterns (actions over time, coups vs actions)
  3. Economic impact: import change in action years
  4. Forecast models (Stage 1, Baselines A to C): leave-one-year-out
  5. Out-of-sample check: score the check year (years.check in config.yaml)
  6. Current risk: score the forecast year (years.forecast), the watch list
"""

import json
import os
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from watchlist import build_watch_list   # src/watchlist.py

# Every setting comes from config.yaml in the repository root
CFG = yaml.safe_load(open("config.yaml"))
BASE = CFG["models"]["baseline"]
CHECK = CFG["years"]["check"]          # 2025: the out-of-sample check year
FORECAST = CFG["years"]["forecast"]    # 2026: the watch list year

OUT = CFG["paths"]["results"]
os.makedirs(OUT, exist_ok=True)
summary = {}

# =============================================================================
# SECTION 1: DATA AND BORDER CONNECTIONS
# The border list is the network: each pair of ECOWAS members that share a land
# border is connected. Cabo Verde is an island and has no connections.
# The pipeline (sql/features.sql) has already built every model input,
# including the border features, so this script only reads them.
# =============================================================================

df = pd.read_csv(CFG["paths"]["features_csv"])          # study period, one row per member per year
forecast = pd.read_csv(CFG["paths"]["forecast_csv"])    # years after the study period, inputs only

BORDERS = CFG["borders"]   # the border network, from config.yaml
pd.DataFrame([(a, b) for a, nbrs in BORDERS.items() for b in nbrs],
             columns=["country", "neighbor"]).to_csv(f"{OUT}/borders.csv", index=False)

# Mali, Burkina Faso and Niger left ECOWAS on January 29, 2025
EXITED_2025 = CFG["exited_2025"]

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
        model = LogisticRegression(C=BASE["C"], class_weight=BASE["class_weight"], max_iter=BASE["max_iter"])
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
full = LogisticRegression(C=BASE["C"], class_weight=BASE["class_weight"], max_iter=BASE["max_iter"])
full.fit(scaler.transform(model_df[NETWORK]), model_df.ecowas_action)
coefs = pd.DataFrame({"feature": NETWORK, "coef": full.coef_[0]})
coefs["odds_multiplier_per_sd"] = np.exp(coefs.coef)
coefs.round(3).to_csv(f"{OUT}/coefficients.csv", index=False)


def fit_full(data, feats):
    sc = StandardScaler().fit(data[feats])
    m = LogisticRegression(C=BASE["C"], class_weight=BASE["class_weight"], max_iter=BASE["max_iter"])
    m.fit(sc.transform(data[feats]), data.ecowas_action)
    return sc, m

# =============================================================================
# SECTION 5: OUT-OF-SAMPLE CHECK
# Train on the study period, then score the check year (2025) using the year
# before (2024). Compare with what happened in 2025: Guinea-Bissau's coup
# (Nov 26) and ECOWAS suspension (Nov 27); Benin's failed coup attempt (Dec 7),
# which drew no suspension or sanctions.
# =============================================================================


# The inputs of Baselines A to C, saved next to each score
INPUTS = OWN + ["neighbor_coups_prev3", "border_military_max_lag1", "border_coups_prev3"]


def score_year(year, feats, sc, m):
    """Score every member in one forecast year, using the inputs the pipeline built."""
    s = forecast.loc[forecast.year == year, ["iso3"] + INPUTS].copy()
    s["risk"] = m.predict_proba(sc.transform(s[feats]))[:, 1]
    s["rank"] = s["risk"].rank(ascending=False, method="first").astype(int)
    return s.sort_values("rank")


sc_n, m_n = fit_full(model_df, NETWORK)
oos = score_year(CHECK, NETWORK, sc_n, m_n)
oos.round(3).to_csv(f"{OUT}/oos_{CHECK}.csv", index=False)
summary[f"oos_{CHECK}_top5"] = oos.head(5)[["iso3", "risk", "rank"]].round(3).to_dict("records")
summary[f"oos_{CHECK}_gnb_rank"] = int(oos[oos.iso3 == "GNB"]["rank"].iloc[0])
summary[f"oos_{CHECK}_ben_rank"] = int(oos[oos.iso3 == "BEN"]["rank"].iloc[0])

# =============================================================================
# SECTION 6: CURRENT RISK (THE WATCH LIST)
# Train on the study period and score the forecast year (2026) using the year
# before (2025). Mali, Burkina Faso and Niger are scored but flagged, since
# they are no longer members and cannot be suspended by ECOWAS.
# The tiers, suspended members, and context notes come from config.yaml
# (see src/watchlist.py).
# =============================================================================

now = build_watch_list(score_year(FORECAST, NETWORK, sc_n, m_n), CFG)
now.to_csv(f"{OUT}/risk_next12.csv", index=False)
summary["risk_next12"] = now[["iso3", "risk", "member_rank", "status", "tier"]] \
    .round(3).astype(str).to_dict("records")

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

with open(f"{OUT}/summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)
print(json.dumps(summary, indent=2, default=str))
