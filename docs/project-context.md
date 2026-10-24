# Project context and development log

This file records the state of the project, the decisions behind it, and the
plan for the next phases. Read it first before starting new work.

Last updated: October 24, 2026, cleanup before Phase 2 (branch
`cleanup-before-phase-2`).

## 1. What the project is

**Question.** Can conditions in an ECOWAS member state, measured the year before,
forecast ECOWAS punitive action (suspension, sanctions, border closure) about 12
months ahead? Does modeling the region as a network of bordering states improve
that forecast? How much economic damage do those actions cause?

**Origin.** The project began as a trial run for a graduate research course at
the University of Maryland (M.S. Applied Political Analytics). It now continues
as a personal portfolio project aimed at Data Engineer roles, and it may become
a second-year master's project later.

**Data.** A country-year panel: 15 ECOWAS members, 2010 to 2024, 225 rows.
- V-Dem v16: successful coups (`e_pt_coup`, Powell and Thyne), coup attempts,
  mass mobilization (`v2cagenmob`), military rule index (`v2x_ex_military`)
- World Bank WDI: GDP growth, inflation, imports as a share of GDP, real imports
  (`NE.IMP.GNFS.KD`)
- 10 ECOWAS actions in the study period (15 event rows), hand-coded in
  `data/raw/ecowas_events.csv`; several dates are flagged for confirmation
  against the original communiques
- Network: shared land borders between members (Cabo Verde has none)

## 2. Methods and results (fixed; do not change these numbers without rerunning)

**Stage 1, baselines.** L2-penalized logistic regressions (C = 1, balanced class
weights, standardized inputs), all inputs lagged one year:
- A: own military rule index and own successful coups
- B: A + successful coups in all other members over the previous 3 years
- C: A + highest military rule score among bordering members + their coups
  over the previous 3 years

**Stage 2, graph neural network.** Two-layer GraphSAGE (PyTorch Geometric),
hidden 16, dropout 0.2, Adam (lr 0.01, weight decay 0.005), 300 epochs,
25-seed ensemble, weighted BCE. Node features are each country's own lagged
inputs only. Ablation: the same model with all edges removed.

**Evaluation.** Leave one year out (train on 14 years, score the held-out year,
repeat for 15). AUC ("share of rankings correct") and top-3 hits per year.
Year-clustered bootstrap (2,000 resamples) for the GNN minus Baseline C
difference. Decision rule set before testing: the GNN replaces Baseline C only
if it beats it.

| Model | AUC | Top-3 hits (of 10) |
|---|---|---|
| Baseline A | 0.76 | 5 |
| Baseline B | 0.71 (0.705) | 5 |
| Baseline C | 0.78 | 6 |
| GNN | 0.74 | 5 |
| GNN, no edges | 0.66 | 4 |

- Bootstrap difference (GNN minus C): [-0.23, +0.12]; GNN ahead in about 31% of
  resamples. Baseline C is retained under the decision rule.
- Baseline C coefficients (odds multiplier per SD): own military rule x2.3,
  border military rule x2.3; coup counts about x0.9 and x1.0 (no effect once
  military rule is included, because ECOWAS acts in the same year as the coup).
- Coups vs. actions: 7 of 9 successful-coup years drew action; 0 of 5
  failed-only years; 3 actions without a coup.
- Import damage: change in real imports from the year before to the average of
  the action year and the next year, minus the median change of members with no
  action that year. 9 of 10 actions trail the region; median gap about -11
  points; Niger 2023 about -29. Coastal targets median -18, landlocked -4.5.
  Associated with action, not proven causal.
- GDP growth: 2.4% in action years vs. 4.7% otherwise.
- Out of sample 2025 (Baseline C): Guinea-Bissau ranked 12th before its
  November 2025 coup; Benin 7th. The GNN ranked Guinea-Bissau 6th.
- Current watch list (Baseline C, 2026): Guinea 99 (discounted: readmitted
  January 2026, which the model does not yet see), Guinea-Bissau 98, Benin 89.
  Both models put Guinea-Bissau and Guinea in their top two.

**Framing rule.** Results are reported honestly. The GNN did not beat Baseline
C. The GNN still supports the main finding: removing border edges drops its AUC
from 0.74 to 0.66, a second, separate test that neighbors matter. Do not present
the GNN as the winner.

## 3. Repository state after Phase 1

```
config.yaml        every setting: years, members, borders, sources, paths, models
pipeline/          ingest.py, transform.py, checks.py, run.py, config.py
sql/               staging.sql, features.sql (DuckDB SQL models)
src/               analysis.py (baselines), gnn.py, charts.py, gnn_chart.py
tests/             pytest: 26 tests + 1 live World Bank API test
scripts/           check_reproducible.py (compares outputs with committed files)
.github/workflows/ ci.yml: test, live-api (non-blocking), docker
Dockerfile, run_all.sh, requirements.txt, requirements-pipeline.txt
```

- Pipeline: ingest (World Bank API with `--live`, or the saved snapshot by
  default; V-Dem release subset; events; reference data from config) into
  DuckDB `raw`, then SQL `staging` and `features`, then 16 data quality checks,
  then export to `data/processed/ecowas_country_year.csv` (study period) and
  `data/processed/ecowas_forecast_inputs.csv` (the years after it).
- Every model input is built once, in `sql/features.sql`, including the border
  features (`border_military_max_lag1`, `border_coups_prev3`) and the GNN's
  `coups_successful_prev3`. `src/analysis.py` and `src/gnn.py` only read the two
  CSV files; they no longer read the V-Dem file or rebuild features.
- Years live in `config.yaml`: `start`, `end` (study period), `check` (2025),
  `forecast` (2026). The watch list tiers, suspended members, and context notes
  are under `watch_list`. Output names that carry a year (for example
  `oos_2025.csv`) are built from these settings.
- The pipeline rebuilds the processed CSV byte for byte, and all results
  reproduce exactly. CI checks this on every push.
- `src/build_dataset.py` was replaced by the pipeline and removed.

**Cleanup before Phase 2 (October 24, 2026).** No model result changed. Checked
by comparing every output with the previous commit: all model inputs, the GNN
input arrays (all 17 years), the results files, and the figures are identical.
Two intended changes:
- `data/processed/ecowas_country_year.csv` has three new columns (the border
  features and `coups_successful_prev3`); the 25 old columns are unchanged.
- `results/summary.json`: the watch list tiers now use the same rule as
  `risk_next12.csv` and the risk map (High 0.85 or more, Elevated 0.70 or more,
  "Left ECOWAS" for Mali, Burkina Faso, and Niger). Before, `summary.json` used
  a second rule (cut points 0.25 and 0.5) and called the exited members "High".
- The GNN was not rerun in the cloud workspace (PyTorch could not be
  installed there). Its inputs were checked to be identical, so its results
  should not change; rerun `python src/gnn.py` and
  `python scripts/check_reproducible.py` locally to confirm.

Run: `python -m pipeline.run [--live]`, `python src/analysis.py`,
`python src/gnn.py` (about 8 minutes on CPU), `pytest`, or `./run_all.sh`.

## 4. Working conventions

- Each phase goes on its own branch with a pull request into `main`. CI must
  pass. The owner approves merges.
- Results must reproduce; `scripts/check_reproducible.py` must pass unless a
  change to results is intended and explained.
- The saved World Bank snapshot is the default so results stay stable; live
  data is opt-in.
- Stage names: Stage 1 = baselines, Stage 2 = graph neural network. The import
  damage measure is "economic impact," not a stage.
- Code style: simple syntax and plain-English comments.

## 5. Environment notes for an AI assistant working on this repository

- A GitHub App with push access is installed on this repository, so pushes work.
- `gh pr create` fails (GraphQL is blocked); open pull requests with the REST
  API: `gh api repos/landontownsend/ecowas-sanctions-gnn/pulls --input pr.json`.
  Merge with `gh api -X PUT .../pulls/N/merge`.
- The cloud workspace cannot reach `api.worldbank.org`, `data.gdeltproject.org`,
  `v-dem.net`, or `download.pytorch.org`. GitHub Actions can, so live-data steps
  run there: `refresh-data.yml` (World Bank and GDELT snapshots) and
  `run-models.yml` (the full run, about 20 to 40 minutes). Both start from the
  Actions tab, or when a pushed commit message contains `[refresh-data]` or
  `[run-models]`, and both commit their outputs to the branch. The owner can
  allow domains in the workspace's network settings.
- GitHub Actions job logs are stored on a host the workspace cannot reach.
  Annotations can be read through the API
  (`gh api repos/OWNER/REPO/check-runs/JOB_ID/annotations`), so the scripts
  report progress and errors with `::notice::` and `::error::` lines.
- The full V-Dem dataset is in the public `vdeminstitute/vdemdata` repository
  (`data/vdem.RData`), which the workspace can clone;
  `scripts/make_vdem_subset.py` builds the subset from it.
- PyTorch installs in the workspace from PyPI (the GPU build, about 4 GB with its
  libraries; it runs on CPU), so `python src/gnn.py --quick` can be tested here.
- The owner's local copy is a clone on his Mac; he runs `git pull` to update it.

## 6. Roadmap

- [x] **Phase 1, foundation:** config, World Bank API ingestion, DuckDB
  warehouse, data checks, tests, CI, Docker (PR #1)
- [ ] **Version 2 (in progress):** 2001 start, more features, GDELT, GAT,
  nested tuning. See section 7.
- [ ] **Phase 2, orchestration:** an Apache Airflow DAG that runs the
  pipeline monthly. Planned tasks: ingest (live) -> transform -> checks ->
  export -> baselines -> publish the dated watch list. The GNN can be a separate
  or less frequent task because it takes about 8 minutes. Run Airflow locally
  with Docker Compose, reuse the functions in `pipeline/` as tasks, and add a
  DAG integrity test to CI. The owner has built an Airflow DAG before (a Reddit
  ETL project).
- [ ] **Phase 3, GDELT:** monthly country features from GDELT events (protests,
  military activity, statements about ECOWAS), for more training records and a
  six-month forecasting horizon. Options: the GDELT public dataset in Google
  BigQuery, or the raw event files. Network access must be solved first.
- [ ] **Phase 4, dashboard:** a Streamlit app with the risk map, the live watch
  list, and the import damage chart, reading from the DuckDB warehouse.
- [ ] **Phase 5, models:** a graph attention network (GAT) as a second graph
  model, and nested cross-validation to tune both networks without touching the
  test years.

## 7. Version 2 design (fixed on October 24, 2026, before any version 2 result)

**Why.** Version 1 has 10 actions in 225 country-years, and the GNN gets only 3
node features. Version 2 gives the GNN more action years, more information per
country, a richer graph, and correct tuning. The aim is a fair test in which
the GNN has a real chance to win. This design is written down before any
version 2 model is run. After the first full run, only bug fixes are allowed,
and each fix that changes a result is recorded below with its reason.

**Data**
- Study period 2001 to 2024 (24 years, 360 rows). ECOWAS adopted the Protocol
  on Democracy and Good Governance in 2001; before it, ECOWAS punished almost
  no coups, so earlier years follow different rules. Mauritania left in 2000,
  so membership is the same 15 states for the whole period.
- Check year 2025, forecast year 2026 (unchanged).
- V-Dem v16, from the official `vdeminstitute/vdemdata` release (identical to
  the version 1 subset in every overlapping value), loaded from 1980 so that
  years since the last coup can be counted.
- World Bank: one fresh pull for 2000 to 2025 (a single vintage for all years).
- GDELT 1.0 event files, 2000 to 2025, counted by country and month
  (ActionGeo country code), committed as a snapshot.
- New hand-coded actions: Togo 2005 (suspension and sanctions, about February
  19), Guinea 2009 (suspension January 10; arms embargo October 17), Niger 2009
  (suspension October 20). Guinea-Bissau 2003: no suspension or sanctions found.
  Guinea-Bissau 2025 (suspension November 27) is added for the forecast inputs
  only; it is outside the study period.

**Outcome.** Unchanged: any new ECOWAS action (suspension, sanctions, border
closure) in year t.

**Own-country features** (all from before year t; used by every version 2 model)
1. `military_rule_index_lag1`
2. `coups_successful_lag1`
3. `coups_successful_prev3` (own coups in t-3 to t-1)
4. `coups_failed_lag1`
5. `mass_mobilization_lag1`
6. `gdp_growth_lag1`
7. `inflation_lag1`
8. `ecowas_action_lag1` (an action in t-1)
9. `years_since_coup` (years since the last successful coup, capped at 20)
10. `gdelt_protest_share_lag1` (protest events, CAMEO root 14, as a share of
    all events in the country in t-1)
11. `gdelt_violence_share_lag1` (CAMEO roots 18, 19, 20, same share)

GDELT shares are missing when the country has fewer than 50 events in the
year. Missing values are filled with the median of the training rows in each
fold. Inputs are standardized with the training rows only.

**Models**
- **Baseline C v2:** L2 logistic regression, balanced class weights, the 11
  own-country features plus the two version 1 border summaries
  (`border_military_max_lag1`, `border_coups_prev3`). C is tuned from
  {0.01, 0.1, 1, 10}.
- **GNN v2:** 2 graph layers, linear output, weighted binary cross-entropy,
  Adam (lr 0.01, weight decay 0.005), dropout 0.2, 300 epochs. Node features:
  the 11 own-country features only. Tuned grid (8 settings):
  - layer type: GraphSAGE or GAT (2 attention heads, averaged)
  - graph: borders only, or borders plus a second link type for WAEMU (the
    CFA franc union: BEN, BFA, CIV, GNB, MLI, NER, SEN, TGO). With two link
    types, each layer has one graph convolution per link type, and their
    outputs are added.
  - hidden size: 16 or 32
- **Ablation:** GNN v2 with the chosen setting and every edge removed.
- **Secondary:** Baselines A, B, C with their version 1 specifications, on the
  new period.

**Tuning (nested, never sees the held-out year).** In each outer fold, the 23
training years are split into 4 inner folds (years in order, assigned in turn).
Each setting is scored by the AUC of its pooled inner-fold predictions (GNN: 3
seeds per inner fit). The best setting wins; ties go to the first setting in
grid order. The final model for the fold is refit on all 23 years (GNN: 25
seeds, averaged). For the 2025 check and the 2026 watch list, tuning uses all
24 years.

**Evaluation.** Leave one year out over 2001 to 2024. Pooled AUC and top-3 hits.
Year-clustered bootstrap (2,000 resamples) of GNN v2 minus Baseline C v2.

**Decision rule.** GNN v2 becomes the main model if its held-out AUC is higher
than Baseline C v2's. Otherwise Baseline C v2 is the main model. The watch list
and the risk map use the main model. The bootstrap interval, the share of
resamples with the GNN ahead, the ablation, and every secondary result are
reported whatever they show. Version 1 results stay in the README as version 1.

**Where it runs.** The GNN needs PyTorch, which the cloud workspace cannot
install. Data downloads and the model run happen in GitHub Actions, which
commits the outputs to the branch.

### Version 2 log
- October 24, 2026: design fixed (this section).
