# Project context and development log

This file records the state of the project, the decisions behind it, and the
plan for the next phases. Read it first before starting new work.

Last updated: October 24, 2026, version 2 (branch `v2-stronger-gnn`).
Sections 1 and 2 describe version 1; section 7 has the version 2 design and
results.

## 1. What the project is

**Question.** Can conditions in an ECOWAS member state, measured the year before,
forecast ECOWAS punitive action (suspension, sanctions, border closure) about 12
months ahead? Does modeling the region as a network of bordering states improve
that forecast? How much economic damage do those actions cause?

**Origin.** The project began as a trial run for a graduate research course at
the University of Maryland. It now continues as a personal portfolio project
and it may become a second-year master's project later.

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

## 3. Repository state (version 2)

```
config.yaml        every setting: years, members, borders, sources, paths, models
pipeline/          ingest.py, transform.py, checks.py, run.py, config.py
sql/               staging.sql, features.sql (DuckDB SQL models)
src/               analysis.py (Baselines A to C), gnn.py (Baseline C v2, GNN v2),
                   watchlist.py, charts.py, gnn_chart.py
dags/              Airflow 3 DAGs: ecowas_monthly, ecowas_models
pipeline/tasks.py  one plain function per Airflow task
docker/            airflow.Dockerfile; docker-compose.yaml at the root
dashboard/         app.py (Streamlit), requirements.txt
tests/             pytest: 44 tests + 1 live World Bank API test (+3 DAG tests
                   that run when Airflow is installed)
scripts/           check_reproducible.py, make_vdem_subset.py,
                   refresh_world_bank.py, refresh_gdelt.py
.github/workflows/ ci.yml: test, live-api (non-blocking), gnn-smoke, dag-test,
                   airflow-compose (runs the monthly DAG in the image), docker
                   refresh-data.yml: World Bank and GDELT snapshots
                   run-models.yml: the full run, outputs committed
Dockerfile, run_all.sh, requirements.txt, requirements-pipeline.txt
```

- Data (version 2): V-Dem v16 subset 1980 to 2025 (from `vdeminstitute/vdemdata`);
  World Bank 2000 to 2025 (one vintage, October 24, 2026); GDELT 1.0 counts by
  member and month, 2000 to 2025 (4,727 files, 833 million rows read, 29.4
  million events in members; 2 daily files listed by GDELT but missing on its
  server, recorded in `data/raw/gdelt_country_month.json`); 21 event rows.
- Pipeline: ingest (World Bank API with `--live`, or the saved snapshot by
  default; V-Dem release subset; events; GDELT snapshot; reference data from
  config) into DuckDB `raw`, then SQL `staging` and `features`, then 20 data
  quality checks,
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
`python src/gnn.py` (version 2: 20 to 40 minutes on 4 cores; `--quick` for a
one-minute smoke test), `pytest`, or `./run_all.sh`.

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
- Airflow can be tested in the workspace: a Python 3.12 virtual environment
  (`uv venv -p 3.12`) with `apache-airflow==3.3.2` and its constraints file
  from raw.githubusercontent.com, then `airflow db migrate` and
  `airflow dags test ecowas_monthly -c '{"live": false, "update_gdelt": false}'`.
  A DAG run whose logical date is before the DAG's start_date creates no tasks
  and still reports success; the DAGs' start dates are in the past for that
  reason. In Airflow 3, `DagBag(...).get_dag()` needs the database; tests use
  `dagbag.dags[...]`. A folder named `airflow/` at the repository root hides
  the Airflow package from Python, so the Docker files are in `docker/`.
- PyTorch installs in the workspace from PyPI (the GPU build, about 4 GB with its
  libraries; it runs on CPU), so `python src/gnn.py --quick` can be tested here.
- The owner's local copy is a clone on his Mac; he runs `git pull` to update it.

## 6. Roadmap

- [x] **Phase 1, foundation:** config, World Bank API ingestion, DuckDB
  warehouse, data checks, tests, CI, Docker (PR #1)
- [x] **Version 2:** 2001 start, more features, GDELT, GAT, nested tuning.
  See section 7.
- [x] **Phase 2, orchestration (October 24, 2026):** Airflow 3.3.2.
  `ecowas_monthly` (2nd of each month): update_gdelt -> ingest (live World
  Bank) -> transform -> checks -> export -> baselines -> watch_list -> publish
  (`runs/YYYY-MM-DD/` with `changes.md`). `ecowas_models` (by hand): gnn_full ->
  figures -> publish_models. GDELT updates are incremental (`refresh_gdelt.py
  --update`: only new complete months). `gnn.py --watch-list` refits Baseline C
  v2 in seconds; its output is identical to the full run's. `ECOWAS_DATA_DIR`
  moves the warehouse, GDELT snapshot, and runs out of the repository. Docker
  Compose: Airflow standalone + Postgres. Verified: `airflow dags test` here,
  and in CI inside the image (54 seconds; live World Bank; GDELT added
  January to September 2026 from 273 files).
- [x] **Phase 3, monthly panel (October 24, 2026):** `sql/monthly.sql` builds
  `features.country_month` (4,320 rows): GDELT protest and violence shares over
  1, 3, and 6 months (missing when the window has fewer than 50 events: 91
  one-month values, 6 three-month values), the previous year's yearly inputs,
  `action_this_month`, and `action_next_6m` (a new action in months t+1 to t+6;
  87 positive months, 16 action months). 3 new checks (23 in total). Exported to
  `data/processed/ecowas_country_month.csv`. Known limits: V-Dem values for a
  year are published in March of the next year, so using them from January is a
  slight look-ahead of up to two months; the six-month outcome for July to
  December 2024 looks into 2025, where only Guinea-Bissau's November 2025
  suspension is coded (no other action is known for January to June 2025).
- [ ] **Version 3 model:** a six-month forecast on the monthly panel. Write its
  design first. Hold out whole years with a six-month gap on each side, because
  neighboring months share outcome windows. The 2001 to 2024 years have been
  seen, so judge it mainly on 2025 onward.
- [x] **Phase 4, dashboard (October 24, 2026):** `dashboard/app.py`, five tabs
  (watch list map and table, monthly GDELT trends, economic impact, models,
  data). Reads committed files, so it runs on Streamlit Community Cloud; with
  `ECOWAS_DATA_DIR` it shows the newest Airflow run and `meta.load_log`. The map
  is drawn as filled shapes from `data/geo/west_africa.geojson`, because Plotly's
  geographic maps download a base map from cdn.plot.ly (blocked in the
  workspace, and an outside dependency). Own `dashboard/requirements.txt`
  (Community Cloud reads the file next to the app, so PyTorch is not
  installed there). Tests: Streamlit AppTest; CI also starts the container.
  Deploying needs the owner's Streamlit account (share.streamlit.io: repository
  `landontownsend/ecowas-sanctions-gnn`, branch `main`, file `dashboard/app.py`).
  Live since October 24, 2026: https://ecowas-sanctions-gnn-qdsx3dzrztyzxsptrsq2gl.streamlit.app/
- [x] **Phase 5, models:** done as part of version 2 (GAT and nested
  cross-validation).

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
- October 24, 2026: design fixed (this section, commit `11940e6`).
- October 24, 2026: data built. V-Dem v16 subset 1980 to 2025; World Bank 2000
  to 2025; GDELT 1.0, 4,727 files, 833 million rows read, 29.4 million events in
  members (2 daily files listed by GDELT but missing on its server). No member
  has fewer than 50 GDELT events in any year (lowest: Cabo Verde 2005, 258), so
  no GDELT share is missing.
- October 24, 2026: first full run (GitHub Actions run 40129198200). No bug
  fixes were needed, and no design change was made after the run.

### Version 2 results (2001 to 2024; 24 held-out years, 360 rows, 13 actions)

| Model | AUC | Top-3 hits (of 13) |
|---|---|---|
| Baseline A (version 1 inputs) | 0.732 | 6 |
| Baseline B (version 1 inputs) | 0.680 | 6 |
| Baseline C (version 1 inputs) | 0.747 | 8 |
| Baseline C v2 | 0.778 | 7 |
| GNN v2 | 0.616 | 7 |
| GNN v2, all links removed | 0.734 | 8 |

- Bootstrap (GNN v2 minus Baseline C v2): [-0.349, 0.001]; GNN ahead in 2.6%
  of resamples. AUC intervals: Baseline C v2 [0.589, 0.923], GNN v2
  [0.388, 0.766].
- Decision rule: **Baseline C v2 is the main model.** The watch list
  (`results/risk_next12_v2.csv`) and the risk map use it.
- Tuning: Baseline C v2 chose C = 0.01 in all 24 folds. GNN v2 chose GAT in 20
  of 24 folds and the WAEMU links in 17.
- Diagnosis: within a year the GNN ranks about as well as Baseline C v2 (7 of
  13 top-3 hits each), but its scores change scale between held-out years (the
  highest score in a year ranges from 0.000 to 1.000; median score 0.002),
  which is overfitting with 11 inputs and 13 actions. Pooled AUC penalizes this.
- The link ablation reversed: in version 1 removing links lowered the GNN's
  AUC (0.74 to 0.66); in version 2 removing them raised it (0.62 to 0.73).
- 2025 check: GNN v2 ranked Guinea-Bissau 3rd (Baseline C v2: 12th) and Benin
  11th (Baseline C v2: 6th).
- 2026 watch list (Baseline C v2): Guinea-Bissau 85 (Elevated), Guinea 74
  (Elevated), Benin 53. Scores are narrower than in version 1 because of the
  strong penalty (C = 0.01).
- Economic impact: 11 of 13 actions trail the region; median about -10 points;
  Niger 2023 -29; coastal median -15, landlocked -4. Coups: 8 of 12
  successful-coup years drew action, 0 of 8 failed-only years.

**Checks of the result**
- Reproducible: a second full run on a different machine (cloud workspace,
  different PyTorch build) gave identical settings, scores, bootstrap, and
  decision.
- Not a code bug: the version 2 code run with the version 1 setup (2010 to
  2024, 3 node features, GraphSAGE, borders) gives GNN 0.743 (5 top-3 hits) and
  0.663 without links (4), the version 1 results. The drop comes from the
  design.

**Framing for version 2.** Report the result as it is: Baseline C v2 is the
best model; GNN v2 did not beat it; the GNN's strengths are its within-year
ranking and the Guinea-Bissau 2025 case. Any later model change (for example
tuning epochs or weight decay to reduce the overfitting) is a new version with
its own design written first, and it has to be read knowing that the 2001 to
2024 years have now been seen; new years (2025 onward) are the clean test.
