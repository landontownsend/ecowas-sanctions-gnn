# Forecasting ECOWAS Sanctions Risk with Graph Neural Networks

When a West African government falls to a coup or stalls its return to civilian rule, the regional bloc ECOWAS often suspends it, sanctions it, or closes its borders, and those actions cut the country's imports well behind the rest of the region. This project asks whether conditions in a member state, measured the year before, can forecast that response, and whether modeling the region as a **network of bordering states** improves the forecast.

It builds a country-year panel of all 15 ECOWAS members (2010 to 2024), benchmarks three penalized logistic regressions, and tests a **graph neural network (GraphSAGE, PyTorch Geometric)** against them under a pre-specified, leave-one-year-out protocol.

![Model comparison](figures/fig4_model_comparison.png)

## Key results

| Model | Held-out AUC | Actual target in that year's top 3 |
|---|---|---|
| Baseline A: own conditions | 0.76 | 5 of 10 |
| Baseline B: + region-wide coups | 0.71 | 5 of 10 |
| **Baseline C: + border neighbors** | **0.78** | **6 of 10** |
| Graph neural network (GraphSAGE) | 0.74 | 5 of 10 |
| Graph neural network, border edges removed | 0.66 | 4 of 10 |

* **Neighbors matter.** Adding conditions in bordering states improves on country-by-country forecasting (Baseline A to C), and removing the border edges from the graph neural network drops its AUC from 0.74 to 0.66. Two independent tests point the same way.
* **The graph neural network does not beat Baseline C.** The year-clustered bootstrap interval for the difference (network minus Baseline C) is [-0.23, +0.12]. Under the decision rule fixed before testing, the simpler model is retained. With ten positive events, learned neighbor weights have too little signal to outperform a theory-driven summary.
* **Actions are costly.** In 9 of 10 actions, real imports fell behind members facing no action that year, by a median of about 11 percentage points (Niger 2023: 29 points).
* **Limit.** The forecast reads continued pressure on military-led governments well but does not anticipate first coups under civilian rule; out of sample, it ranked Guinea-Bissau 12th of 15 for 2025, before its November 2025 coup and suspension.

![Risk map](figures/fig3_risk_map.png)

## Approach

**Data.** One row per member per year (15 x 15 = 225). Political predictors come from V-Dem v16, including the Powell and Thyne coup counts, a military rule index, and mass mobilization. Economic variables come from the World Bank World Development Indicators. The ten ECOWAS actions (suspensions, sanctions, border closures) are hand-coded from ECOWAS communiqués and news coverage.

**Network.** Members are nodes; shared land borders are edges (Cabo Verde has none). One graph per year.

**Outcome.** Whether ECOWAS took any punitive action against a member that year. A 0 to 3 severity scale is kept for description.

**Baselines (Stage 1).** L2-penalized logistic regressions (scikit-learn, C = 1, balanced class weights, standardized inputs), all inputs lagged one year:
* A: military rule index and successful coups at home
* B: A + successful coups elsewhere in ECOWAS over the past three years
* C: A + highest military rule score among bordering members and their coups over the past three years

**Graph neural network (Stage 2).** Two-layer GraphSAGE (hidden size 16, ReLU, dropout 0.2) with a linear output, trained with weighted binary cross-entropy, Adam (learning rate 0.01, weight decay 0.005), 300 epochs, averaged over a 25-seed ensemble. Node features are each country's own lagged conditions only, so the network must learn how to combine its neighbors' information. An ablation removes all edges.

**Evaluation.** Leave one year out: train on 14 years, score the held-out year, repeat for all 15. Metrics are AUC and top-3 hits per year, with a year-clustered bootstrap (2,000 resamples) for the network-versus-Baseline-C difference. The design and decision rule were fixed before any held-out result was examined. A separate out-of-sample check trains on 2010 to 2024 and scores 2025.

**Economic impact.** Change in real imports from the year before an action to the average of the action year and the year after, minus the median change among members with no action that year.

## Repository structure

```
ecowas-sanctions-gnn/
├── data/
│   ├── raw/                  # V-Dem subset, World Bank series, hand-coded ECOWAS events
│   └── processed/            # ecowas_country_year.csv (the analysis panel)
├── src/
│   ├── build_dataset.py      # builds the panel from data/raw
│   ├── analysis.py           # descriptives, economic impact, baselines, 2025 check, current risk
│   ├── gnn.py                # graph neural network, ablation, bootstrap comparison
│   ├── charts.py             # timeline, impact chart, risk map
│   └── gnn_chart.py          # model comparison chart
├── results/                  # CSV and JSON outputs
└── figures/                  # PNG figures
```

## Reproduce

```bash
pip install -r requirements.txt
python src/build_dataset.py   # optional: rebuilds data/processed from data/raw
python src/analysis.py        # baselines and descriptive results
python src/gnn.py             # graph neural network (about 8 minutes on a laptop CPU)
python src/charts.py          # figures 1 to 3 (downloads Natural Earth shapes on first run)
python src/gnn_chart.py       # figure 4
```

Run every command from the repository root. Figures use the Inter typeface if it is installed and fall back to the default font otherwise.

## Limitations

* Ten positive events is a small base for any model, and especially for a neural network.
* Annual data blurs events that unfold over days; a monthly panel would support a six-month horizon and add events.
* Risk scores rank members; because of class weighting they are not calibrated probabilities.
* Import losses are associated with ECOWAS actions but are not causally separated from the effect of the underlying coup.
* Several hand-coded event dates are marked in `data/raw/ecowas_events.csv` for confirmation against the original communiqués.

## Roadmap

Work in progress, in planned order:

* **Data pipeline:** pull V-Dem and World Bank data through their APIs, store raw, staging, and feature tables in DuckDB, and schedule monthly refreshes with Apache Airflow, with data quality checks at each step
* **More events:** monthly country features from GDELT event data, for more training records and a six-month forecasting horizon
* **Dashboard:** a Streamlit app with the risk map, the current watch list, and the import damage chart
* **Engineering:** a Dockerfile, a single configuration file for all model settings, tests, and continuous integration with GitHub Actions
* **Models:** a second graph architecture (graph attention network) and nested cross-validation to tune both networks without touching the test years

## Data sources

* Coppedge, M., et al. *V-Dem Dataset v16.* Varieties of Democracy Project. Licensed CC BY-SA 4.0; the subset in `data/raw/vdem_ecowas_subset.csv` is shared under the same license.
* Powell, J. M., and Thyne, C. L. (2011). Global Instances of Coups from 1950 to 2010: A New Dataset. *Journal of Peace Research*, 48(2), 249 to 259 (via V-Dem).
* World Bank. *World Development Indicators.* Licensed CC BY 4.0.
* Natural Earth. *Admin 0 Countries, 1:50m.* Public domain.
* ECOWAS Commission communiqués and news coverage, hand-coded by the author.

## Author

Landon Townsend, M.S. Applied Political Analytics, University of Maryland. The project began as a trial run for a graduate research course and continues as a personal project.

Code is released under the MIT License; data files keep the licenses of their sources.
