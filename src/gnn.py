"""
gnn.py
ECOWAS sanctions risk: Stage 2, version 2.
The graph neural network (GNN v2) against Baseline C v2, with nested tuning.

The design was fixed before any version 2 result: docs/project-context.md,
section 7. Every setting is in config.yaml (models.features_v2, baseline_v2,
gnn_v2, bootstrap).

Run from the repository root, after the pipeline:
  python src/gnn.py           full run (GitHub Actions: .github/workflows/run-models.yml)
  python src/gnn.py --quick   smoke test: 2 held-out years, 20 epochs, 1 seed;
                              writes to results_quick/ (used by CI)
Reads:  data/processed/ecowas_country_year.csv   (study period)
        data/processed/ecowas_forecast_inputs.csv (2025 and 2026)
Writes: results/gnn_loyo_predictions.csv  held-out scores for every country-year
        results/gnn_tuning.csv            the setting chosen in each held-out year
        results/gnn_action_ranks.csv      where each action ranked in its year
        results/gnn_scores_2025_2026.csv  2025 check and 2026 scores, both models
        results/risk_next12_v2.csv        the watch list from the main model
        results/gnn_summary.json          every number in the README

Sections
  1. Settings, data, and the graph
  2. Missing values and standardizing (training rows only)
  3. Baseline C v2: logistic regression, tuned
  4. GNN v2: GraphSAGE or GAT, borders with or without WAEMU links, tuned
  5. One held-out year: tune, fit, and score every model
  6. The full run: every held-out year in parallel, bootstrap, decision rule
"""

import json
import os
import sys
from multiprocessing import get_context

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch_geometric.nn import GATConv, SAGEConv

from watchlist import build_watch_list   # src/watchlist.py

torch.set_num_threads(1)   # each worker process uses one CPU core

# =============================================================================
# SECTION 1: SETTINGS, DATA, AND THE GRAPH
# =============================================================================

CFG = yaml.safe_load(open("config.yaml"))
FEAT = CFG["models"]["features_v2"]
BASE = CFG["models"]["baseline_v2"]
G = CFG["models"]["gnn_v2"]
BOOT = CFG["models"]["bootstrap"]

QUICK = "--quick" in sys.argv            # smoke test with tiny settings
OUT = "results_quick" if QUICK else CFG["paths"]["results"]
EPOCHS = 20 if QUICK else G["epochs"]
FINAL_SEEDS = [0] if QUICK else list(range(G["seeds"]))
INNER_SEEDS = [0] if QUICK else list(range(G["inner_seeds"]))
BOOT_REPS = 50 if QUICK else BOOT["reps"]

OWN = FEAT["own"]                        # the 11 own-country inputs (node features)
BORDER = FEAT["border"]                  # the 2 hand-built neighbor summaries

# The 8 settings, in the order they are tried (ties go to the first)
GRID = [{"layer": layer, "graph": graph, "hidden": hidden}
        for layer in G["grid"]["layer"]
        for graph in G["grid"]["graph"]
        for hidden in G["grid"]["hidden"]]

YEARS = list(range(CFG["years"]["start"], CFG["years"]["end"] + 1))   # 2001 to 2024
CHECK = CFG["years"]["check"]            # 2025
FORECAST = CFG["years"]["forecast"]      # 2026

# The graph: one node per member, in alphabetical order (BEN = 0, BFA = 1, ...)
NODES = sorted(c["iso3"] for c in CFG["countries"])
IDX = {iso: i for i, iso in enumerate(NODES)}
N = len(NODES)

# Link type 1: shared land borders (each border appears in both directions)
BORDER_EDGES = torch.tensor([[IDX[a], IDX[b]] for a in NODES for b in CFG["borders"][a]],
                            dtype=torch.long).t()
# Link type 2: members of the CFA franc union (WAEMU), every pair, both directions
WAEMU = [c["iso3"] for c in CFG["countries"] if c["waemu"] == 1]
WAEMU_EDGES = torch.tensor([[IDX[a], IDX[b]] for a in WAEMU for b in WAEMU if a != b],
                           dtype=torch.long).t()
NO_EDGES = torch.empty((2, 0), dtype=torch.long)


def graph_edges(graph, use_edges=True):
    """The edge sets for one year's graph: one set per link type."""
    if graph == "borders":
        links = [BORDER_EDGES]
    else:
        links = [BORDER_EDGES, WAEMU_EDGES]
    if not use_edges:
        links = [NO_EDGES for _ in links]   # same model shape, every edge removed
    return links


def stack_edges(edges, n_graphs):
    """
    Training stacks several years into one big graph: year 1 is nodes 0 to 14,
    year 2 is nodes 15 to 29, and so on. This copies one year's edges for each.
    """
    return torch.cat([edges + N * k for k in range(n_graphs)], dim=1)


# Model inputs, built by the pipeline (sql/features.sql)
df = pd.read_csv(CFG["paths"]["features_csv"])
forecast = pd.read_csv(CFG["paths"]["forecast_csv"])
inputs = pd.concat([df, forecast], ignore_index=True)


def year_table(year, cols):
    """One row per member for one year, in NODES order, as a numpy array (NaN = missing)."""
    rows = inputs[inputs.year == year].set_index("iso3").loc[NODES]
    return rows[cols].to_numpy(dtype=float)


SCORED_YEARS = YEARS + [CHECK, FORECAST]
X = {y: year_table(y, OWN) for y in SCORED_YEARS}             # GNN v2 node features
XB = {y: year_table(y, OWN + BORDER) for y in SCORED_YEARS}   # Baseline C v2 inputs
Y = {y: year_table(y, ["ecowas_action"])[:, 0].astype(int) for y in YEARS}

# =============================================================================
# SECTION 2: MISSING VALUES AND STANDARDIZING
# Both are learned from the training years only, then applied to any year.
# =============================================================================


def make_scaler(train_years, arrays):
    """
    Learn the median, mean, and spread of each input from the training years.
    Returns a function that fills missing values with the median and then
    standardizes (mean 0, spread 1).
    """
    stacked = np.vstack([arrays[y] for y in train_years])
    median = np.nanmedian(stacked, axis=0)
    filled = np.where(np.isnan(stacked), median, stacked)
    mean = filled.mean(axis=0)
    spread = filled.std(axis=0) + 1e-6

    def scale(a):
        a = np.where(np.isnan(a), median, a)
        return (a - mean) / spread
    return scale


def inner_folds(train_years):
    """Split the training years into groups for tuning: years in order, assigned in turn."""
    years = sorted(train_years)
    k = G["inner_folds"]
    return [years[i::k] for i in range(k)]


def pooled_auc(labels, preds):
    """AUC of all the predictions together."""
    return float(roc_auc_score(np.concatenate(labels), np.concatenate(preds)))

# =============================================================================
# SECTION 3: BASELINE C v2
# Logistic regression with the 11 own-country inputs and the 2 border summaries.
# The penalty strength C is tuned with the inner folds.
# =============================================================================


def baseline_fit(train_years, C):
    """Fit Baseline C v2 on train_years. Returns a function that scores one year."""
    scale = make_scaler(train_years, XB)
    Xtr = np.vstack([scale(XB[y]) for y in train_years])
    ytr = np.concatenate([Y[y] for y in train_years])
    model = LogisticRegression(C=C, class_weight=BASE["class_weight"], max_iter=BASE["max_iter"])
    model.fit(Xtr, ytr)

    def predict(year):
        return model.predict_proba(scale(XB[year]))[:, 1]
    return predict


def baseline_tune(train_years):
    """Try every C on the inner folds; return the best C and every score."""
    scores = []
    for C in BASE["C_grid"]:
        labels, preds = [], []
        for val in inner_folds(train_years):
            predict = baseline_fit([y for y in train_years if y not in val], C)
            for y in val:
                labels.append(Y[y])
                preds.append(predict(y))
        scores.append(pooled_auc(labels, preds))
    best = int(np.argmax(scores))          # argmax picks the first of any ties
    return BASE["C_grid"][best], scores

# =============================================================================
# SECTION 4: GNN v2
# Two graph layers. With two link types, each layer has one graph convolution
# per link type and adds their outputs together.
# =============================================================================


class GNN(torch.nn.Module):
    def __init__(self, n_in, hidden, layer, n_links):
        super().__init__()

        def conv(n_from, n_to):
            if layer == "sage":
                return SAGEConv(n_from, n_to)
            return GATConv(n_from, n_to, heads=G["gat_heads"], concat=False)

        self.layer1 = torch.nn.ModuleList([conv(n_in, hidden) for _ in range(n_links)])
        self.layer2 = torch.nn.ModuleList([conv(hidden, hidden) for _ in range(n_links)])
        self.out = torch.nn.Linear(hidden, 1)

    def forward(self, x, edge_sets):
        h = sum(c(x, e) for c, e in zip(self.layer1, edge_sets))
        h = F.relu(h)
        h = F.dropout(h, p=G["dropout"], training=self.training)
        h = sum(c(h, e) for c, e in zip(self.layer2, edge_sets))
        h = F.relu(h)
        return self.out(h).squeeze(-1)


def gnn_fit_predict(train_years, score_years, setting, seeds, use_edges=True):
    """
    Train one GNN per seed on train_years and return the averaged scores for
    each year in score_years, as {year: array of 15 scores}.
    """
    scale = make_scaler(train_years, X)
    x = torch.tensor(np.vstack([scale(X[y]) for y in train_years]), dtype=torch.float)
    y = torch.tensor(np.concatenate([Y[y] for y in train_years]), dtype=torch.float)
    one_year = graph_edges(setting["graph"], use_edges)
    train_edges = [stack_edges(e, len(train_years)) for e in one_year]

    # Rare action years count as much as quiet years (as in version 1)
    positives = float(y.sum())
    pos_weight = torch.tensor((len(y) - positives) / positives)

    scores = {yr: np.zeros(N) for yr in score_years}
    for seed in seeds:
        torch.manual_seed(seed)
        model = GNN(x.shape[1], setting["hidden"], setting["layer"], len(one_year))
        opt = torch.optim.Adam(model.parameters(), lr=G["lr"], weight_decay=G["weight_decay"])
        model.train()
        for _ in range(EPOCHS):
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(x, train_edges), y,
                                                      pos_weight=pos_weight)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            for yr in score_years:
                xt = torch.tensor(scale(X[yr]), dtype=torch.float)
                scores[yr] += torch.sigmoid(model(xt, one_year)).numpy() / len(seeds)
    return scores


def gnn_tune(train_years):
    """Try every setting on the inner folds; return the best setting and every score."""
    scores = []
    for setting in GRID:
        labels, preds = [], []
        for val in inner_folds(train_years):
            fit_years = [y for y in train_years if y not in val]
            p = gnn_fit_predict(fit_years, val, setting, INNER_SEEDS)
            for y in val:
                labels.append(Y[y])
                preds.append(p[y])
        scores.append(pooled_auc(labels, preds))
    best = int(np.argmax(scores))          # argmax picks the first of any ties
    return GRID[best], scores

# =============================================================================
# SECTION 5: ONE HELD-OUT YEAR
# Everything here uses only the training years; the held-out year is scored once.
# =============================================================================


def run_fold(held_out):
    train_years = [y for y in YEARS if y != held_out]
    C, c_scores = baseline_tune(train_years)
    setting, g_scores = gnn_tune(train_years)
    p_gnn = gnn_fit_predict(train_years, [held_out], setting, FINAL_SEEDS)[held_out]
    p_none = gnn_fit_predict(train_years, [held_out], setting, FINAL_SEEDS, use_edges=False)[held_out]
    p_base = baseline_fit(train_years, C)(held_out)
    print(f"  held out {held_out}: C = {C}, GNN setting = {setting}", flush=True)
    return {"year": held_out, "C": C, "setting": setting,
            "baseline_inner_aucs": c_scores, "gnn_inner_aucs": g_scores,
            "baseline_c_v2": p_base.tolist(), "gnn": p_gnn.tolist(), "gnn_no_edges": p_none.tolist()}


def run_forecast():
    """Tune and fit on every study year, then score the check and forecast years."""
    C, _ = baseline_tune(YEARS)
    setting, _ = gnn_tune(YEARS)
    p_gnn = gnn_fit_predict(YEARS, [CHECK, FORECAST], setting, FINAL_SEEDS)
    predict = baseline_fit(YEARS, C)
    print(f"  forecast: C = {C}, GNN setting = {setting}", flush=True)
    return {"C": C, "setting": setting,
            "gnn": {str(y): p_gnn[y].tolist() for y in [CHECK, FORECAST]},
            "baseline_c_v2": {str(y): predict(y).tolist() for y in [CHECK, FORECAST]}}

# =============================================================================
# SECTION 6: THE FULL RUN
# =============================================================================


def top3_hits(d, col):
    """Number of actions that ranked in the top 3 of their year."""
    hits = 0
    for _, g in d.groupby("year"):
        r = g[col].rank(ascending=False, method="first")
        hits += int((r[g.action == 1] <= 3).sum())
    return hits


def main():
    os.makedirs(OUT, exist_ok=True)
    outer_years = YEARS[-2:] if QUICK else YEARS
    workers = 2 if QUICK else (os.cpu_count() or 1)
    print(f"{len(outer_years)} held-out years, {len(GRID)} GNN settings, {workers} workers", flush=True)

    with get_context("spawn").Pool(workers) as pool:
        forecast_job = pool.apply_async(run_forecast)      # the largest task, started first
        folds = pool.map(run_fold, outer_years, chunksize=1)
        fc = forecast_job.get()

    # --- Held-out scores for every country-year ---------------------------------
    rows = []
    for f in folds:
        for i, iso in enumerate(NODES):
            rows.append({"iso3": iso, "year": f["year"], "action": int(Y[f["year"]][i]),
                         "baseline_c_v2": f["baseline_c_v2"][i], "gnn": f["gnn"][i],
                         "gnn_no_edges": f["gnn_no_edges"][i]})
    loyo = pd.DataFrame(rows)
    loyo.round(4).to_csv(f"{OUT}/gnn_loyo_predictions.csv", index=False)

    tuning = pd.DataFrame([{"held_out_year": f["year"], "baseline_C": f["C"],
                            "gnn_layer": f["setting"]["layer"], "gnn_graph": f["setting"]["graph"],
                            "gnn_hidden": f["setting"]["hidden"],
                            "gnn_inner_auc": round(max(f["gnn_inner_aucs"]), 3),
                            "baseline_inner_auc": round(max(f["baseline_inner_aucs"]), 3)}
                           for f in folds])
    tuning.to_csv(f"{OUT}/gnn_tuning.csv", index=False)

    models = ["baseline_c_v2", "gnn", "gnn_no_edges"]
    summary = {"design": "docs/project-context.md, section 7",
               "years": [YEARS[0], YEARS[-1]], "held_out_years": len(outer_years),
               "actions": int(loyo.action.sum()),
               "settings": {"grid": GRID, "C_grid": BASE["C_grid"], "epochs": EPOCHS,
                            "seeds": len(FINAL_SEEDS), "inner_seeds": len(INNER_SEEDS),
                            "inner_folds": G["inner_folds"], "dropout": G["dropout"],
                            "lr": G["lr"], "weight_decay": G["weight_decay"]},
               "loyo": {m: {"auc": round(float(roc_auc_score(loyo.action, loyo[m])), 3),
                            "top3_hits": top3_hits(loyo, m)} for m in models}}

    # --- Year-clustered bootstrap: resample the held-out years with replacement --
    rng = np.random.default_rng(BOOT["seed"])
    groups = {y: loyo[loyo.year == y] for y in outer_years}
    diffs, aucs_g, aucs_c = [], [], []
    for _ in range(BOOT_REPS):
        sample = pd.concat([groups[y] for y in rng.choice(outer_years, size=len(outer_years))])
        if sample.action.nunique() < 2:
            continue
        ag = roc_auc_score(sample.action, sample.gnn)
        ac = roc_auc_score(sample.action, sample.baseline_c_v2)
        aucs_g.append(ag)
        aucs_c.append(ac)
        diffs.append(ag - ac)

    def interval(a):
        if not a:
            return None
        return [round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)]
    summary["bootstrap"] = {"reps": len(diffs), "gnn_auc_95ci": interval(aucs_g),
                            "baseline_c_v2_auc_95ci": interval(aucs_c),
                            "difference_95ci": interval(diffs),
                            "share_gnn_better": round(float(np.mean(np.array(diffs) > 0)), 3) if diffs else None}

    # --- Decision rule (fixed in advance) ---------------------------------------
    gnn_wins = summary["loyo"]["gnn"]["auc"] > summary["loyo"]["baseline_c_v2"]["auc"]
    main_model = "gnn" if gnn_wins else "baseline_c_v2"
    summary["decision"] = {"rule": "GNN v2 is the main model if its held-out AUC is higher "
                                   "than Baseline C v2's",
                           "main_model": "GNN v2" if gnn_wins else "Baseline C v2"}

    # --- Which settings the tuning chose -----------------------------------------
    summary["chosen_settings"] = (tuning.groupby(["gnn_layer", "gnn_graph", "gnn_hidden"]).size()
                                  .reset_index(name="held_out_years").to_dict("records"))
    summary["chosen_C"] = tuning.baseline_C.value_counts().sort_index().to_dict()

    # --- Where each action ranked in its own year --------------------------------
    acts = []
    for yr, g in loyo.groupby("year"):
        g = g.copy()
        for m in models:
            g[f"rank_{m}"] = g[m].rank(ascending=False, method="first").astype(int)
        for r in g[g.action == 1].itertuples():
            acts.append({"iso3": r.iso3, "year": int(yr),
                         "rank_baseline_c_v2": r.rank_baseline_c_v2, "rank_gnn": r.rank_gnn,
                         "rank_gnn_no_edges": r.rank_gnn_no_edges,
                         "p_baseline_c_v2": round(r.baseline_c_v2, 3), "p_gnn": round(r.gnn, 3)})
    pd.DataFrame(acts).to_csv(f"{OUT}/gnn_action_ranks.csv", index=False)
    summary["action_ranks"] = acts

    # --- 2025 check and 2026 scores, both models ---------------------------------
    sc = pd.DataFrame({"iso3": NODES})
    for m in ["gnn", "baseline_c_v2"]:
        for y in [CHECK, FORECAST]:
            sc[f"{m}_{y}"] = fc[m][str(y)]
        sc[f"rank_{m}_{CHECK}"] = sc[f"{m}_{CHECK}"].rank(ascending=False, method="first").astype(int)
    sc.round(4).to_csv(f"{OUT}/gnn_scores_{CHECK}_{FORECAST}.csv", index=False)
    summary[f"oos_{CHECK}"] = {
        m: {"gnb_rank": int(sc.loc[sc.iso3 == "GNB", f"rank_{m}_{CHECK}"].iloc[0]),
            "ben_rank": int(sc.loc[sc.iso3 == "BEN", f"rank_{m}_{CHECK}"].iloc[0])}
        for m in ["gnn", "baseline_c_v2"]}
    summary["forecast_fit"] = {"C": fc["C"], "gnn_setting": fc["setting"]}

    # --- The watch list from the main model ---------------------------------------
    scores = pd.DataFrame({"iso3": NODES, "risk": fc[main_model][str(FORECAST)]})
    scores["rank"] = scores.risk.rank(ascending=False, method="first").astype(int)
    watch = build_watch_list(scores.sort_values("rank"), CFG)
    watch["model"] = summary["decision"]["main_model"]
    watch.to_csv(f"{OUT}/risk_next12_v2.csv", index=False)
    summary[f"watch_list_{FORECAST}"] = watch[["iso3", "risk_score_0_100", "member_rank", "tier"]] \
        .astype(str).to_dict("records")

    with open(f"{OUT}/gnn_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: summary[k] for k in ["loyo", "bootstrap", "decision"]}, indent=2))


if __name__ == "__main__":
    main()
