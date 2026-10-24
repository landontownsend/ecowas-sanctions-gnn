"""
gnn.py
ECOWAS sanctions risk: Stage 2 forecast (graph neural network)

Run from the repository root after analysis.py:  python src/gnn.py
Reads:  data/processed/ecowas_country_year.csv   (study period, from the pipeline)
        data/processed/ecowas_forecast_inputs.csv (years after it, from the pipeline)
Writes: results/gnn_*.csv and results/gnn_summary.json

Design (fixed before looking at any held-out result)
  * One graph per year. Nodes: the 15 ECOWAS members. Edges: shared land borders.
  * Node features, all from the pipeline and all from before the year: last
    year's military rule index, last year's successful coups, and the country's
    own successful coups over the previous three years.
    Unlike Baseline C, the network receives NO hand-built neighbor summaries;
    it must learn how to combine neighbors' features itself.
  * Model (settings in config.yaml): 2-layer GraphSAGE (hidden 16, ReLU, dropout 0.2), linear
    output. Two layers let information travel two border steps.
  * Training: weighted binary cross-entropy (positives weighted by the
    negative-to-positive ratio), Adam (lr 0.01, weight decay 0.005), 300 epochs.
  * Ensemble: 25 random seeds, predictions averaged, to reduce run-to-run noise.
  * Ablation: the same model with every edge removed (no network), which
    isolates what the border network itself contributes.
  * Test: identical leave-one-year-out protocol to Baselines A to C, with
    AUC and top-3 hits, plus a year-clustered bootstrap for the difference
    between the network and Baseline C.
"""

import json
import os
import numpy as np
import pandas as pd
import yaml
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import SAGEConv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# Every setting comes from config.yaml in the repository root
CFG = yaml.safe_load(open("config.yaml"))
BASE = CFG["models"]["baseline"]
G = CFG["models"]["gnn"]
B = CFG["models"]["bootstrap"]
torch.set_num_threads(G["threads"])

OUT = CFG["paths"]["results"]
os.makedirs(OUT, exist_ok=True)

SEEDS = G["seeds"]
EPOCHS = G["epochs"]
HIDDEN = G["hidden"]
BOOT = B["reps"]

# =============================================================================
# SECTION 1: DATA, BORDERS, AND FEATURES
# =============================================================================

BORDERS = CFG["borders"]   # the border network, from config.yaml
NODES = sorted(BORDERS)
IDX = {c: i for i, c in enumerate(NODES)}
EDGES = torch.tensor([[IDX[a], IDX[b]] for a in NODES for b in BORDERS[a]], dtype=torch.long).t()
NO_EDGES = torch.empty((2, 0), dtype=torch.long)

# Every input was built by the pipeline (sql/features.sql). Stack the study
# period and the forecast years into one table of inputs.
df = pd.read_csv(CFG["paths"]["features_csv"])
forecast = pd.read_csv(CFG["paths"]["forecast_csv"])
inputs = pd.concat([df, forecast], ignore_index=True)

# Node features: the country's OWN conditions only, all from before the year
NODE_COLS = ["military_rule_index_lag1", "coups_successful_lag1", "coups_successful_prev3"]
# Baseline C inputs, for the paired comparison
BASELINE_C_COLS = ["military_rule_index_lag1", "coups_successful_lag1",
                   "border_military_max_lag1", "border_coups_prev3"]


def year_table(year, cols):
    """One row per member for one year, in NODES order (15 rows), as a numpy array."""
    rows = inputs[inputs.year == year].set_index("iso3").loc[NODES]
    return rows[cols].to_numpy(dtype=float)


YEARS = list(range(CFG["years"]["start"], CFG["years"]["end"] + 1))   # 2010 to 2024
CHECK = CFG["years"]["check"]          # 2025: the out-of-sample check year
FORECAST = CFG["years"]["forecast"]    # 2026: the watch list year
X = {y: year_table(y, NODE_COLS) for y in YEARS + [CHECK, FORECAST]}
Y = {y: np.array([int(df[(df.iso3 == iso) & (df.year == y)].ecowas_action.iloc[0])
                  for iso in NODES]) for y in YEARS}

# =============================================================================
# SECTION 2: THE GRAPH NEURAL NETWORK
# =============================================================================


class GNN(torch.nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.conv1 = SAGEConv(n_in, HIDDEN)
        self.conv2 = SAGEConv(HIDDEN, HIDDEN)
        self.out = torch.nn.Linear(HIDDEN, 1)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=G["dropout"], training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        return self.out(h).squeeze(-1)


def fit_predict(train_years, test_graphs, use_edges=True):
    """Train on train_years and return averaged probabilities for each test graph."""
    edges = EDGES if use_edges else NO_EDGES
    # standardize with training years only
    mu = np.vstack([X[y] for y in train_years]).mean(0)
    sd = np.vstack([X[y] for y in train_years]).std(0) + 1e-6
    train = Batch.from_data_list([
        Data(x=torch.tensor((X[y] - mu) / sd, dtype=torch.float),
             y=torch.tensor(Y[y], dtype=torch.float), edge_index=edges) for y in train_years])
    pos = float(train.y.sum())
    pos_weight = torch.tensor((len(train.y) - pos) / pos)
    preds = [np.zeros(len(NODES)) for _ in test_graphs]
    for seed in range(SEEDS):
        torch.manual_seed(seed)
        model = GNN(train.x.shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=G["lr"], weight_decay=G["weight_decay"])
        model.train()
        for _ in range(EPOCHS):
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(train.x, train.edge_index),
                                                      train.y, pos_weight=pos_weight)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            for k, xg in enumerate(test_graphs):
                xt = torch.tensor((xg - mu) / sd, dtype=torch.float)
                preds[k] += torch.sigmoid(model(xt, edges)).numpy() / SEEDS
    return preds

# =============================================================================
# SECTION 3: BASELINE C, REFIT HERE FOR THE PAIRED COMPARISON
# (identical specification and inputs to analysis.py)
# =============================================================================

XC = {y: year_table(y, BASELINE_C_COLS) for y in YEARS}


def baseline_c_predict(train_years, test_year):
    Xtr = np.vstack([XC[y] for y in train_years])
    ytr = np.concatenate([Y[y] for y in train_years])
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(C=BASE["C"], class_weight=BASE["class_weight"], max_iter=BASE["max_iter"]).fit(sc.transform(Xtr), ytr)
    return m.predict_proba(sc.transform(XC[test_year]))[:, 1]

# =============================================================================
# SECTION 4: LEAVE-ONE-YEAR-OUT TEST
# =============================================================================

rows = []
for yr in YEARS:
    train_years = [y for y in YEARS if y != yr]
    p_gnn = fit_predict(train_years, [X[yr]], use_edges=True)[0]
    p_noedge = fit_predict(train_years, [X[yr]], use_edges=False)[0]
    p_c = baseline_c_predict(train_years, yr)
    for i, iso in enumerate(NODES):
        rows.append({"iso3": iso, "year": yr, "action": Y[yr][i],
                     "gnn": p_gnn[i], "gnn_no_edges": p_noedge[i], "baseline_c": p_c[i]})
    print(f"held out {yr} done")
loyo = pd.DataFrame(rows)
loyo.round(4).to_csv(f"{OUT}/gnn_loyo_predictions.csv", index=False)


def top3(d, col):
    hits = 0
    for yr, g in d.groupby("year"):
        r = g[col].rank(ascending=False, method="first")
        hits += int((r[g.action == 1] <= 3).sum())
    return hits


summary = {"settings": {"seeds": SEEDS, "epochs": EPOCHS, "hidden": HIDDEN, "layers": G["layers"],
                        "dropout": G["dropout"], "lr": G["lr"], "weight_decay": G["weight_decay"]}}
summary["loyo"] = {col: {"auc": round(roc_auc_score(loyo.action, loyo[col]), 3),
                         "top3_hits": top3(loyo, col)}
                   for col in ["baseline_c", "gnn", "gnn_no_edges"]}

# Year-clustered bootstrap: resample the 15 held-out years with replacement
rng = np.random.default_rng(B["seed"])
diffs, aucs_g, aucs_c = [], [], []
groups = {y: loyo[loyo.year == y] for y in YEARS}
for _ in range(BOOT):
    sample = pd.concat([groups[y] for y in rng.choice(YEARS, size=len(YEARS), replace=True)])
    if sample.action.nunique() < 2:
        continue
    ag, ac = roc_auc_score(sample.action, sample.gnn), roc_auc_score(sample.action, sample.baseline_c)
    aucs_g.append(ag); aucs_c.append(ac); diffs.append(ag - ac)
q = lambda a: [round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)]
summary["bootstrap"] = {"reps": len(diffs), "gnn_auc_95ci": q(aucs_g), "baseline_c_auc_95ci": q(aucs_c),
                        "difference_95ci": q(diffs),
                        "share_gnn_better": round(float(np.mean(np.array(diffs) > 0)), 3)}

# Which actions each model ranked in its year's top 3
acts = []
for (yr, g) in loyo.groupby("year"):
    for col in ["baseline_c", "gnn"]:
        g = g.assign(**{f"rank_{col}": g[col].rank(ascending=False, method="first").astype(int)})
    for r in g[g.action == 1].itertuples():
        acts.append({"iso3": r.iso3, "year": yr, "rank_baseline_c": r.rank_baseline_c,
                     "rank_gnn": r.rank_gnn, "p_baseline_c": round(r.baseline_c, 3), "p_gnn": round(r.gnn, 3)})
pd.DataFrame(acts).to_csv(f"{OUT}/gnn_action_ranks.csv", index=False)
summary["action_ranks"] = acts

# =============================================================================
# SECTION 5: OUT-OF-SAMPLE CHECK AND CURRENT SCORES (train on the study period)
# Column and file names carry the years, for example gnn_2025 and gnn_2026.
# =============================================================================

p_check, p_now = fit_predict(YEARS, [X[CHECK], X[FORECAST]], use_edges=True)
gc, gf = f"gnn_{CHECK}", f"gnn_{FORECAST}"
rank_c, mrank_f, score_f = f"rank_{CHECK}", f"member_rank_{FORECAST}", f"score_{FORECAST}"
oos = pd.DataFrame({"iso3": NODES, gc: p_check, gf: p_now})
oos[rank_c] = oos[gc].rank(ascending=False, method="first").astype(int)
EXITED = CFG["exited_2025"]
cur = oos[~oos.iso3.isin(EXITED)].copy()
cur[mrank_f] = cur[gf].rank(ascending=False, method="first").astype(int)
oos = oos.merge(cur[["iso3", mrank_f]], on="iso3", how="left")
oos[score_f] = (oos[gf] * 100).round(0).astype(int)
oos.round(4).to_csv(f"{OUT}/gnn_scores_{CHECK}_{FORECAST}.csv", index=False)
summary[f"oos_{CHECK}"] = {"gnb_rank": int(oos[oos.iso3 == "GNB"][rank_c].iloc[0]),
                           "ben_rank": int(oos[oos.iso3 == "BEN"][rank_c].iloc[0]),
                           "top5": oos.sort_values(rank_c).head(5)[["iso3", gc]].round(3).to_dict("records")}
summary[f"current_members_{FORECAST}"] = oos.dropna(subset=[mrank_f]) \
    .sort_values(mrank_f)[["iso3", score_f, mrank_f]].to_dict("records")

with open(f"{OUT}/gnn_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(json.dumps(summary, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
