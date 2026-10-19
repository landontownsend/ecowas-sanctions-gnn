"""
gnn.py
ECOWAS sanctions risk: Stage 2 forecast (graph neural network)

Run from the repository root after analysis.py:  python src/gnn.py
Reads:  data/processed/ecowas_country_year.csv
        data/raw/vdem_ecowas_subset.csv
Writes: results/gnn_*.csv and results/gnn_summary.json

Design (fixed before looking at any held-out result)
  * One graph per year. Nodes: the 15 ECOWAS members. Edges: shared land borders.
  * Node features, all from the prior year: military rule index, successful
    coups, and the country's own successful coups over the previous three years.
    Unlike Baseline C, the network receives NO hand-built neighbor summaries;
    it must learn how to combine neighbors' features itself.
  * Model: 2-layer GraphSAGE (hidden size 16, ReLU, dropout 0.2) and a linear
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
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import SAGEConv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

torch.set_num_threads(4)
DATA = "data"
OUT = "results"
os.makedirs(OUT, exist_ok=True)

SEEDS = 25
EPOCHS = 300
HIDDEN = 16
BOOT = 2000

# =============================================================================
# SECTION 1: DATA, BORDERS, AND FEATURES
# =============================================================================

BORDERS = {
    "BEN": ["NGA", "NER", "BFA", "TGO"], "BFA": ["MLI", "NER", "BEN", "TGO", "GHA", "CIV"],
    "CPV": [], "CIV": ["LBR", "GIN", "MLI", "BFA", "GHA"], "GMB": ["SEN"],
    "GHA": ["CIV", "BFA", "TGO"], "GIN": ["GNB", "SEN", "MLI", "CIV", "LBR", "SLE"],
    "GNB": ["SEN", "GIN"], "LBR": ["SLE", "GIN", "CIV"], "MLI": ["SEN", "GIN", "CIV", "BFA", "NER"],
    "NER": ["MLI", "BFA", "BEN", "NGA"], "NGA": ["BEN", "NER"], "SEN": ["GMB", "GNB", "GIN", "MLI"],
    "SLE": ["GIN", "LBR"], "TGO": ["GHA", "BFA", "BEN"],
}
NODES = sorted(BORDERS)
IDX = {c: i for i, c in enumerate(NODES)}
EDGES = torch.tensor([[IDX[a], IDX[b]] for a in NODES for b in BORDERS[a]], dtype=torch.long).t()
NO_EDGES = torch.empty((2, 0), dtype=torch.long)

df = pd.read_csv(f"{DATA}/processed/ecowas_country_year.csv")
vdem = pd.read_csv(f"{DATA}/raw/vdem_ecowas_subset.csv")
vdem = vdem.rename(columns={"country_text_id": "iso3", "e_pt_coup": "coups",
                            "v2x_ex_military": "military"})
vdem["coups"] = vdem["coups"].fillna(0)   # blank 2025 coup counts mean no coup
v = vdem.set_index(["iso3", "year"])


def coups(iso, y):
    return float(v.loc[(iso, y), "coups"]) if (iso, y) in v.index else 0.0


def node_features(year):
    """Prior-year features for every member, in NODES order (15 x 3)."""
    rows = []
    for iso in NODES:
        rows.append([float(v.loc[(iso, year - 1), "military"]),
                     coups(iso, year - 1),
                     sum(coups(iso, y) for y in range(year - 3, year))])
    return np.array(rows)


YEARS = list(range(2010, 2025))
X = {y: node_features(y) for y in YEARS + [2025, 2026]}
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
        h = F.dropout(h, p=0.2, training=self.training)
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
        opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.005)
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
# SECTION 3: BASELINE C, RECOMPUTED ROW BY ROW FOR THE PAIRED COMPARISON
# (identical specification to analysis.py)
# =============================================================================

def baseline_c_features(year):
    rows = []
    for iso in NODES:
        nb = BORDERS[iso]
        rows.append([float(v.loc[(iso, year - 1), "military"]), coups(iso, year - 1),
                     max([float(v.loc[(n, year - 1), "military"]) for n in nb], default=0.0),
                     sum(coups(n, y) for n in nb for y in range(year - 3, year))])
    return np.array(rows)


XC = {y: baseline_c_features(y) for y in YEARS}


def baseline_c_predict(train_years, test_year):
    Xtr = np.vstack([XC[y] for y in train_years])
    ytr = np.concatenate([Y[y] for y in train_years])
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000).fit(sc.transform(Xtr), ytr)
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


summary = {"settings": {"seeds": SEEDS, "epochs": EPOCHS, "hidden": HIDDEN, "layers": 2,
                        "dropout": 0.2, "lr": 0.01, "weight_decay": 0.005}}
summary["loyo"] = {col: {"auc": round(roc_auc_score(loyo.action, loyo[col]), 3),
                         "top3_hits": top3(loyo, col)}
                   for col in ["baseline_c", "gnn", "gnn_no_edges"]}

# Year-clustered bootstrap: resample the 15 held-out years with replacement
rng = np.random.default_rng(735)
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
# SECTION 5: 2025 OUT-OF-SAMPLE CHECK AND CURRENT SCORES (train on 2010-2024)
# =============================================================================

p25, p26 = fit_predict(YEARS, [X[2025], X[2026]], use_edges=True)
oos = pd.DataFrame({"iso3": NODES, "gnn_2025": p25, "gnn_2026": p26})
oos["rank_2025"] = oos.gnn_2025.rank(ascending=False, method="first").astype(int)
EXITED = ["MLI", "BFA", "NER"]
cur = oos[~oos.iso3.isin(EXITED)].copy()
cur["member_rank_2026"] = cur.gnn_2026.rank(ascending=False, method="first").astype(int)
oos = oos.merge(cur[["iso3", "member_rank_2026"]], on="iso3", how="left")
oos["score_2026"] = (oos.gnn_2026 * 100).round(0).astype(int)
oos.round(4).to_csv(f"{OUT}/gnn_scores_2025_2026.csv", index=False)
summary["oos_2025"] = {"gnb_rank": int(oos[oos.iso3 == "GNB"].rank_2025.iloc[0]),
                       "ben_rank": int(oos[oos.iso3 == "BEN"].rank_2025.iloc[0]),
                       "top5": oos.sort_values("rank_2025").head(5)[["iso3", "gnn_2025"]].round(3).to_dict("records")}
summary["current_members_2026"] = oos.dropna(subset=["member_rank_2026"]) \
    .sort_values("member_rank_2026")[["iso3", "score_2026", "member_rank_2026"]].to_dict("records")

with open(f"{OUT}/gnn_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(json.dumps(summary, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
