"""
gnn_chart.py
Figure 4: held-out AUC for every model in version 2: Baselines A to C (version 1
specifications), Baseline C v2, the graph neural network (GNN v2), and the GNN
with its edges removed. Whiskers are 95% year-clustered bootstrap intervals
(Baseline C v2 and GNN v2, the primary comparison).
Run from the repository root, after analysis.py and gnn.py:  python src/gnn_chart.py
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, INK, BODY, MUTED, GRID = "#F6F4EF", "#17233A", "#4A5263", "#7A8090", "#DCD8CF"
ACCENT, BLUE, GREY = "#B5472A", "#2A6FB0", "#B9B4AA"
plt.rcParams.update({"font.family": "Inter", "figure.facecolor": BG, "axes.facecolor": BG,
                     "savefig.facecolor": BG, "axes.edgecolor": GRID})

base = {m["model"]: m for m in json.load(open("results/summary.json"))["model_comparison"]}
g = json.load(open("results/gnn_summary.json"))

ACTIONS = g["actions"]
rows = [  # label, auc, top-3 hits, color, interval
    ("Baseline A · own conditions", base["Own conditions only"]["auc"], base["Own conditions only"]["top3_hits"], GREY, None),
    ("Baseline B · + region-wide coups", base["Own + region-wide coups"]["auc"], base["Own + region-wide coups"]["top3_hits"], GREY, None),
    ("Baseline C · + border neighbors", base["Own + border neighbors (network)"]["auc"], base["Own + border neighbors (network)"]["top3_hits"], GREY, None),
    ("Baseline C v2 · 11 inputs, tuned", g["loyo"]["baseline_c_v2"]["auc"], g["loyo"]["baseline_c_v2"]["top3_hits"], ACCENT, g["bootstrap"]["baseline_c_v2_auc_95ci"]),
    ("Graph neural network v2", g["loyo"]["gnn"]["auc"], g["loyo"]["gnn"]["top3_hits"], BLUE, g["bootstrap"]["gnn_auc_95ci"]),
    ("GNN v2, all links removed", g["loyo"]["gnn_no_edges"]["auc"], g["loyo"]["gnn_no_edges"]["top3_hits"], "#9DBBDA", None),
]

os.makedirs("figures", exist_ok=True)
fig, ax = plt.subplots(figsize=(14.5, 7.6), dpi=150)
ys = list(range(len(rows)))[::-1]
for y, (lab, auc, hits, col, ci) in zip(ys, rows):
    ax.barh(y, auc - 0.5, left=0.5, height=0.56, color=col, zorder=2)
    if ci:
        ax.plot(ci, [y, y], color=INK, lw=2, zorder=3)
        ax.plot([ci[0]] * 2, [y - 0.13, y + 0.13], color=INK, lw=2, zorder=3)
        ax.plot([ci[1]] * 2, [y - 0.13, y + 0.13], color=INK, lw=2, zorder=3)
    x_text = max(auc, ci[1] if ci else auc) + 0.012
    ax.text(x_text, y, f"{round((auc + 1e-9) * 100):.0f}%  ·  {hits} of {ACTIONS} in top 3", va="center", fontsize=19, color=INK)
ax.set_yticks(ys)
ax.set_yticklabels([r[0] for r in rows], fontsize=20, color=INK)
ax.set_xlim(0.5, 1.17)
ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_xticklabels(["50%\ncoin flip", "60%", "70%", "80%", "90%", "100%\nperfect"], fontsize=17, color=BODY)
ax.xaxis.grid(True, color=GRID, lw=1, zorder=0)
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.set_xlabel("Share of rankings correct on held-out years (lines show the plausible range)", fontsize=18, color=BODY, labelpad=10)
plt.tight_layout()
fig.savefig("figures/fig4_model_comparison.png", dpi=150)
print("ok")
