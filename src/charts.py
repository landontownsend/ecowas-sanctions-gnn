"""
charts.py
Builds the presentation figures from results/ and the dataset.
Run from the repository root, after analysis.py:  python src/charts.py
The map shapes are Natural Earth (public domain). They are downloaded once to
data/external/ on the first run.
Writes figures/*.png at slide resolution.
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

# Deck colors (shared with the slides)
BG = "#F6F4EF"
INK = "#17233A"
BODY = "#4A5263"
MUTED = "#7A8090"
GRID = "#DCD8CF"
ACCENT = "#B5472A"     # terracotta: ECOWAS action / risk
BLUE = "#2A6FB0"       # secondary: coups
SEV = {1: "#E7A07A", 2: "#CC6B45", 3: "#9E3A20"}   # ordinal: suspension, sanctions, closure
TIER = {"Lower": "#F2C9AC", "Elevated": "#DE8A5E", "High": "#A9422A"}
EXITED = "#CFCBC3"
OTHER = "#E8E5DE"

plt.rcParams.update({
    "hatch.color": "#9C978D",
    "font.family": "Inter",
    "font.size": 20,
    "axes.edgecolor": GRID,
    "axes.labelcolor": BODY,
    "xtick.color": BODY,
    "ytick.color": BODY,
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
})

df = pd.read_csv("data/processed/ecowas_country_year.csv")
impact = pd.read_csv("results/impact.csv")
risk = pd.read_csv("results/risk_next12.csv")

NAMES = {"BEN": "Benin", "BFA": "Burkina Faso", "CPV": "Cabo Verde", "CIV": "Côte d'Ivoire",
         "GMB": "Gambia", "GHA": "Ghana", "GIN": "Guinea", "GNB": "Guinea-Bissau",
         "LBR": "Liberia", "MLI": "Mali", "NER": "Niger", "NGA": "Nigeria", "SEN": "Senegal",
         "SLE": "Sierra Leone", "TGO": "Togo"}
SEV_LABEL = {1: "Suspension", 2: "Sanctions", 3: "Border closure"}

# =============================================================================
# FIGURE 1: TIMELINE OF COUPS AND ECOWAS ACTIONS (dot matrix)
# Rows: the six countries with a coup or an action. Columns: years.
# Blue ring = successful coup. Terracotta dot = ECOWAS action, darker = more severe.
# =============================================================================

rows = ["MLI", "GIN", "BFA", "NER", "GNB", "CIV"]
fig, ax = plt.subplots(figsize=(16, 6.6), dpi=150)
years = list(range(2010, 2025))
for i, iso in enumerate(rows):
    y = len(rows) - 1 - i
    ax.plot([2009.6, 2024.4], [y, y], color=GRID, lw=1, zorder=0)
    sub = df[df.iso3 == iso]
    for r in sub.itertuples():
        if r.action_severity > 0:
            ax.scatter(r.year, y, s=560, color=SEV[r.action_severity], zorder=3,
                       edgecolor=BG, linewidth=2)
        if r.coups_successful > 0:
            ax.scatter(r.year, y, s=1150, facecolor="none", edgecolor=BLUE, linewidth=3, zorder=4)
        elif r.coups_failed > 0:
            ax.scatter(r.year, y, s=1150, facecolor="none", edgecolor=BLUE, linewidth=2,
                       linestyle=(0, (2, 2)), zorder=4)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([NAMES[c] for c in reversed(rows)], fontsize=22, color=INK)
ax.set_xticks(years)
ax.set_xticklabels([str(y) if y % 2 == 0 else "" for y in years], fontsize=20)
ax.set_xlim(2009.4, 2024.6)
ax.set_ylim(-0.7, len(rows) - 0.3)
ax.axvspan(2019.5, 2023.5, color="#ECE6DA", zorder=-1)
ax.text(2021.5, len(rows) - 0.45, "2020 to 2023 coup wave", ha="center", va="bottom",
        fontsize=19, color=BODY)
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
handles = [Line2D([], [], marker="o", ls="", ms=17, color=SEV[k], label=SEV_LABEL[k]) for k in (1, 2, 3)]
handles += [Line2D([], [], marker="o", ls="", ms=24, mfc="none", mec=BLUE, mew=3, label="Successful coup"),
            Line2D([], [], marker="o", ls="", ms=24, mfc="none", mec=BLUE, mew=2, ls_=None) if False else
            Line2D([], [], marker="o", ls="", ms=24, mfc="none", mec=BLUE, mew=1.5, label="Failed attempt (dotted ring)")]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=5,
          frameon=False, fontsize=19, handletextpad=0.4, columnspacing=1.6)
plt.tight_layout()
fig.savefig(f"{OUT}/fig1_timeline.png", dpi=150)
plt.close(fig)

# =============================================================================
# FIGURE 2: IMPORT CHANGE VS. UNAFFECTED NEIGHBORS (diverging bars from zero)
# =============================================================================

imp = impact.sort_values("excess_import_change", ascending=True).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(15, 7.4), dpi=150)
labels = [f"{NAMES[r.iso3]} {r.year}" for r in imp.itertuples()]
vals = imp.excess_import_change.values
colors = [ACCENT if v < 0 else MUTED for v in vals]
ypos = np.arange(len(imp))[::-1]
ax.barh(ypos, vals, color=colors, height=0.62, zorder=2)
for y, v, r in zip(ypos, vals, imp.itertuples()):
    txt = "0 pts" if abs(v) < 0.5 else f"{v:+.0f} pts".replace("-", "−")
    ax.text(v - 0.7 if v < 0 else v + 0.7, y, txt, va="center",
            ha="right" if v < 0 else "left", fontsize=19, color=INK)
    ax.text(1.2 if v < 0 else -1.2, y, SEV_LABEL[r.action_severity], va="center",
            ha="left" if v < 0 else "right", fontsize=17, color=MUTED)
ax.axvline(0, color=INK, lw=1.5, zorder=3)
ax.set_yticks(ypos)
ax.set_yticklabels(labels, fontsize=21, color=INK)
ax.set_xlim(-36, 16)
ax.xaxis.grid(True, color=GRID, lw=1, zorder=0)
ax.set_xlabel("Import change vs. members with no action that year (percentage points)",
              fontsize=19, color=BODY, labelpad=12)
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
plt.tight_layout()
fig.savefig(f"{OUT}/fig2_impact.png", dpi=150)
plt.close(fig)

# =============================================================================
# FIGURE 3: RISK MAP, NEXT 12 MONTHS
# Current members are shaded by risk tier (relative score from the network
# model). Mali, Burkina Faso and Niger are hatched: they left ECOWAS in 2025.
# =============================================================================

GEO_URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
           "master/geojson/ne_50m_admin_0_countries.geojson")
GEO_PATH = "data/external/ne_50m_admin_0_countries.geojson"
if not os.path.exists(GEO_PATH):
    import requests
    os.makedirs(os.path.dirname(GEO_PATH), exist_ok=True)
    with open(GEO_PATH, "wb") as fh:
        fh.write(requests.get(GEO_URL, timeout=120).content)
geo = json.load(open(GEO_PATH))
tier = dict(zip(risk.iso3, risk.tier))
score = dict(zip(risk.iso3, risk.risk_score_0_100))
fig, ax = plt.subplots(figsize=(13.5, 9.2), dpi=150)
ax.set_facecolor("#E4EEF3")   # sea
fig.patch.set_facecolor(BG)


def rings(geom):
    if geom["type"] == "Polygon":
        return [geom["coordinates"][0]]
    return [p[0] for p in geom["coordinates"]]


centroids = {}
for f in geo["features"]:
    iso = f["properties"]["ADM0_A3"]
    if iso == "SDS":
        continue
    polys = [np.array(r) for r in rings(f["geometry"])]
    if iso in tier:
        t = tier[iso]
        if t == "Left ECOWAS":
            fc, hatch = EXITED, "///"
        else:
            fc, hatch = TIER[t], None
        ec, lw = BG, 1.6
        big = max(polys, key=lambda p: len(p))
        centroids[iso] = big.mean(axis=0)
    else:
        fc, hatch, ec, lw = OTHER, None, "#FFFFFF", 0.8
    for p in polys:
        ax.add_patch(Polygon(p, closed=True, facecolor=fc, edgecolor=ec, lw=lw,
                             hatch=hatch, zorder=2 if iso in tier else 1))


# Label positions in longitude/latitude, chosen by hand. Small coastal
# countries get a label in the ocean with a leader line to the country.
INSIDE = {"MLI": (-3.6, 17.6), "NER": (9.4, 17.2), "BFA": (-1.6, 12.4), "NGA": (8.2, 9.4),
          "GHA": (-1.15, 8.1), "CIV": (-5.6, 7.9), "GIN": (-10.6, 10.7), "SEN": (-14.5, 15.0)}
OCEAN = {  # label point, then the point on the country the line goes to
    "CPV": ((-23.6, 19.0), (-23.6, 16.9)),
    "GMB": ((-21.2, 13.1), (-15.6, 13.45)),
    "GNB": ((-21.2, 11.2), (-15.2, 11.95)),
    "SLE": ((-17.6, 8.0), (-12.2, 8.6)),
    "LBR": ((-13.6, 4.1), (-9.6, 6.4)),
    "TGO": ((-1.4, 3.6), (1.05, 6.9)),
    "BEN": ((3.6, 3.6), (2.3, 7.3)),
}
def label_text(iso):
    t = tier[iso]
    star = "*" if iso == "GIN" else ""
    return NAMES[iso], ("left ECOWAS" if t == "Left ECOWAS" else f"{t} risk{star}")
for iso, (x, y) in INSIDE.items():
    name, sub = label_text(iso)
    light = tier[iso] == "High"
    ax.text(x, y + 0.15, name, ha="center", va="bottom", fontsize=17, weight="semibold",
            color="#FFFFFF" if light else INK, zorder=5)
    ax.text(x, y + 0.05, sub, ha="center", va="top", fontsize=15,
            color="#FBEDE6" if light else BODY, zorder=5)
for iso, ((lx, ly), (px, py)) in OCEAN.items():
    name, sub = label_text(iso)
    ax.plot([lx, px], [ly + 0.1, py], color=INK, lw=1, zorder=4)
    ax.scatter([px], [py], s=14, color=INK, zorder=5)
    ax.text(lx, ly + 0.15, name, ha="center", va="bottom", fontsize=17, weight="semibold",
            color=INK, zorder=5, bbox=dict(boxstyle="square,pad=0.1", fc=BG, ec="none"))
    ax.text(lx, ly + 0.05, sub, ha="center", va="top", fontsize=15, color=BODY, zorder=5,
            bbox=dict(boxstyle="square,pad=0.1", fc=BG, ec="none"))

ax.set_xlim(-26.5, 16.5)
ax.set_ylim(2.3, 24.5)
ax.set_aspect("equal")
ax.axis("off")
leg = [Line2D([], [], marker="s", ls="", ms=22, color=TIER["High"], label="High risk"),
       Line2D([], [], marker="s", ls="", ms=22, color=TIER["Elevated"], label="Elevated"),
       Line2D([], [], marker="s", ls="", ms=22, color=TIER["Lower"], label="Lower"),
       Line2D([], [], marker="s", ls="", ms=22, color=EXITED, label="Left ECOWAS (2025)")]
ax.legend(handles=leg, loc="lower left", frameon=False, fontsize=17, bbox_to_anchor=(0.0, 0.0))
plt.tight_layout(pad=0.2)
fig.savefig(f"{OUT}/fig3_risk_map.png", dpi=150, bbox_inches="tight", pad_inches=0.1)
plt.close(fig)
print("figures written")
