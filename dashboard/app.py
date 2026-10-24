"""
ECOWAS sanctions risk dashboard (Streamlit).

Run from the repository root:  streamlit run dashboard/app.py
Hosted on Streamlit Community Cloud (main file: dashboard/app.py).

The app reads files that are committed to the repository, so it needs no
database or download. When it runs next to Airflow (docker-compose.yaml sets
ECOWAS_DATA_DIR), it shows the newest dated run from the monthly DAG instead,
and the data loads recorded in the warehouse.

Tabs
  1. Watch list       risk map and table for the forecast year
  2. Country trends   monthly GDELT protest and violence shares, with ECOWAS actions and coups
  3. Economic impact  import change after each action
  4. Models           how the models compare on held-out years
  5. Data             sources, freshness, and the pipeline
"""

import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[1]          # the repository root
DATA_DIR = os.environ.get("ECOWAS_DATA_DIR")         # set by docker-compose.yaml
REPO_URL = "https://github.com/landontownsend/ecowas-sanctions-gnn"

# Colors shared with the figures in figures/
BG, INK, MUTED = "#F6F4EF", "#17233A", "#7A8090"
ACCENT, BLUE = "#B5472A", "#2A6FB0"
TIER_COLORS = {"High": "#A9422A", "Elevated": "#DE8A5E", "Lower": "#F2C9AC",
               "Left ECOWAS": "#CFCBC3", "Not a member": "#E8E5DE"}

st.set_page_config(page_title="ECOWAS Sanctions Risk", page_icon="🌍", layout="wide")

# =============================================================================
# DATA
# =============================================================================


def path(relative):
    """A file in the repository."""
    return ROOT / relative


@st.cache_data
def load_config():
    return yaml.safe_load(open(path("config.yaml")))


CFG = load_config()
NAMES = {c["iso3"]: c["name"].replace("Cote d'Ivoire", "Côte d'Ivoire") for c in CFG["countries"]}
FORECAST = CFG["years"]["forecast"]


def latest_airflow_run():
    """The newest dated folder written by the Airflow publish task, or None."""
    if not DATA_DIR:
        return None
    runs = Path(DATA_DIR) / "runs"
    if not runs.exists():
        return None
    folders = sorted(p for p in runs.iterdir() if p.is_dir() and not p.name.startswith("models-"))
    return folders[-1] if folders else None


@st.cache_data
def load_watch_list(run_folder):
    """The main model's watch list: from the newest Airflow run, or the committed results."""
    file = (Path(run_folder) if run_folder else path("results")) / "risk_next12_v2.csv"
    df = pd.read_csv(file)
    df["country"] = df.iso3.map(NAMES)
    return df


@st.cache_data
def load_csv(relative):
    return pd.read_csv(path(relative))


@st.cache_data
def load_json(relative):
    return json.load(open(path(relative)))


@st.cache_data
def load_geo():
    return json.load(open(path("data/geo/west_africa.geojson")))


def risk_map(watch, geo):
    """
    The risk map, drawn as filled shapes on a plain chart, so it needs no map
    tiles or outside files. Members are colored by tier; neighbors are grey.
    Hovering over a country shows its score, tier, and rank.
    """
    info = watch.set_index("iso3")
    fig = go.Figure()
    for feature in geo["features"]:
        iso = feature["properties"]["iso3"]
        if iso in info.index:
            r = info.loc[iso]
            tier = r.tier
            rank = "not ranked" if pd.isna(r.member_rank) else f"member rank {int(r.member_rank)}"
            text = f"<b>{r.country}</b><br>Score {r.risk_score_0_100} · {tier} · {rank}"
            if r.context:
                text += f"<br>{r.context}"
        else:
            tier, text = "Not a member", feature["properties"]["name"]
        geom = feature["geometry"]
        polygons = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for polygon in polygons:
            ring = polygon[0]                                  # the outer edge of the shape
            fig.add_trace(go.Scatter(
                x=[pt[0] for pt in ring], y=[pt[1] for pt in ring], mode="lines",
                fill="toself", fillcolor=TIER_COLORS[tier], line=dict(color="white", width=1),
                hoveron="fills", hoverinfo="text", text=text, showlegend=False))
    for tier in ["High", "Elevated", "Lower", "Left ECOWAS"]:   # legend entries
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=tier,
                                 marker=dict(size=14, color=TIER_COLORS[tier], symbol="square")))
    fig.update_xaxes(visible=False, range=[-25.5, 16])
    fig.update_yaxes(visible=False, range=[3.5, 25], scaleanchor="x", scaleratio=1)
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor=BG,
                      plot_bgcolor=BG, legend=dict(orientation="h", y=0, x=0), hovermode="closest")
    return fig


run_folder = latest_airflow_run()
watch = load_watch_list(str(run_folder) if run_folder else None)
summary = load_json("results/summary.json")
gnn = load_json("results/gnn_summary.json")
monthly = load_csv("data/processed/ecowas_country_month.csv")
yearly = load_csv("data/processed/ecowas_country_year.csv")
impact = load_csv("results/impact.csv")
geo = load_geo()

# =============================================================================
# HEADER
# =============================================================================

st.title("ECOWAS sanctions risk")
st.markdown(
    "Which West African governments are most likely to face suspension, sanctions, or a "
    "border closure from the regional bloc ECOWAS in the next 12 months? Scores come from "
    "a model trained on 2001 to 2024 and tested on every year it did not see. "
    f"[Code and method]({REPO_URL})."
)
source = (f"Newest monthly Airflow run: {run_folder.name}" if run_folder
          else "Committed results in the repository")
main_model = gnn["decision"]["main_model"]
c1, c2, c3 = st.columns(3)
c1.metric("Forecast year", FORECAST)
c2.metric("Main model", main_model)
c3.metric("Held-out AUC", f"{gnn['loyo']['baseline_c_v2']['auc']:.2f}" if main_model == "Baseline C v2"
          else f"{gnn['loyo']['gnn']['auc']:.2f}")
st.caption(f"Source: {source}.")

tab_watch, tab_trends, tab_impact, tab_models, tab_data = st.tabs(
    ["Watch list", "Country trends", "Economic impact", "Models", "Data"])

# =============================================================================
# 1. WATCH LIST
# =============================================================================

with tab_watch:
    st.subheader(f"Watch list for {FORECAST}")
    st.markdown(
        "The score ranks members from 0 to 100. It is a ranking, not a probability. "
        "Mali, Burkina Faso, and Niger left ECOWAS in January 2025, so they are scored "
        "but not ranked."
    )

    fig = risk_map(watch, geo)
    left, right = st.columns([3, 2])
    left.plotly_chart(fig, width="stretch")

    table = watch[["country", "risk_score_0_100", "tier", "member_rank", "status", "context"]].copy()
    table["member_rank"] = ["" if pd.isna(v) else str(int(v)) for v in table.member_rank]   # blank: left ECOWAS
    table = table.rename(columns={"country": "Country", "risk_score_0_100": "Score",
                                  "tier": "Tier", "member_rank": "Member rank",
                                  "status": "Status", "context": "Context"})
    right.dataframe(table, hide_index=True, width="stretch", height=520)

# =============================================================================
# 2. COUNTRY TRENDS
# =============================================================================

with tab_trends:
    st.subheader("Protest and violence in the news, month by month")
    st.markdown(
        "Share of each country's GDELT news events that are protests or violence (assault, "
        "fighting, mass violence), averaged over the last 3 months. Dots mark ECOWAS actions; "
        "dotted lines mark the years with a successful coup."
    )
    options = sorted(NAMES, key=lambda iso: NAMES[iso])
    default = options.index("MLI")
    iso = st.selectbox("Country", options, index=default, format_func=lambda i: NAMES[i])

    m = monthly[monthly.iso3 == iso].copy()
    m["date"] = pd.to_datetime(dict(year=m.year, month=m.month, day=1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=m.date, y=m.gdelt_protest_share_3m * 100, name="Protest share (%)",
                             line=dict(color=ACCENT, width=2)))
    fig.add_trace(go.Scatter(x=m.date, y=m.gdelt_violence_share_3m * 100, name="Violence share (%)",
                             line=dict(color=BLUE, width=2)))
    actions = m[m.action_this_month == 1]
    fig.add_trace(go.Scatter(x=actions.date, y=[0] * len(actions), mode="markers", name="ECOWAS action",
                             marker=dict(color=INK, size=12, symbol="circle")))
    for year in yearly[(yearly.iso3 == iso) & (yearly.coups_successful > 0)].year:
        fig.add_vline(x=pd.Timestamp(year=int(year), month=7, day=1), line_dash="dot", line_color=MUTED)
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor=BG, plot_bgcolor=BG,
                      yaxis_title="Share of events (%)", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width="stretch")

# =============================================================================
# 3. ECONOMIC IMPACT
# =============================================================================

with tab_impact:
    st.subheader("Imports after an ECOWAS action")
    st.markdown(
        "Change in real imports from the year before an action to the average of the action year "
        "and the next year, compared with members facing no action that year (percentage points). "
        f"In {summary['impact_actions_below_region']} of {len(impact)} actions, imports fell behind "
        "the region. This is an association, not a measured cause."
    )
    imp = impact.dropna(subset=["excess_import_change"]).copy()
    imp["label"] = imp.iso3.map(NAMES) + " " + imp.year.astype(str)
    imp = imp.sort_values("excess_import_change")
    fig = px.bar(imp, x="excess_import_change", y="label", orientation="h",
                 color=imp.excess_import_change < 0,
                 color_discrete_map={True: ACCENT, False: MUTED},
                 labels={"excess_import_change": "Percentage points vs. the region", "label": ""})
    fig.update_layout(height=520, showlegend=False, margin=dict(l=0, r=0, t=10, b=0),
                      paper_bgcolor=BG, plot_bgcolor=BG, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch")

# =============================================================================
# 4. MODELS
# =============================================================================

with tab_models:
    st.subheader("How the models compare on years they did not see")
    st.markdown(
        "Each model was trained on 23 of the 24 years from 2001 to 2024 and scored on the year "
        "it did not see, 24 times. **AUC** is the share of action and no-action pairs the model "
        "ranks in the right order (0.5 is a coin flip). **Top 3** counts the actions that ranked "
        "in the top 3 of their year. The design and the decision rule were written down before the run."
    )
    base = {r["model"]: r for r in summary["model_comparison"]}
    actions_n = gnn["actions"]
    rows = [
        ("Baseline A: own conditions", base["Own conditions only"]),
        ("Baseline B: + region-wide coups", base["Own + region-wide coups"]),
        ("Baseline C: + border neighbors", base["Own + border neighbors (network)"]),
        ("Baseline C v2: 11 inputs, tuned", gnn["loyo"]["baseline_c_v2"]),
        ("Graph neural network v2", gnn["loyo"]["gnn"]),
        ("GNN v2, links removed", gnn["loyo"]["gnn_no_edges"]),
    ]
    comp = pd.DataFrame([{"Model": name, "AUC": r["auc"], f"Top 3 (of {actions_n})": r["top3_hits"]}
                         for name, r in rows])
    st.dataframe(comp, hide_index=True, width="stretch",
                 column_config={"AUC": st.column_config.NumberColumn(format="%.2f")})
    b = gnn["bootstrap"]
    st.markdown(
        f"**Decision:** {main_model} is the main model. The 95% interval for the difference "
        f"(GNN v2 minus Baseline C v2) is {b['difference_95ci'][0]:+.2f} to {b['difference_95ci'][1]:+.2f}, "
        f"and the GNN is ahead in {b['share_gnn_better']:.1%} of {b['reps']:,} resamples. "
        "Within a year the GNN ranks countries about as well, but with 11 inputs and 13 actions "
        "it overfits, so its scores are not comparable from one year to the next."
    )

# =============================================================================
# 5. DATA
# =============================================================================

with tab_data:
    st.subheader("Data and pipeline")
    gdelt_manifest = load_json("data/raw/gdelt_country_month.json")
    gdelt_last = monthly[["year", "month"]].max()
    st.markdown(f"""
* **V-Dem v16** (coups, military rule, mass mobilization), from the official release.
* **World Bank** World Development Indicators (GDP growth, inflation, imports), via the API.
* **GDELT 1.0** news events: {gdelt_manifest['rows_read']:,} rows from {gdelt_manifest['files']:,} files,
  counted by country and month; snapshot built {gdelt_manifest['built']}.
* **ECOWAS actions**, hand-coded from communiqués and news coverage.

The pipeline loads every source into a DuckDB warehouse, builds the features with SQL, runs 23
data checks, and exports the tables the models read. Apache Airflow runs it on the 2nd of each
month. See the [README]({REPO_URL}#readme) and the
[design document]({REPO_URL}/blob/main/docs/project-context.md).
""")
    if DATA_DIR and (Path(DATA_DIR) / "ecowas.duckdb").exists():
        import duckdb
        try:
            with duckdb.connect(str(Path(DATA_DIR) / "ecowas.duckdb"), read_only=True) as con:
                loads = con.execute("SELECT table_name, source, n_rows, loaded_at FROM meta.load_log "
                                    "ORDER BY loaded_at DESC LIMIT 10").df()
            st.markdown("**Latest loads in the warehouse**")
            st.dataframe(loads, hide_index=True, width="stretch")
        except duckdb.Error:
            st.caption("The warehouse is busy (a DAG is writing to it); reload in a minute.")
