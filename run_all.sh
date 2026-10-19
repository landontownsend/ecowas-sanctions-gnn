#!/usr/bin/env bash
# Runs the whole project from the repository root, in order.
# Pass --live to download the latest World Bank data instead of the saved snapshot.
set -euo pipefail

python -m pipeline.run "$@"      # data: ingest, transform, checks, export
python src/analysis.py           # descriptive results, import damage, Baselines A to C
python src/gnn.py                # graph neural network (about 8 minutes on a laptop CPU)
python src/charts.py             # figures 1 to 3
python src/gnn_chart.py          # figure 4
