"""
ecowas_models: the full model run, started by hand (no schedule).

  gnn_full -> figures -> publish_models

gnn_full runs Baseline C v2 and GNN v2 with nested tuning (20 to 40 minutes on
4 cores). Run ecowas_monthly first, so the feature tables are current.
"""

import sys
from pathlib import Path

import pendulum
from airflow.sdk import dag, get_current_context, task

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the repository root
from pipeline import tasks   # noqa: E402


@dag(
    dag_id="ecowas_models",
    schedule=None,
    start_date=pendulum.datetime(2026, 10, 1, tz="UTC"),   # in the past, so manual runs work now
    catchup=False,
    max_active_runs=1,
    tags=["ecowas"],
)
def ecowas_models():

    @task(execution_timeout=pendulum.duration(hours=3))
    def gnn_full():
        tasks.gnn_full()

    @task
    def figures():
        tasks.figures()

    @task
    def publish_models():
        return tasks.publish_models(get_current_context().get("ds"))

    gnn_full() >> figures() >> publish_models()


ecowas_models()
