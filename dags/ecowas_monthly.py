"""
ecowas_monthly: refresh the data and the watch list once a month.

  update_gdelt -> ingest -> transform -> checks -> export -> baselines -> watch_list -> publish

Each task calls one function in pipeline/tasks.py. If the data checks fail,
the tasks after them do not run, so a bad month never reaches the watch list.
The full GNN run is a separate DAG (ecowas_models), started by hand.

Settings for a manual run (Trigger DAG with config), both on by default:
  live          download the World Bank data from the API (off: use the snapshot)
  update_gdelt  add the newest complete months of GDELT events
"""

import sys
from datetime import timedelta
from pathlib import Path

import pendulum
from airflow.sdk import Param, dag, get_current_context, task

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the repository root
from pipeline import tasks   # noqa: E402

default_args = {"retries": 2, "retry_delay": timedelta(minutes=5)}   # for network hiccups


@dag(
    dag_id="ecowas_monthly",
    schedule="0 6 2 * *",          # 06:00 UTC on the 2nd of each month (last month is complete)
    start_date=pendulum.datetime(2026, 11, 1, tz="UTC"),
    catchup=False,                 # do not run the months before the first start
    max_active_runs=1,             # one run at a time, so one writer for DuckDB
    default_args=default_args,
    params={"live": Param(True, type="boolean"),
            "update_gdelt": Param(True, type="boolean")},
    tags=["ecowas"],
)
def ecowas_monthly():

    @task
    def update_gdelt():
        if get_current_context()["params"]["update_gdelt"]:
            tasks.update_gdelt()
        else:
            print("Skipped: update_gdelt is off for this run")

    @task
    def ingest():
        tasks.ingest(live=get_current_context()["params"]["live"])

    @task
    def transform():
        tasks.transform()

    @task(retries=0)               # a failed check is not a network hiccup: do not retry
    def checks():
        tasks.checks()

    @task
    def export():
        tasks.export()

    @task
    def baselines():
        tasks.baselines()

    @task
    def watch_list():
        tasks.watch_list()

    @task
    def publish():
        run_date = get_current_context()["ds"]     # the run's date, for example "2026-11-02"
        return tasks.publish(run_date)

    update_gdelt() >> ingest() >> transform() >> checks() >> export() >> baselines() >> watch_list() >> publish()


ecowas_monthly()
