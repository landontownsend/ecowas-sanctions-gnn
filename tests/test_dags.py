"""
The Airflow DAGs load without errors and have their tasks in the right order.
Skipped when Airflow is not installed (CI installs it in a separate job).
"""

import pytest

pytest.importorskip("airflow.sdk")
from airflow.dag_processing.dagbag import DagBag   # noqa: E402


@pytest.fixture(scope="module")
def dagbag():
    return DagBag(dag_folder="dags")   # examples are off with AIRFLOW__CORE__LOAD_EXAMPLES=false


def chain(dag, first):
    """Follow the tasks from the first one: [first, next, next, ...]."""
    order, current = [first], dag.get_task(first)
    while current.downstream_list:
        assert len(current.downstream_list) == 1, "each task should lead to one task"
        current = current.downstream_list[0]
        order.append(current.task_id)
    return order


def test_no_import_errors(dagbag):
    assert dagbag.import_errors == {}


def test_monthly_dag_order_and_schedule(dagbag):
    dag = dagbag.dags["ecowas_monthly"]          # parsed DAGs (no database needed)
    assert chain(dag, "update_gdelt") == ["update_gdelt", "ingest", "transform", "checks",
                                          "export", "baselines", "watch_list", "publish"]
    assert dag.max_active_runs == 1
    assert not dag.catchup


def test_models_dag_runs_only_by_hand(dagbag):
    dag = dagbag.dags["ecowas_models"]
    assert chain(dag, "gnn_full") == ["gnn_full", "figures", "publish_models"]
    assert type(dag.timetable).__name__ == "NullTimetable"    # schedule=None: no automatic runs
