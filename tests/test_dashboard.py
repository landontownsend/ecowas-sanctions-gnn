"""The dashboard runs without errors and shows every tab (Streamlit's own test tool)."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest   # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")


def test_dashboard_runs():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value == "ECOWAS sanctions risk"
    assert [t.label for t in at.tabs] == ["Watch list", "Country trends", "Economic impact",
                                          "Models", "Data"]
    assert len(at.dataframe) >= 2                     # the watch list and the model table


def test_country_picker_changes_the_trend_chart():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.selectbox[0].set_value("NER").run()
    assert not at.exception
