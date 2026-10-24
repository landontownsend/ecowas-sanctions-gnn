"""The publish step saves a dated copy and describes what changed."""

import pandas as pd

from pipeline import tasks


def watch(rows):
    return pd.DataFrame(rows, columns=["iso3", "risk_score_0_100", "tier", "member_rank"])


def test_compare_finds_changes():
    old = watch([("GIN", 74, "Elevated", 2), ("GNB", 85, "Elevated", 1)])
    new = watch([("GIN", 88, "High", 1), ("GNB", 85, "Elevated", 2)])
    note = tasks.compare_watch_lists(old, new, "2026-11-02", "2026-12-02")
    assert "- GIN: score 74 -> 88, tier Elevated -> High, member rank 2 -> 1" in note
    assert "- GNB: score 85 -> 85, tier Elevated -> Elevated, member rank 1 -> 2" in note


def test_compare_with_no_changes():
    old = watch([("GIN", 74, "Elevated", 2), ("MLI", 83, "Left ECOWAS", None)])
    note = tasks.compare_watch_lists(old, old.copy(), "2026-11-02", "2026-12-02")
    assert "No changes" in note


def test_publish_writes_a_dated_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("ECOWAS_DATA_DIR", str(tmp_path))
    first = tasks.publish("2026-11-02")
    second = tasks.publish("2026-12-02")
    assert (tmp_path / "runs" / "2026-11-02" / "risk_next12_v2.csv").exists()
    assert "First saved run" in (tmp_path / "runs" / "2026-11-02" / "changes.md").read_text()
    assert "No changes" in (tmp_path / "runs" / "2026-12-02" / "changes.md").read_text()
    assert first.endswith("2026-11-02") and second.endswith("2026-12-02")
