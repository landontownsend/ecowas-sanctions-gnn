"""The watch list rules (src/watchlist.py) are the same for every model."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from watchlist import build_watch_list


def test_tiers_status_and_member_rank(cfg):
    scores = pd.DataFrame({"iso3": ["NER", "GNB", "CIV", "SEN"],
                           "risk": [0.99, 0.90, 0.80, 0.50], "rank": [1, 2, 3, 4]})
    watch = build_watch_list(scores, cfg).set_index("iso3")
    assert watch.loc["NER", "tier"] == "Left ECOWAS"           # left in 2025: not ranked
    assert pd.isna(watch.loc["NER", "member_rank"])
    assert watch.loc["GNB", "status"] == "Suspended (Nov 2025)"
    assert watch.loc["GNB", "tier"] == "High"                  # 0.90 >= 0.85
    assert watch.loc["GNB", "member_rank"] == 1
    assert watch.loc["CIV", "tier"] == "Elevated"              # 0.70 <= 0.80 < 0.85
    assert watch.loc["SEN", "tier"] == "Lower"                 # below 0.70
    assert watch.loc["SEN", "risk_score_0_100"] == 50
