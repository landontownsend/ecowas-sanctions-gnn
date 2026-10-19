"""
Calls the real World Bank API. Skipped unless RUN_LIVE_API=1 is set, because
some networks block the API. GitHub Actions sets it.
"""

import os

import pytest

from pipeline.ingest import fetch_world_bank

live = pytest.mark.skipif(os.environ.get("RUN_LIVE_API") != "1",
                          reason="set RUN_LIVE_API=1 to call the World Bank API")


@live
def test_world_bank_api_returns_every_member_and_indicator(cfg):
    long = fetch_world_bank(cfg)
    assert long.iso3.nunique() == len(cfg["countries"])
    assert set(long.variable) == set(cfg["sources"]["world_bank"]["indicators"].values())
    imports = long[long.variable == "imports_real"]
    assert (imports.value > 0).all()
