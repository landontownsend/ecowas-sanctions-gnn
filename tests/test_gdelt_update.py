"""The monthly GDELT update adds only the new complete months (fake files, no internet)."""

import io
import json
import sys
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import refresh_gdelt as rg   # noqa: E402


def fake_file(rows):
    """A zipped file in GDELT's tab-separated layout (58 columns)."""
    lines = []
    for day, root, fips in rows:
        cols = ["1", day] + ["x"] * 26 + [root] + ["y"] * 22 + [fips] + ["z"] * 6
        lines.append("\t".join(cols))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("day.export.CSV", "\n".join(lines) + "\n")
    return buf.getvalue()


FILES = {
    "20260115.export.CSV.zip": fake_file([("20260115", "14", "NI"), ("20251231", "14", "NI")]),
    "20260210.export.CSV.zip": fake_file([("20260210", "19", "NG"), ("20260210", "01", "US")]),
    "20260301.export.CSV.zip": fake_file([("20260301", "14", "NI")]),      # March: not complete yet
}


def test_update_adds_complete_months_only(tmp_path, monkeypatch):
    snapshot = tmp_path / "gdelt.csv"
    old = rg.month_grid([(2025, 12)]).assign(events_total=5, events_protest=1, events_violence=0)
    old.to_csv(snapshot, index=False)
    monkeypatch.setitem(rg.G, "snapshot", str(snapshot))
    monkeypatch.setattr(rg, "all_files", lambda: sorted(FILES))
    monkeypatch.setattr(rg, "download", lambda name: FILES[name])

    added = rg.update(today=date(2026, 3, 2), workers=1)
    out = pd.read_csv(snapshot)
    assert added == 2                                         # January and February 2026
    assert len(out) == 15 * 3                                 # December 2025 plus the 2 new months
    nga_jan = out[(out.iso3 == "NGA") & (out.year == 2026) & (out.month == 1)].iloc[0]
    assert nga_jan.events_total == 1 and nga_jan.events_protest == 1   # the 2025 event is left out
    ner_feb = out[(out.iso3 == "NER") & (out.year == 2026) & (out.month == 2)].iloc[0]
    assert ner_feb.events_violence == 1
    nga_dec = out[(out.iso3 == "NGA") & (out.year == 2025)].iloc[0]
    assert nga_dec.events_total == 5                          # old months are unchanged
    manifest = json.loads(snapshot.with_suffix(".json").read_text())
    assert manifest["updates"][0]["months"] == ["2026-01", "2026-02"]

    assert rg.update(today=date(2026, 3, 20), workers=1) == 0    # already up to date
