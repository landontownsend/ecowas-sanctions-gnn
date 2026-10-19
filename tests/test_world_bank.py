"""World Bank ingestion, tested with fake API responses (no internet needed)."""

import pytest

from pipeline.ingest import fetch_world_bank, parse_world_bank_page, read_world_bank_snapshot

# A small example in the same shape as a real World Bank API response
PAGE = [
    {"page": 1, "pages": 1, "per_page": 1000, "total": 2},
    [
        {"countryiso3code": "NER", "date": "2023", "value": 3372988000.0},
        {"countryiso3code": "NER", "date": "2022", "value": None},
    ],
]


def test_parse_page_reads_country_year_and_value():
    rows = parse_world_bank_page(PAGE, "imports_real")
    assert rows[0] == {"iso3": "NER", "year": 2023, "variable": "imports_real",
                       "value": 3372988000.0}
    assert rows[1]["value"] is None


def test_parse_page_rejects_an_error_message():
    error = [{"message": [{"id": "120", "key": "Invalid value"}]}]
    with pytest.raises(ValueError):
        parse_world_bank_page(error, "imports_real")


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class FakeSession:
    """Answers every request with two pages of one record each, to test paging."""

    def __init__(self):
        self.calls = []

    def get(self, url, params, timeout):
        self.calls.append((url, params["page"]))
        year = str(2020 + params["page"])
        return FakeResponse([{"page": params["page"], "pages": 2},
                             [{"countryiso3code": "BEN", "date": year, "value": 1.5}]])


def test_fetch_follows_every_page_for_every_indicator(cfg):
    session = FakeSession()
    long = fetch_world_bank(cfg, session=session)
    n_indicators = len(cfg["sources"]["world_bank"]["indicators"])
    assert len(session.calls) == 2 * n_indicators          # two pages per indicator
    assert len(long) == 2 * n_indicators
    assert set(long.columns) == {"iso3", "year", "variable", "value"}


def test_snapshot_is_converted_to_long_format(cfg):
    long = read_world_bank_snapshot(cfg)
    assert set(long.columns) == {"iso3", "year", "variable", "value"}
    assert set(long.variable) == set(cfg["sources"]["world_bank"]["indicators"].values())
    assert long.value.notna().all()
