"""The country-month panel: outcome windows, GDELT shares, and yearly inputs."""

import pandas as pd
import pytest

from pipeline.transform import features_table, month_table


def month(df, iso3, year, m):
    return df[(df.iso3 == iso3) & (df.year == year) & (df.month == m)].iloc[0]


def test_one_row_per_member_per_month(con, cfg):
    df = month_table(con)
    n_years = cfg["years"]["end"] - cfg["years"]["start"] + 1
    assert len(df) == 15 * n_years * 12


def test_niger_2023_outcome_window(con):
    # Suspension and border closure on July 30, 2023, and no other Niger action nearby
    df = month_table(con)
    assert month(df, "NER", 2023, 7).action_this_month == 1
    assert month(df, "NER", 2023, 7).action_severity_this_month == 3
    for m in range(1, 7):                                    # January to June: July is ahead
        assert month(df, "NER", 2023, m).action_next_6m == 1
    assert month(df, "NER", 2022, 12).action_next_6m == 0    # July is 7 months ahead
    assert month(df, "NER", 2023, 7).action_next_6m == 0     # the action month itself is not counted


def test_guinea_2009_two_actions(con):
    # Suspension January 10, 2009; arms embargo October 17, 2009
    df = month_table(con)
    assert month(df, "GIN", 2008, 7).action_next_6m == 1     # January 2009 is 6 months ahead
    assert month(df, "GIN", 2009, 3).action_next_6m == 0     # April to September: nothing
    assert month(df, "GIN", 2009, 4).action_next_6m == 1     # October is 6 months ahead


def test_three_month_protest_share(con):
    # Nigeria, March 2012: protest events / all events, January to March 2012
    g = con.execute("SELECT SUM(events_protest) / SUM(events_total) FROM raw.gdelt "
                    "WHERE iso3 = 'NGA' AND year = 2012 AND month BETWEEN 1 AND 3").fetchone()[0]
    assert month(month_table(con), "NGA", 2012, 3).gdelt_protest_share_3m == pytest.approx(g)


def test_three_month_window_reaches_into_last_year(con):
    # Ghana, January 2010: November 2009 to January 2010
    g = con.execute("SELECT SUM(events_violence) / SUM(events_total) FROM raw.gdelt WHERE iso3 = 'GHA' "
                    "AND (year * 12 + month) BETWEEN 2009 * 12 + 11 AND 2010 * 12 + 1").fetchone()[0]
    assert month(month_table(con), "GHA", 2010, 1).gdelt_violence_share_3m == pytest.approx(g)


def test_yearly_inputs_are_the_previous_years(con):
    yearly = features_table(con)
    mali = yearly[(yearly.iso3 == "MLI") & (yearly.year == 2021)].iloc[0]
    for m in [1, 6, 12]:
        row = month(month_table(con), "MLI", 2021, m)
        assert row.military_rule_index_lag1 == mali.military_rule_index_lag1
        assert row.years_since_coup == mali.years_since_coup


def test_sixteen_action_months(con):
    assert month_table(con).action_this_month.sum() == 16
