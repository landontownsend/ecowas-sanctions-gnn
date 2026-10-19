"""The data quality checks pass on good data and stop the run on bad data."""

import pytest

from pipeline.checks import DataQualityError, run_checks


def test_checks_pass_on_the_real_data(con, cfg):
    run_checks(con, cfg, verbose=False)          # raises if anything fails


def test_out_of_range_military_rule_is_caught(con, cfg):
    con.execute("UPDATE features.country_year SET military_rule_index = 1.5 "
                "WHERE iso3 = 'GHA' AND year = 2015")
    with pytest.raises(DataQualityError, match="between 0 and 1"):
        run_checks(con, cfg, verbose=False)


def test_missing_country_year_is_caught(con, cfg):
    con.execute("DELETE FROM features.country_year WHERE iso3 = 'SEN' AND year = 2018")
    with pytest.raises(DataQualityError, match="one row per member per year"):
        run_checks(con, cfg, verbose=False)


def test_one_way_border_is_caught(con, cfg):
    con.execute("DELETE FROM raw.borders WHERE iso3 = 'GMB' AND neighbor = 'SEN'")
    with pytest.raises(DataQualityError, match="two-way"):
        run_checks(con, cfg, verbose=False)
