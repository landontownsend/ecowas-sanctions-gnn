"""The SQL feature table must match the dataset the published results were built on."""

import pandas as pd
import pytest

from pipeline.transform import features_table


def test_features_match_the_published_dataset(con, cfg):
    built = features_table(con).round(3)
    published = pd.read_csv(cfg["paths"]["features_csv"])
    pd.testing.assert_frame_equal(built.reset_index(drop=True), published, check_dtype=False)


def row(df, iso3, year):
    return df[(df.iso3 == iso3) & (df.year == year)].iloc[0]


def test_import_change_for_niger_2023(con):
    # Niger 2022 imports vs. the average of 2023 and 2024
    wb = con.execute("SELECT year, value FROM raw.world_bank WHERE iso3 = 'NER' "
                     "AND variable = 'imports_real' AND year BETWEEN 2022 AND 2024 "
                     "ORDER BY year").df()
    before, now, after = wb.value
    expected = ((now + after) / 2 / before - 1) * 100
    assert row(features_table(con), "NER", 2023).import_change_2yr == pytest.approx(expected)


def test_region_coups_count_other_members_over_three_prior_years(con):
    vdem = con.execute("SELECT iso3, year, coups_successful FROM staging.vdem").df()
    window = vdem[(vdem.iso3 != "NER") & vdem.year.between(2020, 2022)]
    expected = window.coups_successful.sum()
    assert row(features_table(con), "NER", 2023).neighbor_coups_prev3 == expected


def test_lagged_values_are_last_years_values(con):
    df = features_table(con)
    assert row(df, "MLI", 2021).military_rule_index_lag1 == row(df, "MLI", 2020).military_rule_index


def test_ten_actions_in_the_study_period(con):
    assert features_table(con).ecowas_action.sum() == 10
