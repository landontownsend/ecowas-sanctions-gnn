"""The SQL feature table must match the dataset the published results were built on."""

import pandas as pd
import pytest

from pipeline.transform import features_table, forecast_table


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


def test_border_features_for_niger_2023(con):
    # Niger borders Mali, Burkina Faso, Benin and Nigeria.
    # Highest military rule among them in 2022: Mali, 1.0.
    # Their coups in 2020 to 2022: Mali 2020 (1) + Mali 2021 (1) + Burkina Faso 2022 (2) = 4.
    niger = row(features_table(con), "NER", 2023)
    assert niger.border_military_max_lag1 == 1.0
    assert niger.border_coups_prev3 == 4


def test_cabo_verde_has_no_border_values(con):
    cabo_verde = features_table(con).query("iso3 == 'CPV'")
    assert (cabo_verde.border_military_max_lag1 == 0).all()
    assert (cabo_verde.border_coups_prev3 == 0).all()


def test_forecast_inputs_match_the_published_file(con, cfg):
    built = forecast_table(con).round(3)
    published = pd.read_csv(cfg["paths"]["forecast_csv"])
    pd.testing.assert_frame_equal(built.reset_index(drop=True), published, check_dtype=False)


def test_forecast_inputs_cover_the_years_after_the_study_period(con, cfg):
    years = sorted(forecast_table(con).year.unique())
    assert years == list(range(cfg["years"]["end"] + 1, cfg["years"]["forecast"] + 1))


def test_guinea_bissau_2025_coup_reaches_the_2026_inputs(con):
    # V-Dem codes Guinea-Bissau's November 2025 coup; other 2025 counts are blank (0)
    fc = forecast_table(con)
    assert row(fc, "GNB", 2026).coups_successful_lag1 == 1
    assert row(fc, "SEN", 2026).coups_successful_lag1 == 0
