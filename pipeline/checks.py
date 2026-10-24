"""
Step 3: data quality checks.
Each check is a SQL query that counts PROBLEM rows. A count of 0 means the
check passed. If any check finds problems, the run stops with an error, so bad
data never reaches the models.
"""


class DataQualityError(Exception):
    """Raised when one or more data quality checks fail."""


def build_checks(cfg):
    """Return a list of (check name, SQL that counts problem rows)."""
    start, end = cfg["years"]["start"], cfg["years"]["end"]
    forecast = cfg["years"]["forecast"]
    n_countries = len(cfg["countries"])
    n_years = end - start + 1
    n_forecast_years = forecast - end
    return [
        # --- raw layer --------------------------------------------------------
        ("World Bank data covers every member",
         f"SELECT {n_countries} - COUNT(DISTINCT iso3) FROM raw.world_bank"),
        ("Real imports are positive",
         "SELECT COUNT(*) FROM raw.world_bank WHERE variable = 'imports_real' AND value <= 0"),
        ("Every event type is known",
         "SELECT COUNT(*) FROM raw.ecowas_events "
         "WHERE event_type NOT IN ('suspension', 'sanctions', 'border_closure')"),
        ("Every V-Dem country name matches a member",
         f"SELECT {n_countries} - COUNT(DISTINCT c.iso3) FROM raw.vdem v "
         "JOIN staging.countries c ON v.country_name = c.vdem_name"),
        ("Borders are two-way (if A borders B, B borders A)",
         "SELECT COUNT(*) FROM raw.borders a LEFT JOIN raw.borders b "
         "ON a.iso3 = b.neighbor AND a.neighbor = b.iso3 WHERE b.iso3 IS NULL"),

        # --- feature table ----------------------------------------------------
        ("Feature table has one row per member per year",
         f"SELECT ABS(COUNT(*) - {n_countries * n_years}) FROM features.country_year"),
        ("No duplicate country-years",
         "SELECT COUNT(*) - COUNT(DISTINCT (iso3, year)) FROM features.country_year"),
        (f"Every year from {start} to {end} has all {n_countries} members",
         f"SELECT COUNT(*) FROM (SELECT year, COUNT(*) AS n FROM features.country_year "
         f"GROUP BY year) WHERE n <> {n_countries} OR year < {start} OR year > {end}"),
        ("No missing coup counts or military rule scores",
         "SELECT COUNT(*) FROM features.country_year WHERE coups_successful IS NULL "
         "OR military_rule_index IS NULL OR coups_successful_lag1 IS NULL "
         "OR military_rule_index_lag1 IS NULL"),
        ("Military rule index is between 0 and 1",
         "SELECT COUNT(*) FROM features.country_year "
         "WHERE military_rule_index < 0 OR military_rule_index > 1"),
        ("Coup counts are not negative",
         "SELECT COUNT(*) FROM features.country_year "
         "WHERE coups_successful < 0 OR coups_failed < 0"),
        ("Action flag is 0 or 1 and agrees with severity",
         "SELECT COUNT(*) FROM features.country_year WHERE ecowas_action NOT IN (0, 1) "
         "OR action_severity NOT BETWEEN 0 AND 3 "
         "OR (ecowas_action = 1) <> (action_severity > 0)"),
        ("Every hand-coded action in the study period appears in the features",
         f"SELECT (SELECT COUNT(*) FROM staging.actions WHERE year BETWEEN {start} AND {end}) "
         "- (SELECT SUM(ecowas_action) FROM features.country_year)"),
        ("Border features are filled in, and border military rule is between 0 and 1",
         "SELECT COUNT(*) FROM features.country_year_all "
         "WHERE border_military_max_lag1 IS NULL OR border_coups_prev3 IS NULL "
         "OR border_military_max_lag1 < 0 OR border_military_max_lag1 > 1"),

        # --- forecast inputs --------------------------------------------------
        (f"Forecast inputs have all {n_countries} members for each year from {end + 1} to {forecast}",
         f"SELECT ABS(COUNT(*) - {n_countries * n_forecast_years}) "
         f"+ (SELECT COUNT(*) FROM features.forecast_inputs WHERE year <= {end} OR year > {forecast}) "
         "FROM features.forecast_inputs"),
        ("No missing forecast inputs",
         "SELECT COUNT(*) FROM features.forecast_inputs "
         "WHERE military_rule_index_lag1 IS NULL OR coups_successful_lag1 IS NULL "
         "OR neighbor_coups_prev3 IS NULL OR border_military_max_lag1 IS NULL "
         "OR border_coups_prev3 IS NULL OR coups_successful_prev3 IS NULL"),
    ]


def run_checks(con, cfg, verbose=True):
    """Run every check. Print the results and raise DataQualityError on any failure."""
    failures = []
    for name, sql in build_checks(cfg):
        problems = con.execute(sql).fetchone()[0] or 0
        passed = problems == 0
        if verbose:
            print(f"  [{'PASS' if passed else 'FAIL'}] {name}"
                  + ("" if passed else f" ({problems} problem rows)"))
        if not passed:
            failures.append(name)
    if failures:
        raise DataQualityError(f"{len(failures)} check(s) failed: {failures}")
