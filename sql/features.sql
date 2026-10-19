-- =============================================================================
-- features.sql
-- Step 2b: build the model-ready table, one row per member per year.
-- {start_year} and {end_year} come from config.yaml.
-- The table starts one year early ({start_year} - 1) so that the first year has
-- prior-year (lagged) values; that extra year is removed at the end.
-- =============================================================================

CREATE SCHEMA IF NOT EXISTS features;

CREATE OR REPLACE TABLE features.country_year AS
WITH

-- Every member paired with every year
skeleton AS (
    SELECT c.iso3, c.country, c.landlocked, y.range AS year
    FROM staging.countries AS c
    CROSS JOIN range({start_year} - 1, {end_year} + 1) AS y
),

-- Successful coups in all OTHER members over the previous three years.
-- Example: Niger in 2023 adds up coups in every other member in 2020 to 2022.
region_coups AS (
    SELECT
        s.iso3,
        s.year,
        COALESCE(SUM(v.coups_successful), 0.0) AS neighbor_coups_prev3
    FROM skeleton AS s
    LEFT JOIN staging.vdem AS v
      ON v.iso3 <> s.iso3
     AND v.year BETWEEN s.year - 3 AND s.year - 1
    GROUP BY s.iso3, s.year
),

-- Import damage measure: percent change in real imports from the year before
-- to the average of this year and the next year.
imports AS (
    SELECT iso3, year, imports_real
    FROM staging.world_bank
    WHERE imports_real IS NOT NULL
),
import_change AS (
    SELECT
        s.iso3,
        s.year,
        ((imp_now.imports_real + imp_next.imports_real) / 2 / imp_prev.imports_real - 1) * 100
            AS import_change_2yr          -- NULL if any of the three years is missing
    FROM skeleton AS s
    LEFT JOIN imports AS imp_prev ON imp_prev.iso3 = s.iso3 AND imp_prev.year = s.year - 1
    LEFT JOIN imports AS imp_now  ON imp_now.iso3  = s.iso3 AND imp_now.year  = s.year
    LEFT JOIN imports AS imp_next ON imp_next.iso3 = s.iso3 AND imp_next.year = s.year + 1
),

-- Put everything together. No row in the actions table means no action (0).
joined AS (
    SELECT
        s.iso3,
        s.country,
        s.year,
        s.landlocked,
        COALESCE(a.ecowas_action, 0)   AS ecowas_action,
        COALESCE(a.action_severity, 0) AS action_severity,
        COALESCE(a.suspension, 0)      AS suspension,
        COALESCE(a.sanctions, 0)       AS sanctions,
        COALESCE(a.border_closure, 0)  AS border_closure,
        v.coups_successful,
        v.coups_failed,
        r.neighbor_coups_prev3,
        v.mass_mobilization,
        v.military_rule_index,
        w.gdp_growth,
        w.inflation,
        w.imports_pct_gdp,
        i.import_change_2yr
    FROM skeleton AS s
    LEFT JOIN staging.actions    AS a ON a.iso3 = s.iso3 AND a.year = s.year
    LEFT JOIN staging.vdem       AS v ON v.iso3 = s.iso3 AND v.year = s.year
    LEFT JOIN region_coups       AS r ON r.iso3 = s.iso3 AND r.year = s.year
    LEFT JOIN staging.world_bank AS w ON w.iso3 = s.iso3 AND w.year = s.year
    LEFT JOIN import_change      AS i ON i.iso3 = s.iso3 AND i.year = s.year
),

-- Prior-year values: what the model is allowed to know before the year starts
lagged AS (
    SELECT
        *,
        LAG(coups_successful)    OVER w AS coups_successful_lag1,
        LAG(coups_failed)        OVER w AS coups_failed_lag1,
        LAG(mass_mobilization)   OVER w AS mass_mobilization_lag1,
        LAG(military_rule_index) OVER w AS military_rule_index_lag1,
        LAG(gdp_growth)          OVER w AS gdp_growth_lag1,
        LAG(inflation)           OVER w AS inflation_lag1,
        LAG(imports_pct_gdp)     OVER w AS imports_pct_gdp_lag1
    FROM joined
    WINDOW w AS (PARTITION BY iso3 ORDER BY year)
)

SELECT *
FROM lagged
WHERE year >= {start_year}
ORDER BY iso3, year;
