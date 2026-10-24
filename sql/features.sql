-- =============================================================================
-- features.sql
-- Step 2b: build the model inputs, one row per member per year.
-- {start_year}, {end_year} and {forecast_year} come from config.yaml.
--
-- Three tables:
--   features.country_year_all   every year from {start_year} to {forecast_year}
--   features.country_year       the study period ({start_year} to {end_year}),
--                               with the outcome; the models train on this table
--   features.forecast_inputs    the years after the study period, model inputs
--                               only; the out-of-sample check and the watch list
--                               score these rows
--
-- The query starts one year early ({start_year} - 1) so that the first year has
-- prior-year (lagged) values; that extra year is removed at the end.
-- =============================================================================

CREATE SCHEMA IF NOT EXISTS features;

CREATE OR REPLACE TABLE features.country_year_all AS
WITH

-- Every member paired with every year
skeleton AS (
    SELECT c.iso3, c.country, c.landlocked, y.range AS year
    FROM staging.countries AS c
    CROSS JOIN range({start_year} - 1, {forecast_year} + 1) AS y
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

-- Highest military rule score among BORDER neighbors last year.
-- Example: Niger in 2023 looks at Mali, Burkina Faso, Benin and Nigeria in 2022.
-- Mali scored 1.0, so the value is 1.0.
-- A member with no land borders (Cabo Verde) gets 0.
border_military AS (
    SELECT
        s.iso3,
        s.year,
        COALESCE(MAX(v.military_rule_index), 0.0) AS border_military_max_lag1
    FROM skeleton AS s
    LEFT JOIN staging.borders AS b
      ON b.iso3 = s.iso3
    LEFT JOIN staging.vdem AS v
      ON v.iso3 = b.neighbor
     AND v.year = s.year - 1
    GROUP BY s.iso3, s.year
),

-- Successful coups in BORDER neighbors over the previous three years.
-- Example: Niger in 2023 adds up coups in its four neighbors in 2020 to 2022:
-- Mali 2020 (1) + Mali 2021 (1) + Burkina Faso 2022 (2) = 4.
border_coups AS (
    SELECT
        s.iso3,
        s.year,
        COALESCE(SUM(v.coups_successful), 0.0) AS border_coups_prev3
    FROM skeleton AS s
    LEFT JOIN staging.borders AS b
      ON b.iso3 = s.iso3
    LEFT JOIN staging.vdem AS v
      ON v.iso3 = b.neighbor
     AND v.year BETWEEN s.year - 3 AND s.year - 1
    GROUP BY s.iso3, s.year
),

-- The member's OWN successful coups over the previous three years
-- (a node feature for the graph neural network).
own_coups AS (
    SELECT
        s.iso3,
        s.year,
        COALESCE(SUM(v.coups_successful), 0.0) AS coups_successful_prev3
    FROM skeleton AS s
    LEFT JOIN staging.vdem AS v
      ON v.iso3 = s.iso3
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

-- The network features go at the end, after the lagged values
SELECT
    l.*,
    bm.border_military_max_lag1,
    bc.border_coups_prev3,
    oc.coups_successful_prev3
FROM lagged AS l
LEFT JOIN border_military AS bm ON bm.iso3 = l.iso3 AND bm.year = l.year
LEFT JOIN border_coups    AS bc ON bc.iso3 = l.iso3 AND bc.year = l.year
LEFT JOIN own_coups       AS oc ON oc.iso3 = l.iso3 AND oc.year = l.year
WHERE l.year >= {start_year}
ORDER BY l.iso3, l.year;


-- The study period: what the models train and are tested on
CREATE OR REPLACE TABLE features.country_year AS
SELECT *
FROM features.country_year_all
WHERE year <= {end_year}
ORDER BY iso3, year;


-- The years after the study period: model inputs only, no outcome.
-- V-Dem leaves a coup count blank when it has not yet coded the year (for
-- example 2025 in the v16 release). Here a blank count is treated as 0 coups.
CREATE OR REPLACE TABLE features.forecast_inputs AS
SELECT
    iso3,
    year,
    military_rule_index_lag1,
    COALESCE(coups_successful_lag1, 0.0) AS coups_successful_lag1,
    neighbor_coups_prev3,
    border_military_max_lag1,
    border_coups_prev3,
    coups_successful_prev3
FROM features.country_year_all
WHERE year > {end_year}
ORDER BY iso3, year;
