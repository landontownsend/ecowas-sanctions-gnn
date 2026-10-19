-- =============================================================================
-- staging.sql
-- Step 2a: clean and rename the raw tables. One staging table per source.
-- No business logic here beyond renaming, type casting, and simple derivations.
-- =============================================================================

CREATE SCHEMA IF NOT EXISTS staging;

-- The 15 members
CREATE OR REPLACE TABLE staging.countries AS
SELECT
    iso3,
    name                     AS country,
    vdem_name,
    CAST(landlocked AS INTEGER) AS landlocked
FROM raw.countries;

-- The border network, one row per (country, neighbor) pair
CREATE OR REPLACE TABLE staging.borders AS
SELECT iso3, neighbor
FROM raw.borders;

-- ECOWAS actions, one row per country-year.
-- Severity uses the strongest action taken that year:
--   1 = suspension, 2 = sanctions, 3 = border closure or cutoff of trade and finance
CREATE OR REPLACE TABLE staging.actions AS
WITH events AS (
    SELECT
        iso3,
        year(CAST(event_date AS DATE)) AS year,
        event_type,
        CASE event_type
            WHEN 'suspension'     THEN 1
            WHEN 'sanctions'      THEN 2
            WHEN 'border_closure' THEN 3
        END AS severity
    FROM raw.ecowas_events
)
SELECT
    iso3,
    year,
    1                                                      AS ecowas_action,
    MAX(severity)                                          AS action_severity,
    MAX(CASE WHEN event_type = 'suspension'     THEN 1 ELSE 0 END) AS suspension,
    MAX(CASE WHEN event_type = 'sanctions'      THEN 1 ELSE 0 END) AS sanctions,
    MAX(CASE WHEN event_type = 'border_closure' THEN 1 ELSE 0 END) AS border_closure
FROM events
GROUP BY iso3, year;

-- V-Dem political measures, matched to ISO codes
--   e_pt_coup           successful coups (Powell and Thyne)
--   e_pt_coup_attempts  all attempts, successful and failed
--   v2cagenmob          mass mobilization (frequency and size of protests)
--   v2x_ex_military     how much executive power rests on the military, 0 to 1
CREATE OR REPLACE TABLE staging.vdem AS
SELECT
    c.iso3,
    v.year,
    CAST(v.e_pt_coup AS DOUBLE)                              AS coups_successful,
    CAST(v.e_pt_coup_attempts - v.e_pt_coup AS DOUBLE)       AS coups_failed,
    CAST(v.v2cagenmob AS DOUBLE)                             AS mass_mobilization,
    CAST(v.v2x_ex_military AS DOUBLE)                        AS military_rule_index
FROM raw.vdem AS v
JOIN staging.countries AS c
  ON v.country_name = c.vdem_name;

-- World Bank indicators, one column per indicator
CREATE OR REPLACE TABLE staging.world_bank AS
SELECT
    iso3,
    CAST(year AS INTEGER) AS year,
    MAX(value) FILTER (WHERE variable = 'gdp_growth')      AS gdp_growth,
    MAX(value) FILTER (WHERE variable = 'inflation')       AS inflation,
    MAX(value) FILTER (WHERE variable = 'imports_pct_gdp') AS imports_pct_gdp,
    MAX(value) FILTER (WHERE variable = 'imports_real')    AS imports_real
FROM raw.world_bank
GROUP BY iso3, year;
