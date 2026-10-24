-- =============================================================================
-- monthly.sql
-- Step 2c: the country-month panel, one row per member per month,
-- {start_year} to {end_year}. Runs after features.sql.
--
-- For a forecast made at the end of a month:
--   GDELT shares       the last 1, 3, and 6 months, up to and including this month
--   yearly inputs      the previous year's values (V-Dem, World Bank, ECOWAS
--                      history), the same ones the yearly models use
--   action_next_6m     1 if ECOWAS takes a new action in any of the next 6 months
--
-- Example: Niger's suspension and border closure came on July 30, 2023, so
-- action_this_month = 1 for July 2023, and action_next_6m = 1 for January to
-- June 2023 (July is within the next 6 months of each).
-- =============================================================================

CREATE OR REPLACE TABLE features.country_month AS
WITH

-- Every member and month. It starts a year early (for the 6-month GDELT
-- windows) and ends a year late (for the 6 months after December {end_year}).
skeleton AS (
    SELECT
        c.iso3,
        y.range AS year,
        m.range AS month,
        y.range * 12 + m.range AS month_number      -- consecutive, for the windows
    FROM staging.countries AS c
    CROSS JOIN range({start_year} - 1, {end_year} + 2) AS y
    CROSS JOIN range(1, 13) AS m
),

-- ECOWAS actions by month, from the exact event dates
actions AS (
    SELECT
        iso3,
        year(CAST(event_date AS DATE))  AS year,
        month(CAST(event_date AS DATE)) AS month,
        MAX(CASE event_type
                WHEN 'suspension'     THEN 1
                WHEN 'sanctions'      THEN 2
                WHEN 'border_closure' THEN 3
            END) AS action_severity
    FROM raw.ecowas_events
    GROUP BY 1, 2, 3
),

joined AS (
    SELECT
        s.*,
        COALESCE(g.events_total, 0)    AS gdelt_events,
        COALESCE(g.events_protest, 0)  AS gdelt_protest_events,
        COALESCE(g.events_violence, 0) AS gdelt_violence_events,
        CASE WHEN a.iso3 IS NULL THEN 0 ELSE 1 END AS action_this_month,
        COALESCE(a.action_severity, 0) AS action_severity_this_month
    FROM skeleton AS s
    LEFT JOIN raw.gdelt AS g ON g.iso3 = s.iso3 AND g.year = s.year AND g.month = s.month
    LEFT JOIN actions   AS a ON a.iso3 = s.iso3 AND a.year = s.year AND a.month = s.month
),

-- Event totals over the last 3 and 6 months, and the outcome window
windows AS (
    SELECT
        *,
        SUM(gdelt_events)          OVER last3 AS events_3m,
        SUM(gdelt_protest_events)  OVER last3 AS protest_3m,
        SUM(gdelt_violence_events) OVER last3 AS violence_3m,
        SUM(gdelt_events)          OVER last6 AS events_6m,
        SUM(gdelt_protest_events)  OVER last6 AS protest_6m,
        SUM(gdelt_violence_events) OVER last6 AS violence_6m,
        MAX(action_this_month)     OVER next6 AS action_next_6m
    FROM joined
    WINDOW
        last3 AS (PARTITION BY iso3 ORDER BY month_number ROWS BETWEEN 2 PRECEDING AND CURRENT ROW),
        last6 AS (PARTITION BY iso3 ORDER BY month_number ROWS BETWEEN 5 PRECEDING AND CURRENT ROW),
        next6 AS (PARTITION BY iso3 ORDER BY month_number ROWS BETWEEN 1 FOLLOWING AND 6 FOLLOWING)
)

SELECT
    w.iso3,
    w.year,
    w.month,
    w.gdelt_events,
    -- Shares are missing when the window has fewer than {gdelt_min_events} events,
    -- as in the yearly table
    CASE WHEN w.gdelt_events >= {gdelt_min_events}
         THEN w.gdelt_protest_events / w.gdelt_events END  AS gdelt_protest_share_1m,
    CASE WHEN w.events_3m >= {gdelt_min_events}
         THEN w.protest_3m / w.events_3m END               AS gdelt_protest_share_3m,
    CASE WHEN w.events_6m >= {gdelt_min_events}
         THEN w.protest_6m / w.events_6m END               AS gdelt_protest_share_6m,
    CASE WHEN w.gdelt_events >= {gdelt_min_events}
         THEN w.gdelt_violence_events / w.gdelt_events END AS gdelt_violence_share_1m,
    CASE WHEN w.events_3m >= {gdelt_min_events}
         THEN w.violence_3m / w.events_3m END              AS gdelt_violence_share_3m,
    CASE WHEN w.events_6m >= {gdelt_min_events}
         THEN w.violence_6m / w.events_6m END              AS gdelt_violence_share_6m,
    -- The previous year's values, from the yearly table
    f.military_rule_index_lag1,
    f.coups_successful_lag1,
    f.coups_failed_lag1,
    f.mass_mobilization_lag1,
    f.gdp_growth_lag1,
    f.inflation_lag1,
    f.ecowas_action_lag1,
    f.years_since_coup,
    f.border_military_max_lag1,
    f.border_coups_prev3,
    -- Outcomes
    w.action_this_month,
    w.action_severity_this_month,
    w.action_next_6m
FROM windows AS w
JOIN features.country_year AS f
  ON f.iso3 = w.iso3 AND f.year = w.year
WHERE w.year BETWEEN {start_year} AND {end_year}
ORDER BY w.iso3, w.year, w.month;
