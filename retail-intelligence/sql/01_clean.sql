-- Couche CLEAN : calendrier complet (le 25 dec. manque dans le dataset) + dedoublonnage
CREATE SCHEMA IF NOT EXISTS clean;

CREATE OR REPLACE TABLE clean.sales AS
WITH bounds AS (
    SELECT MIN(date) AS d0, MAX(date) AS d1 FROM raw.train
),
calendar AS (
    SELECT CAST(d AS DATE) AS date
    FROM bounds, generate_series(d0, d1, INTERVAL 1 DAY) AS t(d)
),
series AS (
    SELECT DISTINCT store_nbr, family FROM raw.train
),
dedup AS (
    SELECT date, store_nbr, family,
           MAX(sales) AS sales,
           MAX(onpromotion) AS onpromotion
    FROM raw.train
    GROUP BY date, store_nbr, family
)
SELECT c.date, s.store_nbr, s.family,
       COALESCE(d.sales, 0)       AS sales,
       COALESCE(d.onpromotion, 0) AS onpromotion
FROM calendar c
CROSS JOIN series s
LEFT JOIN dedup d
       ON d.date = c.date AND d.store_nbr = s.store_nbr AND d.family = s.family;

-- Jours feries nationaux effectifs (on exclut les fetes transferees)
CREATE OR REPLACE TABLE clean.national_holidays AS
SELECT DISTINCT date, 1 AS is_holiday
FROM raw.holidays_events
WHERE locale = 'National'
  AND CAST(transferred AS VARCHAR) ILIKE 'false'
  AND type IN ('Holiday', 'Transfer', 'Additional', 'Bridge');

