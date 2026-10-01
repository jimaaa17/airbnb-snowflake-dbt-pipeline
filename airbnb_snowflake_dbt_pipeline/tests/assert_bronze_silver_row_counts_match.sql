-- Test: Bronze-to-Silver row counts must match for each entity.
-- During incremental loads, if rows are dropped between bronze and silver,
-- this test catches completeness issues.
-- Returns a row for any entity where counts differ; 0 rows = PASS.

WITH bronze_counts AS (
    SELECT 'bookings' AS entity, COUNT(*) AS bronze_cnt FROM {{ ref('bronze_bookings') }}
    UNION ALL
    SELECT 'listings', COUNT(*) FROM {{ ref('bronze_listings') }}
    UNION ALL
    SELECT 'hosts', COUNT(*) FROM {{ ref('bronze_hosts') }}
),
silver_counts AS (
    SELECT 'bookings' AS entity, COUNT(*) AS silver_cnt FROM {{ ref('silver_bookings') }}
    UNION ALL
    SELECT 'listings', COUNT(*) FROM {{ ref('silver_listings') }}
    UNION ALL
    SELECT 'hosts', COUNT(*) FROM {{ ref('silver_hosts') }}
)

SELECT 
    b.entity,
    b.bronze_cnt,
    s.silver_cnt
FROM bronze_counts b
JOIN silver_counts s ON b.entity = s.entity
WHERE b.bronze_cnt != s.silver_cnt
