-- Test: OBT row count must exactly match silver_bookings row count.
-- OBT is built from silver_bookings LEFT JOIN silver_listings LEFT JOIN silver_hosts.
-- If OBT has MORE rows, a join fanned out (duplicate keys).
-- If OBT has FEWER rows, rows were dropped.
-- Returns a row if counts differ; 0 rows = PASS.

WITH booking_count AS (
    SELECT COUNT(*) AS total_bookings FROM {{ ref('silver_bookings') }}
),
obt_count AS (
    SELECT COUNT(*) AS total_obt FROM {{ ref('obt') }}
)

SELECT 
    b.total_bookings,
    o.total_obt
FROM booking_count b
CROSS JOIN obt_count o
WHERE b.total_bookings != o.total_obt
