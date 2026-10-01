-- Test: Total booking amount must always be positive.
-- A non-positive total indicates a calculation bug in the multiply macro
-- or corrupt source data (negative prices/fees).
-- Returns failing rows; 0 rows = PASS.

SELECT 
    BOOKING_ID,
    TOTAL_AMOUNT
FROM {{ ref('silver_bookings') }}
WHERE TOTAL_AMOUNT <= 0
