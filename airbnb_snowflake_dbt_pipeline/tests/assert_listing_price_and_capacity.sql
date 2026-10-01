-- Test: Every listing must have a positive price and accommodate at least 1 guest.
-- Catches data quality issues where a listing has invalid capacity or pricing.
-- Returns failing rows; 0 rows = PASS.

SELECT 
    LISTING_ID,
    PRICE_PER_NIGHT,
    ACCOMMODATES
FROM {{ ref('silver_listings') }}
WHERE PRICE_PER_NIGHT <= 0 
   OR ACCOMMODATES < 1
