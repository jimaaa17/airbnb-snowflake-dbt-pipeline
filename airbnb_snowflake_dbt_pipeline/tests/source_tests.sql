-- tests/source_tests.sql
-- Guardrail: Validates raw data freshly loaded from S3 into AIRBNB.staging.
-- Catches corrupt records, null primary keys, negative prices/fees, and invalid ranges
-- at the very source before data enters the Bronze, Silver, or Gold layers.
-- Returns failing records; 0 rows = PASS.

{{ config(
    severity = 'warn',
    warn_if = '> 0'
) }}

WITH invalid_bookings AS (
    SELECT 
        'staging.bookings' AS source_table,
        BOOKING_ID AS record_identifier,
        'Invalid booking values: NULL ID, non-positive nights, or negative amount/fees' AS failure_reason
    FROM {{ source('staging', 'bookings') }}
    WHERE BOOKING_ID IS NULL 
       OR LISTING_ID IS NULL 
       OR BOOKING_DATE IS NULL
       OR NIGHTS_BOOKED <= 0 
       OR BOOKING_AMOUNT < 0 
       OR CLEANING_FEE < 0 
       OR SERVICE_FEE < 0
),

invalid_listings AS (
    SELECT 
        'staging.listings' AS source_table,
        CAST(LISTING_ID AS VARCHAR) AS record_identifier,
        'Invalid listing values: NULL ID, non-positive price, or accommodates < 1' AS failure_reason
    FROM {{ source('staging', 'listings') }}
    WHERE LISTING_ID IS NULL 
       OR HOST_ID IS NULL 
       OR PRICE_PER_NIGHT <= 0 
       OR ACCOMMODATES < 1
),

invalid_hosts AS (
    SELECT 
        'staging.hosts' AS source_table,
        CAST(HOST_ID AS VARCHAR) AS record_identifier,
        'Invalid host values: NULL ID, NULL name, or response rate not in 0-100 range' AS failure_reason
    FROM {{ source('staging', 'hosts') }}
    WHERE HOST_ID IS NULL 
       OR HOST_NAME IS NULL 
       OR RESPONSE_RATE < 0 
       OR RESPONSE_RATE > 100
)

SELECT * FROM invalid_bookings
UNION ALL
SELECT * FROM invalid_listings
UNION ALL
SELECT * FROM invalid_hosts
