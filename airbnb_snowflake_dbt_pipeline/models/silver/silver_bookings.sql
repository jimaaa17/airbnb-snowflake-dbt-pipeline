{{ config(
    materialized = 'incremental', 
    unique_key = 'BOOKING_ID'
) }}

SELECT 
    BOOKING_ID,
    LISTING_ID,
    BOOKING_DATE,
    {{ multiply('nights_booked', 'BOOKING_AMOUNT') }} + CLEANING_FEE + SERVICE_FEE AS TOTAL_AMOUNT,
    BOOKING_STATUS,
    CREATED_AT
FROM {{ ref('bronze_bookings') }}

{% if is_incremental() %}
    WHERE CREATED_AT > (SELECT COALESCE(MAX(CREATED_AT), '1900-01-01') FROM {{ this }})
{% endif %}