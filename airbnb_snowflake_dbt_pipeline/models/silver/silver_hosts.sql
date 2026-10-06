{{ config(
    materialized = 'incremental',
    unique_key = 'HOST_ID'
) }}

SELECT
    HOST_ID,
    REPLACE(HOST_NAME, ' ', '_') AS HOST_NAME,
    IS_SUPERHOST,
    HOST_SINCE,
    RESPONSE_RATE,
    CASE 
        WHEN RESPONSE_RATE > 95 THEN 'VERY GOOD'
        WHEN RESPONSE_RATE > 80 THEN 'GOOD'
        WHEN RESPONSE_RATE > 60 THEN 'AVERAGE'
        ELSE 'POOR'
    END AS RESPONSE_RATE_BAND,
    CREATED_AT
FROM {{ ref('bronze_hosts') }}

{% if is_incremental() %}
    WHERE CREATED_AT > (SELECT COALESCE(MAX(CREATED_AT), '1900-01-01') FROM {{ this }})
{% endif %}