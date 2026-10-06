{{ config(materialized = 'ephemeral') }}

WITH hosts AS (
    SELECT DISTINCT
        HOST_ID,
        HOST_NAME,
        HOST_SINCE,
        IS_SUPERHOST,
        RESPONSE_RATE,
        RESPONSE_RATE_BAND,
        HOST_CREATED_AT
    FROM {{ ref('obt') }}
)

SELECT * FROM hosts