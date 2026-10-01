-- Test: Host response rate must be between 0 and 100 (inclusive).
-- A value outside this range indicates corrupt source data
-- or a transformation error.
-- Returns failing rows; 0 rows = PASS.

SELECT 
    HOST_ID,
    RESPONSE_RATE
FROM {{ ref('silver_hosts') }}
WHERE RESPONSE_RATE < 0 OR RESPONSE_RATE > 100
