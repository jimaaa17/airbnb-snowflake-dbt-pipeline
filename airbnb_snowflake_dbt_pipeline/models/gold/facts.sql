{% set configs = [
    {
        "table": ref('obt'),
        "columns": "obt.BOOKING_ID, obt.LISTING_ID, obt.HOST_ID, obt.CLEANING_FEE, obt.SERVICE_FEE, obt.TOTAL_AMOUNT, obt.ACCOMMODATES, obt.BEDROOMS, obt.BATHROOMS, obt.PRICE_PER_NIGHT, obt.RESPONSE_RATE",
        "alias": "obt"
    },
    {
        "table": ref('dim_listings'),
        "columns": "",
        "alias": "dim_listings",
        "join_condition": "obt.LISTING_ID = dim_listings.LISTING_ID"
    },
    {
        "table": ref('dim_hosts'),
        "columns": "",
        "alias": "dim_hosts",
        "join_condition": "obt.HOST_ID = dim_hosts.HOST_ID"
    }
] %}

SELECT 
    {{ configs[0]['columns'] }}
FROM 
    {% for config in configs %}
    {% if loop.first %}
        {{ config['table'] }} AS {{ config['alias'] }}
    {% else %}
        LEFT JOIN {{ config['table'] }} AS {{ config['alias'] }} 
        ON {{ config['join_condition'] }}
    {% endif %}
    {% endfor %}
    
    