{% set configs = [
    {
        "table": ref('silver_bookings'),
        "columns": "SILVER_bookings.BOOKING_ID, SILVER_bookings.BOOKING_DATE, SILVER_bookings.TOTAL_AMOUNT, SILVER_bookings.BOOKING_STATUS, SILVER_bookings.CREATED_AT AS BOOKING_CREATED_AT",
        "alias": "SILVER_bookings"
    },
    {
        "table": ref('silver_listings'),
        "columns": "SILVER_listings.LISTING_ID, SILVER_listings.PROPERTY_TYPE, SILVER_listings.ROOM_TYPE, SILVER_listings.CITY, SILVER_listings.COUNTRY, SILVER_listings.ACCOMMODATES, SILVER_listings.BEDROOMS, SILVER_listings.BATHROOMS, SILVER_listings.PRICE_PER_NIGHT, SILVER_listings.PRICE_PER_NIGHT_TAG, SILVER_listings.CREATED_AT AS LISTING_CREATED_AT",
        "alias": "SILVER_listings",
        "join_condition": "SILVER_bookings.LISTING_ID = SILVER_listings.LISTING_ID"
    },
    {
        "table": ref('silver_hosts'),
        "columns": "SILVER_hosts.HOST_ID, SILVER_hosts.HOST_NAME, SILVER_hosts.HOST_SINCE, SILVER_hosts.RESPONSE_RATE, SILVER_hosts.RESPONSE_RATE_BAND, SILVER_hosts.CREATED_AT AS HOST_CREATED_AT",
        "alias": "SILVER_hosts",
        "join_condition": "SILVER_listings.HOST_ID = SILVER_hosts.HOST_ID"
    }
] %}

SELECT 
    {% for config in configs%}
    {{ config['columns']}}{{',' if not loop.last}}
    {%endfor%}
FROM 
    {% for config in configs %}
    {%if loop.first %}
        {{ config['table'] }} AS {{ config['alias'] }}
    {% else %}
        LEFT JOIN {{ config['table'] }} AS {{ config['alias'] }} 
        ON {{ config['join_condition'] }}
    {% endif %}
    {%endfor%}
    
    