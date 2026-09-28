{% set cols = ['BOOKING_ID', 'NIGHTS_BOOKED', 'BOKING_AMOUNT']%}

SELECT 
    {% for col in cols%}
        {{ col }}
        {% if not loop.last %}, {% endif %} 
    {% endfor %}
FROM bronze_bookings