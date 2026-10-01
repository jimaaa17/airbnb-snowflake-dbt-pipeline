{% macro multiply(a, b, decimal_places=2) %}
round({{ a }}* {{ b }}, {{decimal_places}})
{% endmacro %}