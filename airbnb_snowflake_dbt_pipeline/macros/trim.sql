{% macro trimmer(column_name) %}
    trim({{ column_name }})
{% endmacro %}
