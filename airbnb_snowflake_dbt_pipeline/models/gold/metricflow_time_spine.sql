{{
    config(
        materialized='table'
    )
}}

with days as (
    select
        dateadd(day, seq4(), '2020-01-01'::date) as date_day
    from
        table(generator(rowcount => 3650))
)

select
    date_day
from
    days
where
    date_day <= '2030-12-31'::date
