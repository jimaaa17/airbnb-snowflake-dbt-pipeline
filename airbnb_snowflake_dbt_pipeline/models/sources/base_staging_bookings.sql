with source as (
        select * from {{ source('staging', 'bookings') }}
  ),
  renamed as (
      select *
      from source
  )
  select * from renamed
    