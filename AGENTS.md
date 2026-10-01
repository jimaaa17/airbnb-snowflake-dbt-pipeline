# AGENTS.md

Guidance for AI coding agents working in this repository.

## Overview

A dbt project on Snowflake that transforms Airbnb data (listings, bookings, hosts) through a Medallion architecture: `AIRBNB.staging` (raw tables loaded from S3 via `COPY INTO`, outside dbt) → `bronze` → `silver` → `gold`. The repo root is a Python/uv wrapper; the actual dbt project lives in `airbnb_snowflake_dbt_pipeline/`.

## Commands

Environment is managed with `uv` (Python 3.12, `package = false` — there is no Python package, only dependencies):

```bash
uv sync                        # install deps into .venv
source .venv/bin/activate      # or prefix commands with `uv run`
```

All dbt commands must run from `airbnb_snowflake_dbt_pipeline/` (where `dbt_project.yml` is). They require a working Snowflake connection via `profiles.yml` (profile `airbnb_snowflake_dbt_pipeline`, in `~/.dbt/` or the project dir — git-ignored).

```bash
dbt debug                                  # verify connection
dbt compile                                # render Jinja/check DAG without hitting tables
dbt run --select silver                    # run a layer (folder selector)
dbt run --select +obt                      # a model plus all upstream
dbt run --select silver_bookings --full-refresh   # rebuild an incremental model from scratch
dbt test --select silver                   # tests for a layer
dbt test --select silver_bookings          # all tests attached to one model
dbt test --select assert_total_amount_is_positive  # a single singular test
dbt build --select gold                    # run + test in DAG order
```

SQL formatting: `sqlfmt` is a dependency (`sqlfmt models/`).

## Architecture

- **Schema routing**: [macros/generate_schema_name.sql](airbnb_snowflake_dbt_pipeline/macros/generate_schema_name.sql) overrides dbt's default so `+schema: bronze` produces schema `bronze` exactly (not `<target_schema>_bronze`). Layer schemas are assigned per folder in `dbt_project.yml`; a new model's folder determines its schema.
- **Materialization**: `dbt_project.yml` sets every layer to `table`, but bronze and silver models override this in-file with `config(materialized='incremental')`. Gold (`obt`) is a full `table`.
- **Incremental pattern** (bronze and silver): filter on a `CREATED_AT` watermark — `WHERE CREATED_AT > (SELECT COALESCE(MAX(CREATED_AT), '1900-01-01') FROM {{ this }})` inside `{% if is_incremental() %}`. Silver models also set `unique_key` (the entity's `*_ID`) for merge. Follow this pattern for new bronze/silver models.
- **Layers**:
  - Bronze reads `{{ source('staging', ...) }}` (declared in `models/sources/sources.yml`) with `SELECT *`.
  - Silver reads `{{ ref('bronze_*') }}` and applies business logic via macros: `multiply(a, b, decimal_places=2)` (rounded product, used for `TOTAL_AMOUNT`), `tag(col)` (LOW/MEDIUM/HIGH price tier at <100/<200), `trimmer(col)`.
  - Gold `obt.sql` is a denormalized One Big Table built from a Jinja list of `{table, columns, alias, join_condition}` dicts looped into a `SELECT ... FROM bookings LEFT JOIN listings LEFT JOIN hosts`. To add an entity to the OBT, append a dict rather than hand-writing SQL.
- **Grain / keys**: bookings → listings via `LISTING_ID`, listings → hosts via `HOST_ID`. The OBT grain is one row per booking.
- **Column naming**: uppercase Snowflake identifiers (`BOOKING_ID`, `CREATED_AT`).

## Tests

- Generic tests live in each layer's `properties.yml` using the dbt ≥1.10 syntax: `data_tests:` (not `tests:`) with parameters nested under `arguments:` (e.g. `accepted_values: { arguments: { values: [...] } }`, `relationships: { arguments: { to: ref(...), field: ... } }`). Keep that form when adding tests.
- Singular tests in `tests/` return failing rows (0 rows = pass). They cover cross-layer invariants: bronze/silver row counts match, OBT row count equals `silver_bookings` (detects join fan-out), plus value-range checks.

## Notes

- `analyses/` holds scratch/Jinja experiments, not production models.
- `models/sources/base_staging_bookings.sql` is a scaffolded staging model outside the layer folders, so it falls back to the target's default schema.
- Planned next step (per README): a gold star schema (`fact_bookings`, `dim_listings`, `dim_hosts`) alongside the OBT.
