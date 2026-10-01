# 🏡 Airbnb Snowflake dbt Pipeline

An end-to-end data transformation and analytics pipeline for Airbnb data on Snowflake, modeled using **dbt** with a **Medallion (Bronze → Silver → Gold) Architecture**.

---

## 📌 Project Overview

This project implements an end-to-end ELT data pipeline for Airbnb datasets using **AWS S3**, **Snowflake**, and **dbt** with a **Medallion Architecture**:

```text
Source Data (CSV) ──> AWS S3 ──> Snowflake (Staging) ──> Bronze Layer ──> Silver Layer ──> Gold Layer
                                          │                     │               │              │
                                     Raw Tables            Raw Ingestion   Cleaned Data    Analytics
```

- **Source Data (CSV)**: Raw Airbnb CSV files (`listings.csv`, `bookings.csv`, `hosts.csv`).
- **AWS S3**: Cloud object storage acting as the external landing and staging repository.
- **Snowflake (Staging Layer)**: Raw relational tables in `AIRBNB.staging` loaded via Snowflake external stages / `COPY INTO`.
- **Bronze Layer (`AIRBNB.bronze`)**: Incremental 1-to-1 raw ingestion models managed by dbt.
- **Silver Layer (`AIRBNB.silver`)**: Cleaned, standardized, and enriched models applying business transformation macros.
- **Gold Layer (`AIRBNB.gold`)**: High-performance analytical fact and dimension models optimized for BI and metrics.

---

## 🏗️ Architecture & Data Flow

```mermaid
flowchart LR
    CSV["📄 Source Data<br/>(CSV Files)"]
    S3[("☁️ AWS S3<br/>(Landing Zone)")]
    
    subgraph Snowflake ["Snowflake Cloud Data Platform (AIRBNB DB)"]
        subgraph Staging ["Staging Layer (staging)"]
            RAW[("Raw Tables<br/>• listings<br/>• bookings<br/>• hosts")]
        end

        subgraph Bronze ["Bronze Layer (bronze)"]
            BRZ["Raw Ingestion<br/>• bronze_listings<br/>• bronze_bookings<br/>• bronze_hosts"]
        end

        subgraph Silver ["Silver Layer (silver)"]
            SLV["Cleaned Data<br/>• Type Casting<br/>• Macro Transformations<br/>• Null Handling"]
        end

        subgraph Gold ["Gold Layer (gold)"]
            GLD["Analytics Layer<br/>• Fact Bookings<br/>• Dim Listings & Hosts<br/>• Business Metrics"]
        end
    end

    CSV --> S3
    S3 -->|"COPY INTO / Stage"| RAW
    RAW -->|"dbt source()"| BRZ
    BRZ -->|"dbt ref() / Incremental"| SLV
    SLV -->|"dbt ref() / Dimensional"| GLD
```

### ✅ Completed Work:
1. **Modern Python & Tooling Environment**:
   - Packaged and managed using **Astral `uv`** on Python 3.12.
   - Core libraries: `dbt-core (v1.12.5)`, `dbt-snowflake (v1.12.1)`, `boto3`.
2. **Snowflake Integration**:
   - Configured dbt Snowflake adapter connected to database `AIRBNB` and warehouse `SNOWFLAKE_LEARNING_WH`.
3. **Custom Schema Control**:
   - Custom `generate_schema_name` macro implemented to route tables directly into their dedicated schemas (`bronze`, `silver`, `gold`) without default schema prefixes.
4. **Source Declarations**:
   - Structured `sources.yml` mapping `AIRBNB.staging` tables (`listings`, `bookings`, `hosts`).
5. **Bronze Layer Ingestion**:
   - Models created for `bronze_listings`, `bronze_bookings`, and `bronze_hosts`.
   - Materialization configured as `incremental` utilizing dbt's `is_incremental()` macro with timestamp watermarking on `CREATED_AT`.
6. **Silver Layer Cleansing & Enrichment**:
   - Models created for `silver_bookings`, `silver_listings`, and `silver_hosts` (incremental).
   - DRY Jinja macros developed: `multiply` (dynamic rounding), `tag` (tier bucketing), `trimmer` (string trimming).
   - Schema tests configured (`unique`, `not_null` on all primary keys).
7. **Gold Layer (One Big Table - OBT)**:
   - Denormalized wide table `obt` joining bookings, listings, and hosts using dynamic Jinja loops.
   - Fully tested with primary key uniqueness and non-null referential integrity tests.

---

## 📂 Project Structure

```text
Airbnb Snowflake DBT Pipeline/
├── pyproject.toml                         # Project metadata and dependencies (managed via uv)
├── uv.lock                                # Deterministic lockfile
├── .gitignore                             # Ignored credentials, venvs, and build artifacts
├── README.md                              # Project documentation
│
└── airbnb_snowflake_dbt_pipeline/        # Core dbt project
    ├── dbt_project.yml                    # dbt project configuration & schema mapping
    ├── profiles.yml                       # Connection profile (git-ignored for security)
    │
    ├── macros/
    │   ├── generate_schema_name.sql       # Custom macro for clean bronze/silver/gold schemas
    │   ├── multiply.sql                   # Dynamic precision multiplication macro
    │   ├── tag.sql                        # Conditional price tier bucketing macro
    │   └── trim.sql                       # Reusable whitespace trimming macro
    │
    ├── models/
    │   ├── sources/
    │   │   └── sources.yml                # Raw staging source definitions
    │   ├── bronze/                        # Bronze layer models (incremental)
    │   │   ├── bronze_listings.sql
    │   │   ├── bronze_bookings.sql
    │   │   ├── bronze_hosts.sql
    │   │   └── properties.yml
    │   ├── silver/                        # Silver layer models (incremental)
    │   │   ├── silver_bookings.sql
    │   │   ├── silver_listings.sql
    │   │   ├── silver_hosts.sql
    │   │   └── properties.yml
    │   └── gold/                          # Gold layer models (analytical marts)
    │       ├── obt.sql                    # One Big Table (OBT) denormalized mart
    │       └── properties.yml
    │
    ├── analyses/                          # Ad-hoc exploratory queries & Jinja experiments
    ├── seeds/                             # CSV seed data
    ├── snapshots/                         # Type-2 SCD snapshots
    └── tests/                             # Custom singular and generic data tests
```

---

## 🚀 Getting Started

### 1. Prerequisites
- [uv](https://docs.astral.sh/uv/) installed: `brew install uv`
- Access to a Snowflake account with appropriate database and warehouse privileges.

### 2. Install Dependencies
```bash
uv sync
source .venv/bin/activate
```

### 3. Configure Connection Profile
Ensure your `~/.dbt/profiles.yml` or `airbnb_snowflake_dbt_pipeline/profiles.yml` is configured:

```yaml
airbnb_snowflake_dbt_pipeline:
  target: dev
  outputs:
    dev:
      type: snowflake
      account: <YOUR_SNOWFLAKE_ACCOUNT>
      user: <YOUR_USERNAME>
      password: <YOUR_PASSWORD>   # Or use private_key_path for key-pair auth
      role: ACCOUNTADMIN
      warehouse: SNOWFLAKE_LEARNING_WH
      database: AIRBNB
      schema: dbt_schema
      threads: 1
```

### 4. Validate Setup
```bash
cd airbnb_snowflake_dbt_pipeline
dbt debug
```

### 5. Compile and Run Models
```bash
# Compile and check DAG
dbt compile

# Run Bronze models
dbt run --select bronze

# Run Silver models and tests
dbt run --select silver
dbt test --select silver

# Run Gold OBT model and tests
dbt run --select obt
dbt test --select obt
```

---

## 🗺️ Next Steps
- [x] Develop **Silver Layer**: Data cleansing, handling nulls, type-casting, and business transformations (`silver_bookings`, `silver_listings`, `silver_hosts`).
- [x] Create reusable Jinja macros for transformations (`multiply`, `tag`, `trimmer`).
- [x] Implement data quality tests (uniqueness, not-null on primary keys).
- [x] Build **Gold Layer - OBT**: Denormalized One Big Table (`obt`) combining Silver layer models using dynamic Jinja loops.
- [ ] Build **Gold Layer - Star Schema**: Dimensional fact and dimension models (`fact_bookings`, `dim_listings`, `dim_hosts`).
