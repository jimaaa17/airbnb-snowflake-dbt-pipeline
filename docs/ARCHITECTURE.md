# Platform Architecture & Enterprise Production Guide

This document details the complete end-to-end architecture, directory layout, and enterprise production extensions for the Airbnb Snowflake dbt Pipeline.

---

## 1. Complete Repository Directory Layout

```text
Airbnb Snowflake DBT Pipeline/
├── pyproject.toml                                # Project metadata and dependencies (Astral uv)
├── uv.lock                                       # Deterministic dependency lockfile
├── Dockerfile.inference                          # Production inference microservice container
├── .gitignore                                    # Credentials, venvs, and artifacts exclusion
├── README.md                                     # Production documentation & architectural specs
│
├── .github/workflows/                            # Decoupled CI/CD automation pipelines
│   ├── dbt_ci_cd.yml                             # Snowflake dbt build, snapshot & data quality tests
│   └── ml_and_apps_ci_cd.yml                     # ML training, SLA gate, pytest, API smoke & Docker
│
├── airbnb_snowflake_dbt_pipeline/                # Core dbt transformation project
│   ├── dbt_project.yml                           # dbt project configuration & schema mapping
│   ├── macros/                                   # Jinja transformation macros
│   │   ├── generate_schema_name.sql              # Clean bronze/silver/gold schema routing
│   │   ├── multiply.sql                          # Precision multiplication macro
│   │   ├── tag.sql                               # Price tier bucketing macro
│   │   └── trim.sql                              # Whitespace trimming macro
│   ├── models/
│   │   ├── sources/sources.yml                   # Raw staging source definitions
│   │   ├── bronze/                               # Incremental bronze models (watermark filtered)
│   │   ├── silver/                               # Standardized silver models (deduped & cleansed)
│   │   └── gold/                                 # Gold marts: OBT, facts, and semantic_models.yml
│   ├── snapshots/                                # SCD Type 2 dimension snapshots (dim_*)
│   └── tests/                                    # Gatekeeper and reconciliation data tests
│
├── semantic_api/                                 # FastAPI Semantic Gateway
│   ├── main.py                                   # REST API endpoints & MetricFlow query router
│   └── airbnb_semantic_layer_postman_collection.json # Automated Postman test suite
│
├── ml/                                           # Predictive Intelligence & MLOps Platform
│   ├── configs/                                  # Declarative model & feature hyperparameters
│   ├── data/                                     # Snowflake connector & dataset loaders
│   ├── features/                                 # Feature store, cyclical transformers, registry
│   │   ├── feature_store.py                      # Zipline point-in-time as-of sliding windows
│   │   ├── transformers.py                       # Cyclical waves, lead-time & ratio transformers
│   │   └── definitions.py                        # Centralized feature registry metadata
│   ├── models/                                   # Gradient Boosting regression & classification
│   ├── evaluation/                               # Evaluation SLA gates, SHAP attribution, SME metrics
│   │   ├── eval_gate.py                          # Automated CI/CD performance quality gate
│   │   ├── shap_diagnostics.py                   # TreeExplainer feature attribution & cohort drivers
│   │   └── metrics.py                            # Technical & SME financial impact metrics
│   ├── tracking/                                 # MLflow tracking & Model Registry manager
│   │   └── tracker.py                            # SQLite backend & @champion alias staging
│   ├── inference/                                # Real-time & batch inference engines
│   │   ├── service.py                            # FastAPI-integrated prediction service
│   │   └── batch_predictor.py                    # Scalable batch scoring pipeline
│   ├── tests/                                    # Pytest unit, regression & zero-leakage tests
│   ├── train_all.py                              # Master training pipeline & registry promotion
│   └── run_shap_analysis.py                      # CLI tool generating SHAP explanation plots
│
├── apps/                                         # Airbnb Analytics Studio (Streamlit Data App)
│   ├── semantic_bi_app.py                        # Main multi-page navigation shell
│   ├── components/                               # Reusable UI components, header & Airbnb theme
│   ├── assets/                                   # Official Airbnb Bélo vector & branding
│   └── views/                                    # 5 dedicated workspace modules
│       ├── executive_overview.py                 # Executive financial KPIs & YoY performance
│       ├── diagnostic_rca.py                     # Multi-dimensional RCA & A/B hypothesis test
│       ├── metric_explorer.py                    # Self-service slice & dice MetricFlow explorer
│       ├── predictive_studio.py                  # Dynamic pricing, cancellation risk & SHAP
│       └── catalog_hub.py                        # Metric ownership directory & dbt test monitor
│
└── docs/                                         # Architectural reports & enterprise roadmaps
    ├── ARCHITECTURE.md                           # Comprehensive architecture & production extension guide
    ├── EXPERIMENTS.md                            # Experiment lineage, benchmarks & scientific trade-offs
    ├── FEATURE_ENGINEERING.md                    # Deep-dive feature store & mathematical specs
    ├── SEMANTIC_ARCHITECTURE_AND_ML_REPORT.md    # SME architectural guide & workflows
    └── semantic_layer_and_predictive_roadmap.md  # Engineering roadmap & design patterns
```

---

## 2. Medallion Warehouse Modeling Deep Dive

### Incremental Watermark Loading
All Bronze and Silver tables implement incremental merges using a `CREATED_AT` watermark filter:
```sql
{% if is_incremental() %}
    WHERE CREATED_AT > (SELECT COALESCE(MAX(CREATED_AT), '1900-01-01') FROM {{ this }})
{% endif %}
```
This minimizes compute consumption by transforming only net-new or modified records.

### SCD Type 2 Dimension History
Dimension history for listings and hosts is captured using dbt snapshots (`snapshots/dim_*.yml`):
* `strategy: timestamp` tracks changes via `UPDATED_AT`
* Active records maintain a sentinel valid-to date (`9999-12-31`)
* Historical versions enable point-in-time dimensional analysis without mutating transactional facts.

---

## 3. Enterprise Production Extensions

| Layer | Demonstrated in Repository (Live / POC) | Recommended Enterprise Production Extensions |
| :--- | :--- | :--- |
| **Data Warehouse** | Snowflake Medallion (`staging` → `bronze` → `silver` → `gold.obt`), SCD2 snapshots, 82 passing dbt tests. | Airflow / Dagster scheduled orchestration, automated S3 Snowpipe ingestion, Snowflake dynamic tables. |
| **Semantic Layer** | Dynamic MetricFlow YAML parsing via FastAPI Gateway (`semantic_api/main.py`), certified metrics (SSOT). | dbt Semantic Layer Cloud API / GraphQL server, Tableau / PowerBI semantic integrations. |
| **MLOps & Tracking** | Local MLflow tracking backend (`sqlite:///ml/mlruns.db`), `@champion` Model Registry staging, automated CI SLA gates. | Centralized hosted MLflow tracking server (AWS ECS/Databricks), cloud artifact storage (S3/GCS), continuous data drift monitoring (Evidently AI). |
| **Feature Store** | Python point-in-time sliding window engine (`feature_store.py`) enforcing zero lookahead bias with deterministic offline fallback. | Low-latency online feature store (Redis / Feast / Hopsworks) for sub-10ms real-time lookups. |
| **Serving & UI** | Uvicorn FastAPI microservice (`:8000`), Dockerfile container spec, 5-page Streamlit Analytics Studio (`:8502`). | Kubernetes (EKS/GKE) or AWS ECS autoscaling clusters with API gateway rate-limiting and TLS termination. |
