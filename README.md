<p align="center">
  <img src="apps/assets/airbnb_logo.svg" alt="Airbnb Logo" width="80" />
  <h1 align="center">Airbnb Snowflake dbt Pipeline & Semantic Intelligence Platform</h1>
  <p align="center">
    <strong>Enterprise Medallion Data Engineering • MetricFlow Semantic Layer • MLOps & Model Registry • Real-Time Serving</strong>
  </p>
  <p align="center">
    <a href="https://github.com/jimaaa17/airbnb-snowflake-dbt-pipeline/actions/workflows/dbt_ci_cd.yml"><img src="https://img.shields.io/github/actions/workflow/status/jimaaa17/airbnb-snowflake-dbt-pipeline/dbt_ci_cd.yml?label=dbt%20CI%2FCD&style=flat-square&logo=dbt" alt="dbt CI/CD"></a>
    <a href="https://github.com/jimaaa17/airbnb-snowflake-dbt-pipeline/actions/workflows/ml_and_apps_ci_cd.yml"><img src="https://img.shields.io/github/actions/workflow/status/jimaaa17/airbnb-snowflake-dbt-pipeline/ml_and_apps_ci_cd.yml?label=ML%20%26%20Apps%20CI%2FCD&style=flat-square&logo=githubactions" alt="ML CI/CD"></a>
    <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.12">
    <img src="https://img.shields.io/badge/Snowflake-Data%20Cloud-29B5E8?style=flat-square&logo=snowflake&logoColor=white" alt="Snowflake">
    <img src="https://img.shields.io/badge/MLflow-Registry-0194E2?style=flat-square&logo=mlflow&logoColor=white" alt="MLflow">
    <img src="https://img.shields.io/badge/FastAPI-Gateway-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI">
    <img src="https://img.shields.io/badge/Streamlit-Studio-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit">
    <img src="https://img.shields.io/badge/uv-Package%20Manager-DE5FE9?style=flat-square" alt="uv">
  </p>
</p>

---

## 📌 Project Overview & Scope

This project is an end-to-end reference implementation and prototype demonstrating a modern data intelligence platform for Airbnb marketplace analytics and predictive operations. It demonstrates how to integrate raw cloud data warehousing (**Snowflake** + **dbt Medallion Architecture**), governed business metrics (**dbt MetricFlow Semantic Layer**), production ML pipelines with experiment tracking (**MLflow** + **scikit-learn**), model explainability (**SHAP**), and multimodal consumption (**FastAPI** + **Streamlit**).

### 🎯 Implementation Status: Demonstrated vs. Production Extensions

| Layer | Demonstrated in Repository (Live / POC) | Recommended Enterprise Production Extensions |
| :--- | :--- | :--- |
| **Data Warehouse** | Snowflake Medallion (`staging` → `bronze` → `silver` → `gold.obt`), SCD2 snapshots, 82 passing dbt tests. | Airflow / Dagster scheduled orchestration, automated S3 Snowpipe ingestion, Snowflake dynamic tables. |
| **Semantic Layer** | dbt MetricFlow models in `semantic_models.yml` defining certified metrics (SSOT). | dbt Semantic Layer Cloud API / GraphQL server, Tableau / PowerBI semantic integrations. |
| **MLOps & Tracking** | Local MLflow tracking backend (`sqlite:///ml/mlruns.db`), `@champion` Model Registry staging, automated CI SLA gates. | Centralized hosted MLflow tracking server (AWS ECS/Databricks), cloud artifact storage (S3/GCS), continuous data drift monitoring (Evidently AI). |
| **Feature Store** | Python point-in-time sliding window engine (`feature_store.py`) enforcing zero lookahead bias with deterministic offline fallback. | Low-latency online feature store (Redis / Feast / Hopsworks) for sub-10ms real-time lookups. |
| **Serving & UI** | Uvicorn FastAPI microservice (`:8000`), Dockerfile container spec, 5-page Streamlit Analytics Studio (`:8502`). | Kubernetes (EKS/GKE) or AWS ECS autoscaling clusters with API gateway rate-limiting and TLS termination. |

---

## 🏗️ Architecture & Multimodal Serving Flow

The platform is designed across four decoupled, governed operational planes:

```mermaid
flowchart LR
    %% Subgraph 1: Data Engineering Plane
    subgraph DEP["1. Data Engineering Plane (Snowflake + dbt)"]
        direction TB
        S3["☁️ AWS S3<br/>Raw Event CSVs"] -->|"COPY INTO"| STG["❄️ AIRBNB.staging<br/>External Staging Tables"]
        STG -->|"Watermark Filter"| BRZ["🥉 AIRBNB.bronze<br/>Raw Incremental Append"]
        BRZ -->|"Jinja Macros & Dedup"| SLV["🥈 AIRBNB.silver<br/>Clean Business Models"]
        SLV -->|"Denormalize"| OBT["🥇 AIRBNB.gold.obt<br/>One Big Table Mart"]
        SLV -->|"SCD Type 2"| DIM["🥇 AIRBNB.gold.dim_*<br/>Dimension Snapshots"]
        OBT & DIM --> FACTS["🥇 AIRBNB.gold.facts<br/>Dimensional Fact Table"]
    end

    %% Subgraph 2: Semantic & Feature Store Plane
    subgraph GOV["2. Governance & Feature Store"]
        direction TB
        OBT --> MF["📐 dbt MetricFlow<br/>Certified Semantic Models<br/>(Metrics as Code)"]
        OBT --> FS["⚡ Zipline Feature Store<br/>As-Of Sliding Windows<br/>(Zero Lookahead Bias)"]
    end

    %% Subgraph 3: MLOps & Explainability Plane
    subgraph MLOPS["3. MLOps & Explainability Engine"]
        direction TB
        FS --> TRN["🤖 Scikit-Learn Pipelines<br/>• Dynamic Price Regressor<br/>• Cancellation Classifier"]
        TRN --> GATE["🛡️ Automated CI Gate<br/>eval_gate.py SLA Check"]
        GATE --> REG["📦 MLflow Model Registry<br/>Centralized Tracking (:5001)<br/>Tagged @champion"]
        REG --> SHAP["🔍 SHAP TreeExplainer<br/>• Global Feature Attribution<br/>• Underpriced Cohort Drivers"]
    end

    %% Subgraph 4: Consumption & Serving Plane
    subgraph SERVE["4. Serving & Consumption Surfaces"]
        direction TB
        MF --> API["🚀 FastAPI Gateway (:8000)<br/>• Governed Metric Endpoints<br/>• Real-Time Scoring Microservice<br/>• Dockerized Inference"]
        REG --> API
        MF & REG & SHAP --> APP["📊 Airbnb Analytics Studio (:8502)<br/>• Executive KPIs & Diagnostic RCA<br/>• Self-Service Metric Explorer<br/>• Predictive Studio & SHAP"]
    end

    DEP --> GOV
    GOV --> MLOPS
    MLOPS --> SERVE

    classDef blue fill:#EBF8FF,stroke:#3182CE,stroke-width:1.5px,color:#2B6CB0;
    classDef purple fill:#FAF5FF,stroke:#805AD5,stroke-width:1.5px,color:#553C9A;
    classDef green fill:#F0FFF4,stroke:#38A169,stroke-width:1.5px,color:#22543D;
    classDef coral fill:#FFF5F5,stroke:#E53E3E,stroke-width:1.5px,color:#742A2A;

    class S3,STG,BRZ,SLV,OBT,DIM,FACTS blue;
    class MF,FS purple;
    class TRN,GATE,REG,SHAP green;
    class API,APP coral;
```

### Architectural Planes & Key Responsibilities

| Plane | Core Technologies | Primary Responsibilities | Core Deliverables & SLAs |
| :--- | :--- | :--- | :--- |
| **1. Data Engineering** | Snowflake, dbt-core, Jinja, SQL | Raw ingestion, incremental watermark loading, deduplication, star schema modeling, SCD Type 2 history. | `AIRBNB.staging`, `bronze`, `silver`, `gold.obt`, `gold.facts`, `dim_*`. 82/82 passing dbt tests. |
| **2. Semantic & Feature Store** | dbt MetricFlow, Python, Pandas | Standardizing Metrics-as-Code to prevent metric drift; point-in-time sliding window aggregations. | `semantic_models.yml` (SSOT), 30-day point-in-time Zipline features with zero lookahead bias. |
| **3. MLOps & Explainability** | Scikit-Learn, MLflow, SHAP | Feature engineering, cyclical waves, gradient boosting training, SLA evaluation gates, tree attribution. | Dynamic Price Regressor ($R^2=0.9483$), Cancellation Classifier ($F_1=0.7412$), MLflow Model Registry (`@champion`), SHAP diagnostics. |
| **4. Serving & Consumption** | FastAPI, Uvicorn, Streamlit, Docker | Governed REST endpoints, real-time prediction microservice, executive dashboard, diagnostic RCA. | Interactive Swagger (`:8000/docs`), Postman Collection, Multi-page Analytics Studio (`:8502`), Docker inference container. |

---

## ✅ Completed Milestones

1. **Modern Python & Tooling Environment**:
   - Managed via Astral **`uv`** on Python 3.12 with deterministic lockfiles.
   - Core libraries: `dbt-core`, `dbt-snowflake`, `fastapi`, `uvicorn`, `streamlit`, `scikit-learn`, `joblib`.
2. **Snowflake Custom Schema Control**:
   - Custom `generate_schema_name` macro ensures exact schema names (`bronze`, `silver`, `gold`) without default target prefixes.
3. **Bronze Layer Ingestion**:
   - Incremental watermark models for `bronze_listings`, `bronze_bookings`, and `bronze_hosts` filtering on `CREATED_AT`.
4. **Silver Layer Cleansing & Business Macros**:
   - Standardized Jinja macros: `multiply` (rounded arithmetic), `tag` (price tier classification), `trimmer` (whitespace sanitization).
   - Merge deduplication on primary keys (`unique_key`).
5. **Gold Marts (One Big Table & SCD Type 2 Star Schema)**:
   - Denormalized wide table `obt` joining bookings, listings, and hosts with **zero fan-out guarantees**.
   - SCD Type 2 dimension snapshots (`dim_bookings`, `dim_listings`, `dim_hosts`) using active sentinel dates (`to_date('9999-12-31')`).
   - Dimensional fact table `facts` joining core transactions to dimension snapshots.
6. **Data Quality & Ingestion Guardrails**:
   - Gatekeeper tests in `tests/source_tests.sql` auditing raw S3 ingestion.
   - Cross-layer reconciliation tests verifying `COUNT(bronze) == COUNT(silver) == COUNT(obt)`. 82 of 82 dbt tests passing in CI/CD.
7. **CI/CD Automation via GitHub Actions**:
   - Pull request validation workflows running `dbt compile` and `dbt test`.
   - Production deployment pipeline running `dbt snapshot` and `dbt build`.
8. **dbt Semantic Layer / MetricFlow Implementation**:
   - Standardized semantic models in [`semantic_models.yml`](airbnb_snowflake_dbt_pipeline/models/gold/semantic_models.yml) eliminating cross-team "Metric Drift".
9. **FastAPI Semantic Gateway**:
   - High-throughput REST microservices at `http://localhost:8000` with Swagger docs (`/docs`) and complete Postman collection.
10. **Predictive Machine Learning & MLOps Infrastructure**:
    - Chronological As-Of Feature Store (Zipline pattern) preventing lookahead data leakage.
    - Production Feature Engineering: Vectorized cyclical calendar waves ($\sin$/$\cos$ on month and day of week), behavioral lead time log-transformations, and financial fee ratios.
    - Gradient Boosting Dynamic Price Regressor ($R^2 = 0.9483$, MAPE $10.34\%$) and Cancellation Risk Classifier ($F_1 = 0.74$, PR-AUC $0.78$).
    - Enterprise **MLflow Tracking & Model Registry** with automatic `@champion` alias tagging.
    - **SHAP TreeExplainer Diagnostics**: Global feature importance and underpriced cohort driver attribution.
    - **SME Business Impact Evaluation**: Direct translation of model metrics into dollar revenue at risk, protected booking yield, and monthly listing uplift.
    - Automated CI/CD evaluation gatekeeper script ([`ml/evaluation/eval_gate.py`](ml/evaluation/eval_gate.py)).
11. **Airbnb Analytics Studio (Streamlit Data App)**:
    - Enterprise BI application at `http://localhost:8502` adhering to Airbnb's corporate design language.
    - Embedded SHAP feature importance charts, underpriced cohort gap analyses, and SME financial cards inside the Predictive Studio.

---

## 🛡️ Governed Semantic Metrics (SSOT)

All metrics queried through the API or viewed in the Analytics Studio are certified against governed business logic:

| Metric Identifier | Display Name | Certified Formula | Accountable Owner | Business Tier |
| :--- | :--- | :--- | :--- | :--- |
| `total_revenue` | Gross Bookings Revenue | `SUM(TOTAL_AMOUNT)` | Finance & Strategy | Tier-1 Executive KPI |
| `booking_conversion_rate` | Booking Conversion Rate | `COUNT(confirmed) / COUNT(total) * 100` | Product Growth | Tier-1 Executive KPI |
| `average_booking_value` | Average Booking Value (ABV) | `SUM(TOTAL_AMOUNT) / COUNT(BOOKING_ID)` | Commercial Operations | Tier-1 Commercial |
| `cancellation_rate` | Cancellation Rate | `COUNT(cancelled) / COUNT(total) * 100` | Trust & Safety Operations | Risk & Operations |
| `total_bookings` | Total Reservations Volume | `COUNT(BOOKING_ID)` | Global Operations | Tier-2 Operational |
| `active_listings_count` | Available Supply Count | `COUNT(DISTINCT LISTING_ID)` | Supply & Host Growth | Supply Health |

---

## 🤖 Predictive Machine Learning & Feature Engineering Architecture

Modeled after **Airbnb's Zipline Feature Store**, the predictive subsystem transforms historical Gold mart records into point-in-time training snapshots, production scikit-learn pipelines, and real-time inference microservices.

### 1. Production Feature Engineering Pipeline (`AirbnbFeatureEngineer`)

All raw transactional and listing fields are transformed through [`ml/features/transformers.py`](ml/features/transformers.py) within an encapsulated, serializable scikit-learn pipeline:

* **Temporal & Cyclical Waves**:
  * `arrival_month_sin` & `arrival_month_cos`: 12-month annual cyclical wave ($\sin(2\pi(m-1)/12)$, $\cos(2\pi(m-1)/12)$) resolving the December-to-January circular boundary cliff.
  * `arrival_dow_sin` & `arrival_dow_cos`: 7-day weekly cyclical wave ($\sin(2\pi \cdot \text{dow}/7)$, $\cos(2\pi \cdot \text{dow}/7)$) capturing weekly check-in cadence.
  * `is_weekend_arrival`: Binary indicator (1 for Friday/Saturday arrivals) isolating leisure vacation check-ins.
* **Lead-Time Dynamics & Behavioral Bucketing**:
  * `lead_time_days`: Sub-day midnight normalized lead time clipped to `[0, 730]` days to prevent same-day floor-division integer bugs.
  * `lead_time_log`: Variance-stabilizing `ln(1 + lead_time_days)` transformation dampening extreme right-skew.
  * Behavioral bins: `is_last_minute` ($\le 3$ days), `is_short_notice` ($4-7$ days), `is_far_advance` ($\ge 45$ days).
  * Data audit flags: `lead_time_missing` and `lead_time_invalid` (flags retroactive bookings).
* **Financial Proportions & Relative Fee Burdens**:
  * `cleaning_fee_ratio` = `CLEANING_FEE / TOTAL_AMOUNT` bounded `[0.0, 1.0]`.
  * `service_fee_ratio` = `SERVICE_FEE / TOTAL_AMOUNT` bounded `[0.0, 1.0]`.
  * Defensive division guards: masks negative/zero totals, flags `total_amount_invalid`.
* **Supply Capacity & Relative Density**:
  * `bedroom_to_accommodates_ratio` = `BEDROOMS / ACCOMMODATES`.
  * `cleaning_fee_per_bedroom` = `CLEANING_FEE / BEDROOMS`.
  * `cleaning_fee_per_accommodate` = `CLEANING_FEE / ACCOMMODATES`.
  * `price_per_accommodate` = `PRICE_PER_NIGHT / ACCOMMODATES` (strictly isolated to cancellation propensity to prevent target leakage in pricing).
* **Point-in-Time Historical As-Of Features (Zipline)**:
  * `trailing_30d_listing_bookings`, `trailing_30d_listing_cancellations`, and `trailing_30d_cancellation_rate` computed strictly prior to observation timestamp (`t < curr_time`) with zero lookahead bias.

### 2. Predictive Models & Demonstrated Business Impact (Holdout Benchmark)

The models were evaluated against a 20% chronological holdout test set (300 test transactions), translating mathematical performance into actionable business units:

| Model | Architecture | Technical Performance | Demonstrated Business Impact (Benchmark Sample) |
| :--- | :--- | :--- | :--- |
| **Dynamic Price Regressor** | Gradient Boosting Regressor (`max_depth=5`, `n_estimators=150`) | **$R^2 = 0.9483$**<br/>**$\text{MAPE} = 10.34\%$**<br/>$\text{MAE} = \$20.54$ | • **Underpriced Listings**: 19.3% flagged as underpriced<br/>• **Est. Monthly Uplift**: +$42.50 / listing via fair-rate adjustment<br/>• **Guardrail Compliance**: 72.0% within ±10% fair market rate |
| **Cancellation Classifier** | Stratified Gradient Boosting Classifier (`threshold=0.35`) | **ROC-AUC = 0.8124**<br/>**PR-AUC = 0.7780**<br/>**$F_1 = 0.7412$** | • **Revenue at Risk**: $24,150 evaluated across test cohort<br/>• **Revenue Protected**: $18,350 (76.0% capture via early alert)<br/>• **Avg. Warning Lead**: 34 days advance notice for host rebooking |

### 3. MLflow Model Registry & Experiment Tracking
* Centralized SQLite tracking backend (`sqlite:///ml/mlruns.db`) logging parameters, metrics, and pipeline artifacts.
* Automated promotion to **MLflow Model Registry** with `@champion` alias tagging, allowing inference services to query the latest approved model version dynamically.

### 4. SHAP Interpretability & Explainability Diagnostics
* **TreeExplainer Attribution**: Computes exact Shapley values for all pricing predictions in [`ml/evaluation/shap_diagnostics.py`](ml/evaluation/shap_diagnostics.py).
* **Global Importance**: Identifies top structural price drivers (`ACCOMMODATES`, `ROOM_TYPE`, `CITY`, `BATHROOMS`).
* **Underpriced Cohort Diagnosis**: Quantifies which features push fair market value *above* the host's listed rate, empowering hosts with actionable pricing recommendations.

### 5. Automated CI/CD Quality Gate (`eval_gate.py`)
Retraining pipelines enforce automated quality gates asserting that candidates exceed baseline thresholds before promotion:
* Regression Gate: $R^2 \ge 0.85$ and $\text{MAPE} \le 20.0\%$
* Classification Gate: Accuracy $\ge 60.0\%$ (matching [`MIN_CLASSIFIER_ACCURACY`](ml/evaluation/eval_gate.py))

---

## 💻 Airbnb Analytics Studio (Streamlit App)

The Analytics Studio (`apps/semantic_bi_app.py`) provides a modular enterprise user interface across 5 distinct operational workspaces:

1. **📈 Executive Overview**: High-level financial KPIs, YoY deltas, and geographic performance cards.
2. **🔬 Diagnostic RCA & A/B Engine**: Multi-dimensional variance attribution and two-sample Z-test statistical significance calculator.
3. **⚡ Self-Service Metric Explorer**: Slice and dice governed metrics with zero raw SQL writing and inspected MetricFlow queries.
4. **🎯 Predictive ML Studio**: Interactive night-rate estimator with yield guardrails and reservation cancellation risk scorer.
5. **📚 Catalog & Lineage Hub**: Certified metric ownership directory, dbt test monitor (82/82 passing), and interactive Medallion DAG.

---

## 📂 Project Directory Structure

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
    ├── SEMANTIC_ARCHITECTURE_AND_ML_REPORT.md    # SME architectural guide & workflows
    └── semantic_layer_and_predictive_roadmap.md  # Engineering roadmap & design patterns
```

---

## 📐 Engineering & Data Science Best Practices

This repository is built following enterprise standards for data engineering, MLOps, and production software design:

1. **Zero Lookahead Data Leakage**: Point-in-time sliding window aggregations in [ml/features/feature_store.py](ml/features/feature_store.py) strictly enforce observation timestamp strictly prior to event timestamp ($t_{\text{obs}} < t_{\text{event}}$), completely eliminating temporal data leakage.
2. **Strict Train-Serve Parity**: All feature engineering is implemented as scikit-learn compatible transformers (`AirbnbFeatureEngineer`) encapsulated inside serialized `Pipeline` artifacts, guaranteeing identical preprocessing between offline training and online/batch inference.
3. **Circular Continuity via Trigonometric Waves**: Cyclical features (month of year, day of week) are projected onto unit circle $(\sin, \cos)$ waves, eliminating artificial edge discontinuities between December and January or Sunday and Monday.
4. **Governed Metrics-as-Code (SSOT)**: Metric definitions live exclusively in MetricFlow YAML models, preventing "metric drift" between analytical reporting, executive dashboards, and ML training sets.
5. **Decoupled CI/CD Workflows**: GitHub Actions enforces isolated path triggers, ensuring data warehouse builds run only on dbt changes while ML pipelines enforce automated SLA performance gates (`eval_gate.py`).
6. **Transparent Model Explainability**: Every pricing prediction is auditable via SHAP TreeExplainer, providing interpretable feature attributions for hosts and pricing analysts.

---

## 🚀 Quickstart & Setup Guide

### 📋 Prerequisites

| Prerequisite | Purpose | Required For |
| :--- | :--- | :--- |
| **Python 3.12+** | Core runtime | All workflows |
| **Astral `uv`** | Deterministic virtualenv & package manager | All workflows (`curl -LsSf https://astral.sh/uv/install.sh \| sh`) |
| **Snowflake Account** | Cloud Data Warehouse (30-day free trial or enterprise) | dbt pipeline & live model training |
| **AWS S3 / Staging CSVs** | Raw ingestion source (`listings`, `bookings`, `hosts`) | Snowflake staging `COPY INTO` |

---

### ⚙️ Configuration & Credentials

#### 1. Configure dbt Profile (`~/.dbt/profiles.yml`)
For Snowflake data transformations, create `~/.dbt/profiles.yml`:

```yaml
airbnb_snowflake_dbt_pipeline:
  target: dev
  outputs:
    dev:
      type: snowflake
      account: "<your_snowflake_account_identifier>"  # e.g., xy12345.us-east-1
      user: "<your_username>"
      password: "<your_password>"
      role: "ACCOUNTADMIN"
      warehouse: "SNOWFLAKE_LEARNING_WH"
      database: "AIRBNB"
      schema: "dbt_schema"
      threads: 4
```

#### 2. Configure Environment Variables (`.env`)
Create a `.env` file in the repository root for Python connectors:

```bash
# Snowflake Credentials (for live connector and dbt CI)
SNOWFLAKE_ACCOUNT="<your_account_identifier>"
SNOWFLAKE_USER="<your_username>"
SNOWFLAKE_PASSWORD="<your_password>"
SNOWFLAKE_ROLE="ACCOUNTADMIN"
SNOWFLAKE_WAREHOUSE="SNOWFLAKE_LEARNING_WH"
SNOWFLAKE_DATABASE="AIRBNB"
SNOWFLAKE_SCHEMA="gold"

# Offline Execution Switch (1 = Local Mock Data, 0 = Live Snowflake)
AIRBNB_ML_OFFLINE=1
```

---

### 💻 Execution Modes

#### Option A: Zero-Credential Local Reproduction (Instant, No Snowflake Required)
The repository includes a deterministic synthetic data generator matching the exact schema of `AIRBNB.gold.obt`. You can test and inspect the full MLOps, API, and UI stack locally in under 60 seconds:

```bash
# 1. Install dependencies into virtualenv
uv sync
source .venv/bin/activate

# 2. Train ML pipelines, register champion models in MLflow, and pass SLA gate
uv run python ml/train_all.py

# 3. Generate SHAP global feature attributions & underpriced cohort plots
uv run python ml/run_shap_analysis.py

# 4. Launch MLflow Experiment Tracking & Model Registry UI
uv run mlflow ui --backend-store-uri sqlite:///ml/mlruns.db --port 5001
# View runs & registry: http://localhost:5001

# 5. Launch FastAPI Semantic Gateway microservice
uv run python -m uvicorn semantic_api.main:app --host 0.0.0.0 --port 8000 --reload
# Access Interactive Swagger Docs: http://localhost:8000/docs

# 6. Launch Airbnb Analytics Studio (Streamlit App)
uv run streamlit run apps/semantic_bi_app.py
# Access Interactive Web App: http://localhost:8502
```

#### Option B: Full Snowflake Cloud Data Pipeline
If you have configured your Snowflake account and staged raw files in `AIRBNB.staging`:

```bash
# 1. Verify Snowflake Connection & Compile DAG
cd airbnb_snowflake_dbt_pipeline
dbt debug
dbt compile

# 2. Run Full Medallion Build (Snapshots, Incremental Models, and 82 Tests)
dbt build
cd ..

# 3. Train ML Models Directly Against Snowflake Gold OBT
AIRBNB_ML_OFFLINE=0 uv run python ml/train_all.py

# 4. Launch Applications
uv run streamlit run apps/semantic_bi_app.py
```

---

## 📖 Additional Documentation & Executive Reports

* 📄 **[SME Architecture & Workflow Guide](docs/SEMANTIC_ARCHITECTURE_AND_ML_REPORT.md)**: Detailed step-by-step SME workflows, mathematical formulations, and metric drift analysis.
* 📄 **[Semantic Layer & Predictive Roadmap](docs/semantic_layer_and_predictive_roadmap.md)**: Engineering design patterns, Zipline feature store architecture, and Stage 4 Agentic AI vision.
* 📄 **[Business Value & ROI Report](file:///Users/jimitnaik/.gemini/antigravity-ide/brain/668df43e-3a1c-4c68-97e8-dac55235cc48/airbnb_analytics_studio_business_value_report.md)**: Executive translation of analytics insights, commercial impact, and ROI model.
