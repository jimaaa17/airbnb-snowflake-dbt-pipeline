# Enterprise Semantic Architecture & ML Insights Translation Report

**Target Audience:** Subject Matter Experts (SMEs), Analytics Leaders, Product Managers, and Data Engineers  
**Platform:** Airbnb Analytics Studio & Semantic Gateway  
**Data Infrastructure:** Snowflake Medallion Marts, dbt Core / MetricFlow, FastAPI, Streamlit, Scikit-Learn  

---

## 1. Executive Summary & The Problem Solved

### 1.1 The "Metric Drift" Dilemma in Traditional Enterprises
In traditional business intelligence setups (using legacy Tableau or Power BI reports), multiple departments independently write custom SQL queries or create calculated fields to compute business metrics. 

This inevitably produces **Metric Drift**:
* **Example:** The *Product Growth* team calculates **Booking Conversion Rate** as `confirmed_bookings / search_sessions`, while the *Operations* team calculates it as `confirmed_bookings / total_booking_attempts`.
* **Consequence:** Two executives enter the same quarterly review with conflicting metrics, eroding organizational trust and slowing critical decision-making.

### 1.2 The Semantic Layer as the Single Source of Truth (SSOT)
By defining business logic as code in the **dbt Semantic Layer / MetricFlow** directly on top of **Snowflake**, all consumers—whether they are executive dashboards, operational APIs, batch pipelines, or future Agentic AI systems—query the exact same governed definition.

```
       [ Snowflake Warehouse: Gold Marts (AIRBNB.gold.obt) ]
                               │
                               ▼
        [ dbt Semantic Layer / MetricFlow (Metric-as-Code) ]
                               │
          ┌────────────────────┴────────────────────┐
          ▼                                         ▼
[ FastAPI Semantic Gateway ]              [ Airbnb Analytics Studio ]
   • REST Endpoints (:8000)                  • Executive Overview
   • Automated Postman Suite                 • Diagnostic RCA & A/B
   • Future AI Agents (Stage 4)              • Predictive ML Studio
                                             • Metric Explorer & Catalog
```

---

## 2. End-to-End Data Pipeline Architecture (Medallion Flow)

The data infrastructure transforms raw event streams into curated, certified dimensions and marts across four distinct architectural tiers:

| Tier | Snowflake Schema | Primary Responsibility | Materialization & Governance Pattern |
| :--- | :--- | :--- | :--- |
| **Ingestion** | `AIRBNB.staging` | Raw landing from AWS S3 via `COPY INTO` | External batch loader outside dbt |
| **Bronze** | `AIRBNB.bronze` | Clean schema casting, audit columns | Incremental table with `CREATED_AT` watermark |
| **Silver** | `AIRBNB.silver` | Deduplication, standardized macros, entity keys | Incremental merge on `*_ID` (`multiply`, `tag`, `trimmer`) |
| **Gold Marts** | `AIRBNB.gold` | Dimensional Star Schema & Denormalized OBT | `obt.sql` (OBT) and `facts.sql` with SCD Type 2 `dim_*` |

### Architectural Invariants & Quality Assurance
* **Zero Fan-Out Guarantee:** 82 out of 82 dbt tests pass continuously in CI/CD. Singular reconciliation tests verify that `COUNT(bronze) == COUNT(silver) == COUNT(obt)`.
* **SCD Type 2 Historical Tracking:** Dimension changes in `dim_listings` and `dim_hosts` track validity using active sentinel dates (`to_date('9999-12-31')`).

---

## 3. Governed Semantic Metrics Catalog

All metrics displayed in the user application or queried via API are audited against the following certified definitions:

| Metric Identifier | Business Display Name | Governed Formula | Accountable Owner | Business Tier |
| :--- | :--- | :--- | :--- | :--- |
| `total_revenue` | Total Gross Bookings Revenue | `SUM(TOTAL_AMOUNT)` | Finance & Revenue Operations | Tier-1 Executive KPI |
| `booking_conversion_rate` | Booking Conversion Rate | `COUNT(confirmed) / COUNT(total) * 100` | Product Growth (SSOT) | Tier-1 Executive KPI |
| `average_booking_value` | Average Booking Value (ABV) | `SUM(TOTAL_AMOUNT) / COUNT(BOOKING_ID)` | Commercial Strategy | Tier-1 Commercial |
| `cancellation_rate` | Cancellation Rate | `COUNT(cancelled) / COUNT(total) * 100` | Trust & Safety Operations | Risk & Operations |
| `total_bookings` | Total Reservations Volume | `COUNT(BOOKING_ID)` | Global Operations | Tier-2 Operational |
| `active_listings_count` | Active Supply Count | `COUNT(DISTINCT LISTING_ID)` | Supply & Host Growth | Supply Health |

---

## 4. Machine Learning & Predictive Modeling (Stages 1 to 3)

The pipeline transitions from descriptive and diagnostic analytics directly into production-grade predictive intelligence, inspired by **Airbnb's Zipline Feature Store architecture**.

### 4.1 Chronological Point-in-Time Feature Store
* **Leakage Prevention:** Real-world booking systems suffer from future lookahead leakage if historical aggregations include data timestamped after the reservation creation.
* **30-Day Sliding As-Of Joins:** The feature store computes rolling host performance, historical lead time, and regional market density using strict As-Of chronological boundaries prior to `BOOKING_CREATED_AT`.

### 4.2 Production Models & Evaluation Benchmarks

#### Model 1: Dynamic Fair Price Regressor
* **Algorithm:** Gradient Boosting Regressor with Scikit-learn Pipeline preprocessors.
* **Performance Benchmark:**
  * **$R^2$ Score:** `0.9483` (Explains ~95% of market price variance)
  * **Mean Absolute Percentage Error (MAPE):** `10.34%`
  * **Root Mean Squared Error (RMSE):** `$25.78`
  * **Mean Absolute Error (MAE):** `$20.54`
* **Business Application:** Recommends fair market nightly rates along with bounded **Pricing Guardrails** (Floor: `-15%`, Ceiling: `+25%`) to maximize listing occupancy while defending revenue yield.

#### Model 2: Booking Cancellation Propensity Classifier
* **Algorithm:** Gradient Boosting Classifier with stratified class balancing.
* **Performance Benchmark:**
  * **Holdout Accuracy:** `65.33%`
  * **PR-AUC:** `0.3176` (Adjusted for reservation imbalance)
* **Business Application:** Identifies high-risk reservations at booking time to trigger automated retention workflows (e.g., non-refundable discount incentives or proactive host messaging).

### 4.3 Automated CI/CD Quality Gates
Every model retraining run in GitHub Actions must pass the automated evaluation gate (`ml/evaluation/eval_gate.py`):
1. **Regression Gate:** $R^2 \ge 0.85$ and $\text{MAPE} \le 20.0\%$
2. **Classification Gate:** Accuracy $\ge 60.0\%$
3. **Artifact Integrity:** SHA-256 serialization verification

---

## 5. SME Application Guide: Using the Airbnb Analytics Studio

The application has been modularized as a scalable enterprise platform featuring 5 distinct operational workspaces:

```
Airbnb Analytics Studio
├── Executive Analytics
│   ├── 📈 Executive Overview: High-level KPI cards with YoY deltas and market trends.
│   ├── 🔬 Diagnostic RCA & A/B: Root-cause decomposition and 2-sample Z-test calculator.
│   └── ⚡ Metric Explorer: Self-service slicing without writing raw SQL.
├── Predictive Intelligence
│   └── 🎯 Predictive ML Studio: Interactive pricing estimator & cancellation risk scorer.
└── Data Governance
    └── 📚 Catalog & Lineage Hub: Certified metric dictionary, dbt test monitor, & Medallion DAG.
```

### 5.1 Step-by-Step SME Workflows

#### Workflow A: Investigating a Market Dip (Growth Lead)
1. Navigate to **"Executive Overview"** to confirm regional revenue and conversion variance.
2. Switch to **"Diagnostic RCA & A/B"** $\rightarrow$ select `Booking Conversion Rate (%)` $\rightarrow$ decompose across `PRICE_PER_NIGHT_TAG` or `CITY`.
3. Isolate the underperforming cohort and formulate a mitigation hypothesis.
4. If testing a new incentive, use the **A/B Experimentation Engine** to calculate the required sample size and verify statistical significance ($p < 0.05$).

#### Workflow B: Pricing a New Listing (Host Operations SME)
1. Navigate to **"Predictive ML Studio"** $\rightarrow$ open **"Dynamic Pricing & Guardrails"**.
2. Input the listing attributes: Market City, Room Type, Capacity, Bathrooms, and Host Quality.
3. Review the **Estimated Fair Market Price** and adhere to the suggested **Floor and Ceiling Guardrails**.

#### Workflow C: Auditing Metric Definitions (Financial Auditor)
1. Navigate to **"Catalog & Lineage Hub"**.
2. Inspect the **Certified Formula** and **Accountable Owner** for each Tier-1 KPI.
3. Verify test pass rates (82/82 dbt tests) and view the dependency flow in the **Interactive Medallion DAG**.
