# Airbnb Feature Store & Feature Engineering Specification

**Target Audience:** Machine Learning Engineers, Data Scientists, and Backend Engineers  
**Subsystems:** Feature Store (`ml/features/feature_store.py`), Feature Transformers (`ml/features/transformers.py`), Model Pipelines (`ml/models/`)  
**Design Reference:** Inspired by Airbnb's Zipline Feature Store Architecture & Scikit-Learn Pipeline Standards  

---

## 1. Architectural Overview & Design Principles

The predictive subsystem bridges historical Snowflake Gold Marts (`AIRBNB.gold.obt`) to real-time inference endpoints through a governed, dual-phase feature engineering pipeline:

```
[ Snowflake Gold OBT / Offline Fixture ]
                   │
                   ▼
  [ Zipline Point-in-Time Engine ]
    • As-of sliding windows (30-day lookback)
    • Strict temporal condition: t < curr_time
    • Zero future-to-past lookahead leakage
                   │
                   ▼
     [ AirbnbFeatureEngineer ]  <── Encapsulated in scikit-learn Pipeline
    • Vectorized calendar & cyclical waves
    • Robust lead-time normalization
    • Bounded financial ratios & division guards
    • Supply density & host attributes
                   │
                   ▼
        [ ColumnTransformer ]
    • Imputation (Median / Constant)
    • Scaling (StandardScaler)
    • One-Hot Encoding (handle_unknown="ignore")
                   │
                   ▼
    [ Gradient Boosting Estimators ]
    (Dynamic Price Regressor / Cancellation Classifier)
```

### Core Design Guarantees
1. **Strict Train-Serve Parity**: All feature transformations are encapsulated in custom, serializable scikit-learn transformers (`AirbnbFeatureEngineer` in [`ml/features/transformers.py`](../ml/features/transformers.py)) embedded directly into the persisted model pipeline. Both offline model training and real-time FastAPI endpoints (`/predict/price`, `/predict/cancellation`) execute identical code paths.
2. **Zero Lookahead Leakage**: Temporal features and sliding-window aggregations strictly exclude observations occurring at or after the prediction timestamp (`t < curr_time`).
3. **Defensive Typing & Zero Div/0 Crashes**: Financial ratios and capacity metrics implement strict mathematical guards (masking non-positive totals, clipping ratios to `[0.0, 1.0]`, and tracking invalid data indicators).

---

## 2. Feature Catalog & Mathematical Formulations

### 2.1 Calendar Seasonality & Cyclical Waves
Raw calendar numbers (such as month 12 and month 1) suffer from artificial numeric boundary cliffs in tree and linear models. Cyclical trigonometric transformations project these features onto a continuous unit circle.

| Feature Name | Type | Mathematical Formula / Derivation | Business Rationale & Guardrails |
| :--- | :--- | :--- | :--- |
| `arrival_month` | Float | `month ∈ [1, 12]` (defaults to 6.0 if missing) | Base calendar month of check-in date. |
| `arrival_month_sin` | Float | `sin(2π * (month - 1) / 12)` | Cyclical annual wave resolving the December (m=12) to January (m=1) circular continuity. |
| `arrival_month_cos` | Float | `cos(2π * (month - 1) / 12)` | Orthogonal cyclical component completing the 2D annual seasonal circle. |
| `arrival_dow` | Float | `dow ∈ [0, 6]` (0 = Monday, 6 = Sunday) | Day of week index of check-in. |
| `arrival_dow_sin` | Float | `sin(2π * dow / 7)` | Weekly cyclical cadence resolving Sunday (6) to Monday (0) transition. |
| `arrival_dow_cos` | Float | `cos(2π * dow / 7)` | Orthogonal weekly cyclical component. |
| `is_weekend_arrival`| Binary | `1 if dow in [4, 5] else 0` | Flags Friday and Saturday check-ins isolating high-demand leisure vacation travel. |
| `arrival_quarter` | Float | `ceil(month / 3) ∈ [1, 4]` | Macro-quarter indicator for quarterly seasonal demand. |
| `arrival_date_missing` | Binary | `1 if arrival_date is NaT else 0` | Audit indicator tracking missing or corrupt reservation arrival dates. |

---

### 2.2 Lead-Time Dynamics & Behavioral Bucketing
Lead time (the gap between reservation creation and check-in) is the single strongest behavioral predictor of cancellation likelihood and pricing elasticity.

| Feature Name | Type | Mathematical Formula / Derivation | Business Rationale & Guardrails |
| :--- | :--- | :--- | :--- |
| `lead_time_days` | Float | `clip(date_arr - date_creat, 0, 730)` | Midnight calendar-normalized day delta clipped to `[0, 730]`. Eliminates Python `timedelta` sub-day negative floor-division integer bugs. |
| `lead_time_log` | Float | `ln(1 + lead_time_days)` | Variance-stabilizing natural logarithmic transformation dampening extreme right-skewed reservations. |
| `is_last_minute` | Binary | `1 if lead_time_days <= 3 else 0` | Flags urgent reservations with near-zero cancellation likelihood. |
| `is_short_notice`| Binary | `1 if 3 < lead_time_days <= 7 else 0` | Flags short-window bookings (4 to 7 days advance). |
| `is_far_advance` | Binary | `1 if lead_time_days >= 45 else 0` | Flags long-horizon reservations exhibiting highest cancellation vulnerability. |
| `lead_time_missing` | Binary | `1 if lead_time is NaT else 0` | Audit indicator identifying reservations lacking creation or arrival timestamps. |
| `lead_time_invalid` | Binary | `1 if date_arr < date_creat else 0` | Audit flag isolating retroactive or corrupted timestamps. |

---

### 2.3 Financial Proportions & Fee Relative Burdens
Absolute dollar fees (cleaning and service fees) vary widely by property size; relative financial proportions normalize fee burdens across budget and luxury listings.

| Feature Name | Type | Mathematical Formula / Derivation | Business Rationale & Guardrails |
| :--- | :--- | :--- | :--- |
| `cleaning_fee_ratio` | Float | `clip(CLEANING_FEE / TOTAL_AMOUNT, 0.0, 1.0)` | Proportion of total booking bill attributable to cleaning costs. Division by non-positive total masked to NaN → 0.0. |
| `service_fee_ratio` | Float | `clip(SERVICE_FEE / TOTAL_AMOUNT, 0.0, 1.0)` | Proportion of total booking bill attributable to Airbnb service fee. Bounded in `[0.0, 1.0]`. |
| `total_amount_invalid` | Binary | `1 if TOTAL_AMOUNT <= 0 or is NaN else 0` | Defensive audit indicator preventing mathematical inversion or divide-by-zero crashes. |

---

### 2.4 Supply Capacity & Relative Density
Captures property density, space per guest, and fee burdens normalized by physical capacity.

| Feature Name | Type | Mathematical Formula / Derivation | Business Rationale & Guardrails |
| :--- | :--- | :--- | :--- |
| `bedroom_to_accommodates_ratio` | Float | `clip(BEDROOMS / ACCOMMODATES, 0.0, 2.0)` | Measures privacy density (defaults to 0.5 if missing). Bounded to prevent distortion from outlier studio configs. |
| `cleaning_fee_per_bedroom` | Float | `CLEANING_FEE / max(1, BEDROOMS)` | Unit cleaning fee per bedroom. |
| `cleaning_fee_per_accommodate` | Float | `CLEANING_FEE / max(1, ACCOMMODATES)` | Unit cleaning fee per guest capacity. |
| `price_per_accommodate` | Float | `PRICE_PER_NIGHT / max(1, ACCOMMODATES)` | **Strictly isolated to Cancellation Classifier**. Banned from Price Regressor to eliminate direct target leakage. |
| `accommodates_invalid` | Binary | `1 if ACCOMMODATES <= 0 or is NaN else 0` | Data quality indicator flagging uninitialized capacity metadata. |

---

### 2.5 Host Reputation & Reliability
Encodes host responsiveness and verified Superhost badges.

| Feature Name | Type | Mathematical Formula / Derivation | Business Rationale & Guardrails |
| :--- | :--- | :--- | :--- |
| `is_superhost_binary` | Binary | `1 if IS_SUPERHOST in ['true', 't', '1', 'yes'] else 0` | Robust case-insensitive string parsing converting diverse boolean formats to 0 or 1. |
| `host_response_rate` | Float | `parse_pct(RESPONSE_RATE) ∈ [0.0, 100.0]` | Strips whitespace and trailing `%` symbols; imputes missing records to median baseline (80.0%). |
| `response_rate_missing`| Binary | `1 if RESPONSE_RATE is NaN else 0` | Audit indicator flagging hosts without public response rate telemetry. |

---

### 2.6 Zipline Point-in-Time Sliding Window Features
Implemented in [`ml/features/feature_store.py`](../ml/features/feature_store.py), these features simulate an enterprise point-in-time feature store (Zipline pattern), computing listing behavioral velocity prior to reservation timestamp:

```text
Window(t) = [t - 30 days, t)    (strictly prior to current observation timestamp)
```

| Feature Name | Type | Derivation | Leakage Prevention Rule |
| :--- | :--- | :--- | :--- |
| `trailing_30d_listing_bookings` | Integer | `count(bookings in Window)` | Sum of bookings on the listing strictly prior to current reservation creation. |
| `trailing_30d_listing_cancellations` | Integer | `count(cancellations in Window)` | Sum of cancellations on the listing strictly prior to current reservation creation. |
| `trailing_30d_cancellation_rate` | Float | `trailing_30d_cancellations / max(1, trailing_30d_bookings)` | Point-in-time listing cancellation propensity (defaults to 0.0 if zero bookings). |

---

## 3. Preprocessing & Model Feature Assignment

The preprocessor is structured as a `ColumnTransformer` executing distinct scikit-learn transformers across numeric and categorical features:

### 3.1 Numerical Pipeline
* **Imputer**: `SimpleImputer(strategy="median")`
* **Scaler**: `StandardScaler()`

### 3.2 Categorical Pipeline
* **Imputer**: `SimpleImputer(strategy="constant", fill_value="missing")`
* **Encoder**: `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`

### 3.3 Feature Allocations by Model

| Pipeline Category | Dynamic Price Regressor (`PriceRegressor`) | Cancellation Risk Classifier (`CancellationClassifier`) |
| :--- | :--- | :--- |
| **Numeric Features** | `ACCOMMODATES`, `BEDROOMS`, `BATHROOMS`, `bedroom_to_accommodates_ratio`, `cleaning_fee_per_bedroom`, `cleaning_fee_per_accommodate`, `arrival_month`, `arrival_dow`, `is_weekend_arrival`, `arrival_quarter`, `arrival_month_sin`, `arrival_month_cos`, `arrival_dow_sin`, `arrival_dow_cos`, `CLEANING_FEE`, `is_superhost_binary`, `host_response_rate` | `lead_time_days`, `lead_time_log`, `is_last_minute`, `is_short_notice`, `is_far_advance`, `arrival_month`, `arrival_dow`, `is_weekend_arrival`, `arrival_quarter`, `arrival_month_sin`, `arrival_month_cos`, `arrival_dow_sin`, `arrival_dow_cos`, `TOTAL_AMOUNT`, `CLEANING_FEE`, `SERVICE_FEE`, `cleaning_fee_ratio`, `service_fee_ratio`, `cleaning_fee_per_bedroom`, `cleaning_fee_per_accommodate`, `ACCOMMODATES`, `BEDROOMS`, `BATHROOMS`, `price_per_accommodate`, `host_response_rate`, `trailing_30d_listing_bookings`, `trailing_30d_listing_cancellations`, `trailing_30d_cancellation_rate` |
| **Categorical Features** | `PROPERTY_TYPE`, `ROOM_TYPE`, `CITY`, `RESPONSE_RATE_BAND` | `PROPERTY_TYPE`, `ROOM_TYPE`, `CITY`, `PRICE_PER_NIGHT_TAG`, `is_superhost_binary` |
| **Target Variable** | `PRICE_PER_NIGHT` (Continuous USD) | `IS_CANCELLED` (Binary {0, 1}) |

---

## 4. Verification & Unit Testing

The feature engineering subsystem is thoroughly verified via continuous automated unit tests in [`ml/tests/test_ml_pipeline.py`](../ml/tests/test_ml_pipeline.py):

* `test_feature_engineer_transformation`: Verifies shape, non-null guarantees, and trigonometric ranges `[-1.0, 1.0]`.
* `test_lead_time_edge_cases`: Verifies sub-day truncation protections, negative retroactive booking flags, and missing date coercion.
* `test_calendar_seasonality_features_edge_cases`: Validates cyclical boundary continuity across New Year transitions.
* `test_zipline_feature_store_no_lookahead`: Strictly asserts that future booking events do not contaminate past point-in-time calculation windows.
