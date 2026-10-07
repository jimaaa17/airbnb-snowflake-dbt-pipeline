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
1. **Strict Train-Serve Parity**: All feature transformations are encapsulated in custom, serializable scikit-learn transformers (`AirbnbFeatureEngineer` in [`ml/features/transformers.py`](file:///Users/jimitnaik/Documents/Projects/Airbnb%20Snowflake%20DBT%20Pipeline/ml/features/transformers.py)) embedded directly into the persisted model pipeline. Both offline model training and real-time FastAPI endpoints (`/predict/price`, `/predict/cancellation`) execute identical code paths.
2. **Zero Lookahead Leakage**: Temporal features and sliding-window aggregations strictly exclude observations occurring at or after the prediction timestamp ($t < \text{curr\_time}$).
3. **Defensive Typing & Zero Div/0 Crashes**: Financial ratios and capacity metrics implement strict mathematical guards (masking non-positive totals, clipping ratios to $[0.0, 1.0]$, and tracking invalid data indicators).

---

## 2. Feature Catalog & Mathematical Formulations

### 2.1 Calendar Seasonality & Cyclical Waves
Raw calendar numbers (such as month 12 and month 1) suffer from artificial numeric boundary cliffs in tree and linear models. Cyclical trigonometric transformations project these features onto a continuous unit circle.

| Feature Name | Type | Mathematical Formula / Derivation | Business Rationale & Guardrails |
| :--- | :--- | :--- | :--- |
| `arrival_month` | Float | $\text{month} \in [1, 12]$ (defaults to 6.0 if missing) | Base calendar month of check-in date. |
| `arrival_month_sin` | Float | $\sin\left(\frac{2\pi \cdot (\text{month} - 1)}{12}\right)$ | Cyclical annual wave resolving the December ($m=12$) to January ($m=1$) circular continuity. |
| `arrival_month_cos` | Float | $\cos\left(\frac{2\pi \cdot (\text{month} - 1)}{12}\right)$ | Orthogonal cyclical component completing the 2D annual seasonal circle. |
| `arrival_dow` | Float | $\text{dow} \in [0, 6]$ (0 = Monday, 6 = Sunday) | Day of week index of check-in. |
| `arrival_dow_sin` | Float | $\sin\left(\frac{2\pi \cdot \text{dow}}{7}\right)$ | Weekly cyclical cadence resolving Sunday ($6$) to Monday ($0$) transition. |
| `arrival_dow_cos` | Float | $\cos\left(\frac{2\pi \cdot \text{dow}}{7}\right)$ | Orthogonal weekly cyclical component. |
| `is_weekend_arrival`| Binary | $\mathbb{I}(\text{dow} \in \{4, 5\})$ | Flags Friday and Saturday check-ins isolating high-demand leisure vacation travel. |
| `arrival_quarter` | Float | $\lceil \text{month} / 3 \rceil \in [1, 4]$ | Macro-quarter indicator for quarterly seasonal demand. |
| `arrival_date_missing` | Binary | $\mathbb{I}(\text{arrival\_date is NaT})$ | Audit indicator tracking missing or corrupt reservation arrival dates. |

---

### 2.2 Lead-Time Dynamics & Behavioral Bucketing
Lead time (the gap between reservation creation and check-in) is the single strongest behavioral predictor of cancellation likelihood and pricing elasticity.

| Feature Name | Type | Mathematical Formula / Derivation | Business Rationale & Guardrails |
| :--- | :--- | :--- | :--- |
| `lead_time_days` | Float | $\text{clip}\left(\lfloor \text{date}_{\text{arr}} - \text{date}_{\text{creat}} \rfloor, 0, 730\right)$ | Midnight calendar-normalized day delta clipped to $[0, 730]$. Eliminates Python `timedelta` sub-day negative floor-division integer bugs. |
| `lead_time_log` | Float | $\ln(1 + \text{lead\_time\_days})$ | Variance-stabilizing natural logarithmic transformation dampening extreme right-skewed reservations. |
| `is_last_minute` | Binary | $\mathbb{I}(\text{lead\_time\_days} \le 3)$ | Flags urgent reservations with near-zero cancellation likelihood. |
| `is_short_notice`| Binary | $\mathbb{I}(3 < \text{lead\_time\_days} \le 7)$ | Flags short-window bookings (4 to 7 days advance). |
| `is_far_advance` | Binary | $\mathbb{I}(\text{lead\_time\_days} \ge 45)$ | Flags long-horizon reservations exhibiting highest cancellation vulnerability. |
| `lead_time_missing` | Binary | $\mathbb{I}(\text{lead\_time is NaT})$ | Audit indicator identifying reservations lacking creation or arrival timestamps. |
| `lead_time_invalid` | Binary | $\mathbb{I}(\text{date}_{\text{arr}} < \text{date}_{\text{creat}})$ | Audit flag isolating retroactive or corrupted timestamps. |

---

### 2.3 Financial Proportions & Fee Relative Burdens
Absolute dollar fees (cleaning and service fees) vary widely by property size; relative financial proportions normalize fee burdens across budget and luxury listings.

| Feature Name | Type | Mathematical Formula / Derivation | Business Rationale & Guardrails |
| :--- | :--- | :--- | :--- |
| `cleaning_fee_ratio` | Float | $\text{clip}\left(\frac{\text{CLEANING\_FEE}}{\text{TOTAL\_AMOUNT}}, 0.0, 1.0\right)$ | Proportion of total booking bill attributable to cleaning costs. Division by non-positive total masked to NaN $\rightarrow$ 0.0. |
| `service_fee_ratio` | Float | $\text{clip}\left(\frac{\text{SERVICE\_FEE}}{\text{TOTAL\_AMOUNT}}, 0.0, 1.0\right)$ | Proportion of total booking bill attributable to Airbnb service fee. Bounded in $[0.0, 1.0]$. |
| `total_amount_invalid` | Binary | $\mathbb{I}(\text{TOTAL\_AMOUNT} \le 0 \lor \text{is NaN})$ | Defensive audit indicator preventing mathematical inversion or divide-by-zero crashes. |

---

### 2.4 Supply Capacity & Relative Density
Captures property density, space per guest, and fee burdens normalized by physical capacity.

| Feature Name | Type | Mathematical Formula / Derivation | Business Rationale & Guardrails |
| :--- | :--- | :--- | :--- |
| `bedroom_to_accommodates_ratio` | Float | $\text{clip}\left(\frac{\text{BEDROOMS}}{\text{ACCOMMODATES}}, 0.0, 2.0\right)$ | Measures privacy density (defaults to 0.5 if missing). Bounded to prevent distortion from outlier studio configs. |
| `cleaning_fee_per_bedroom` | Float | $\frac{\text{CLEANING\_FEE}}{\max(1, \text{BEDROOMS})}$ | Unit cleaning fee per bedroom. |
| `cleaning_fee_per_accommodate` | Float | $\frac{\text{CLEANING\_FEE}}{\max(1, \text{ACCOMMODATES})}$ | Unit cleaning fee per guest capacity. |
| `price_per_accommodate` | Float | $\frac{\text{PRICE\_PER\_NIGHT}}{\max(1, \text{ACCOMMODATES})}$ | **Strictly isolated to Cancellation Classifier**. Banned from Price Regressor to eliminate direct target leakage. |
| `accommodates_invalid` | Binary | $\mathbb{I}(\text{ACCOMMODATES} \le 0 \lor \text{is NaN})$ | Data quality indicator flagging uninitialized capacity metadata. |

---

### 2.5 Host Reputation & Reliability
Encodes host responsiveness and verified Superhost badges.

| Feature Name | Type | Mathematical Formula / Derivation | Business Rationale & Guardrails |
| :--- | :--- | :--- | :--- |
| `is_superhost_binary` | Binary | $\mathbb{I}(\text{IS\_SUPERHOST} \in \{\text{'true'}, \text{'t'}, \text{'1'}, \text{'yes'}\})$ | Robust case-insensitive string parsing converting diverse boolean formats to 0 or 1. |
| `host_response_rate` | Float | $\text{parse\_pct}(\text{RESPONSE\_RATE}) \in [0.0, 100.0]$ | Strips whitespace and trailing `%` symbols; imputes missing records to median baseline ($80.0\%$). |
| `response_rate_missing`| Binary | $\mathbb{I}(\text{RESPONSE\_RATE is NaN})$ | Audit indicator flagging hosts without public response rate telemetry. |

---

### 2.6 Zipline Point-in-Time Sliding Window Features
Implemented in [`ml/features/feature_store.py`](file:///Users/jimitnaik/Documents/Projects/Airbnb%20Snowflake%20DBT%20Pipeline/ml/features/feature_store.py), these features simulate an enterprise point-in-time feature store (Zipline pattern), computing listing behavioral velocity prior to reservation timestamp:

$$\text{Window}(t) = \left\{\tau \mid t - 30\text{ days} \le \tau < t\right\}$$

| Feature Name | Type | Derivation | Leakage Prevention Rule |
| :--- | :--- | :--- | :--- |
| `trailing_30d_listing_bookings` | Integer | $\sum_{\tau \in \text{Window}} 1$ | Sum of bookings on the listing strictly prior to current reservation creation. |
| `trailing_30d_listing_cancellations` | Integer | $\sum_{\tau \in \text{Window}} \mathbb{I}(\text{status} = \text{'cancelled'})$ | Sum of cancellations on the listing strictly prior to current reservation creation. |
| `trailing_30d_cancellation_rate` | Float | $\frac{\text{trailing\_30d\_cancellations}}{\max(1, \text{trailing\_30d\_bookings})}$ | Point-in-time listing cancellation propensity (defaults to 0.0 if zero bookings). |

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

The feature engineering subsystem is thoroughly verified via continuous automated unit tests in [`ml/tests/test_ml_pipeline.py`](file:///Users/jimitnaik/Documents/Projects/Airbnb%20Snowflake%20DBT%20Pipeline/ml/tests/test_ml_pipeline.py):

* `test_feature_engineer_transformation`: Verifies shape, non-null guarantees, and trigonometric ranges $[-1.0, 1.0]$.
* `test_lead_time_edge_cases`: Verifies sub-day truncation protections, negative retroactive booking flags, and missing date coercion.
* `test_calendar_seasonality_features_edge_cases`: Validates cyclical boundary continuity across New Year transitions.
* `test_zipline_feature_store_no_lookahead`: Strictly asserts that future booking events do not contaminate past point-in-time calculation windows.
