# Model Experimentation, Lineage & Scientific Benchmarks

This document provides complete experimental lineage, technical benchmarks, and trade-off analyses for all machine learning models in the Airbnb Snowflake dbt Pipeline platform.

---

## 1. Experiment Lineage & Evaluation Runs

To ensure strict scientific reproducibility and auditable provenance, all models logged to the MLflow Model Registry are tracked against exact dataset versions, temporal splits, and Git commits:

| Experiment ID | Model Target | Dataset Version | Split Methodology | Sample Size (Train / Test) | Git Commit | Evaluation Date | Headline Test Metrics | Modeled Business Impact | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EXP-GOLD-001** *(Primary Baseline)* | **Dynamic Nightly Price Regressor** | v1.0 Gold OBT | Chronological Temporal (80/20) | 1,200 / 300 reservations | `49096f5` | Oct 7, 2026 | **R² = 0.9438**<br>MAPE = 10.64%<br>MAE = $20.58<br>RMSE = $25.92 | **19.3% underpriced**<br>Est. Monthly Uplift: **+$498.45/listing**<br>Guardrails: 60.7% compliant | **Champion (`@champion`)** |
| **EXP-GOLD-001** *(Primary Baseline)* | **Booking Cancellation Classifier** | v1.0 Gold OBT | Chronological Temporal (80/20) | 1,200 / 300 reservations | `49096f5` | Oct 7, 2026 | **ROC-AUC = 0.5427**<br>PR-AUC = 0.3750<br>F₁ = 0.2520<br>Accuracy = 68.33%<br>Recall = 18.39%<br>Precision = 40.00% | **$16,203.45 flagged** (18.7% capture)<br>**$5,671.21** modeled recoverable revenue (35% salvage)<br>23.6d lead warning | **Champion (`@champion`)** |
| **EXP-SNOW-002** *(Warehouse Extended)* | **Booking Cancellation Classifier** | v1.1 Warehouse Mart | Multi-Month Snowflake Backtest | 4,800 / 1,200 reservations | `3a18e02` | Sep 28, 2026 | **ROC-AUC = 0.8124**<br>PR-AUC = 0.7780<br>F₁ = 0.7412<br>Accuracy = 78.40%<br>Recall = 71.20% | **$68,400.00 flagged**<br>$23,940.00 modeled recoverable revenue | Experimental Branch |
| **EXP-CLASS-003** *(Cost-Sensitive)* | **Booking Cancellation Classifier** | v1.0 Gold OBT (Class-Weighted) | Temporal Split (τ = 0.25) | 1,200 / 300 reservations | `e719ab4` | Oct 2, 2026 | ROC-AUC = 0.5710<br>PR-AUC = 0.3820<br>F₁ = 0.3120<br>Accuracy = 61.20%<br>Recall = 34.48% | $28,910.00 flagged<br>$10,118.50 modeled recoverable revenue (Higher false alarms) | Exploratory Staging |

> **Audit Trail Note**: The reproducible **EXP-GOLD-001** run on the 300-reservation holdout is the canonical benchmark reported in the repository's primary documentation. Experimental runs **EXP-SNOW-002** and **EXP-CLASS-003** represent alternative cohorts with differing class balancing and warehouse volumes.

---

## 2. Comparative Evaluation Against Realistic Baselines

Candidate models are benchmarked directly against naive heuristics and simpler linear models on the exact same 300-reservation holdout set:

### A. Dynamic Nightly Price Regressor
| Model / Pipeline | Architecture & Configuration | R² Score | MAPE (%) | MAE (USD) | RMSE (USD) | Demonstrated Lift & Performance Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Naive Median Baseline** | Constant prediction of training median rate ($248.94) | -0.0006 | 53.08% | $90.27 | $109.32 | Zero variance explained; severe rate misallocations across market tiers. |
| **Linear OLS Baseline** | Univariate Ordinary Least Squares (`ACCOMMODATES` only) | 0.6871 | 25.14% | $49.60 | $61.13 | Explains capacity, but ignores city tier, room type, and seasonal waves. |
| **Production GBDT (Champion)** | Gradient Boosting Regressor (`max_depth=5`, `n_estimators=150`) with full feature suite | **0.9438** | **10.64%** | **$20.58** | **$25.92** | **+0.2567 R² lift** over linear baseline; cuts error by **58%** vs linear and **80%** vs median. |

### B. Booking Cancellation Risk Classifier
| Model / Pipeline | Architecture & Decision Threshold | Accuracy | Precision | Recall | F₁ Score | ROC-AUC | PR-AUC | Demonstrated Lift & Performance Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Zero-Rule Baseline** | Always predict confirmed (majority class, 71.0% prevalence) | 71.00% | 0.00% | 0.00% | 0.0000 | 0.5000 | 0.2900 | 0% recall: completely blind to all cancellation risks. |
| **Heuristic Cutoff** | Static rule: flag bookings with `lead_time >= 45` days | 71.00% | 0.00% | 0.00% | 0.0000 | 0.5000 | 0.2900 | Fails to isolate cancellations; zero discriminative power. |
| **Production GBDT (Champion)** | Stratified Gradient Boosting (`max_depth=6`, `subsample=0.85`, threshold τ = 0.35) | **68.33%** | **40.00%** | **18.39%** | **0.2520** | **0.5427** | **0.3750** | Unlocks positive recall where simple rules fail, catching high-risk cancellations 23.6 days in advance. |

---

## 3. Scientific Discussion: Model Quality vs. Business Utility

### The Discriminative Quality of the Cancellation Classifier
In a rigorous statistical evaluation, the cancellation classifier exhibits:
* **Accuracy**: 68.33%
* **Precision**: 40.00%
* **Recall**: 18.39%
* **ROC-AUC**: 0.5427
* **PR-AUC**: 0.3750 (vs. naive base rate of 0.2900)

#### Why is ROC-AUC 0.5427?
Predicting a booking cancellation *strictly at the moment of reservation creation* using only static listing and host metadata is inherently difficult. In real marketplaces, cancellations are frequently triggered by exogenous events occurring weeks after booking (guest illness, flight cancellations, personal schedule changes, weather disruptions) that are unobservable at $t_{\text{booking\_created}}$.

### Business Utility vs. Zero-Rule Baselines
Despite modest overall discrimination, the model delivers actionable operational value:
1. **Zero-Rule & Heuristic Failure**: Naive baselines achieve 71% accuracy by predicting "never cancels," but deliver **0% recall**, detecting \$0 of revenue at risk.
2. **Dollar Capture**: The model flags 16 actual cancellation reservations in the holdout cohort, representing **\$16,203.45 in at-risk booking value** (18.7% capture rate).
3. **Actionable Horizon**: The warnings arrive an average of **23.6 days prior to check-in**, providing hosts with ample calendar lead time to re-list and secure replacement guests.

### Intervention Cost Economics
Operationalizing predictive models requires weighing false alarms against missed interventions:
* **Host Outreach Cost**: An automated alert or proactive rescheduling credit costs approximately **$2–$5** in platform overhead.
* **Vacancy Cost**: An unrecovered cancellation results in a complete room vacancy averaging **$350–$1,000+** in lost gross revenue.
* **Asymmetric Payoff**: Because the cost of an unrecovered vacancy is 70× to 200× higher than the cost of a false-positive outreach, a threshold calibrated for targeted positive recall ($\tau = 0.35$) delivers net positive business ROI even with 40% precision.
* **Modeled Recoverable Revenue**: Under a conservative 35% rebooking salvage rate for flagged reservations with >14 days lead time, the modeled preserved revenue is **$5,671.21** on the 300-reservation holdout cohort ($16,203.45 × 0.35$).
