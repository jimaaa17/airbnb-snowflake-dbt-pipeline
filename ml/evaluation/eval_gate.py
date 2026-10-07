"""Model Quality Gatekeeper for CI/CD pipelines.

Asserts that newly trained models meet minimal enterprise accuracy and calibration thresholds
before allowing deployment to production. Exits with code 1 if thresholds are breached.
"""

import os
import sys

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import yaml
import logging
from ml.data.datasets import load_gold_obt_dataset, temporal_train_test_split
from ml.models.cancellation_classifier import CancellationClassifier
from ml.models.price_regressor import PriceRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from ml.tracking.tracker import MLflowTracker, get_default_tracking_uri

# Minimum production quality thresholds (Technical & SME SLA)
MIN_PRICE_R2 = 0.85
MAX_PRICE_MAPE = 0.20  # 20% max error
MIN_CLASSIFIER_ACCURACY = 0.60
MIN_CLASSIFIER_PR_AUC = 0.25  # Must demonstrate positive precision-recall lift over naive prior
MIN_CLASSIFIER_RECALL = 0.10  # Must catch >=10% of cancellations to justify intervention workflows
MIN_PRICING_GUARDRAIL_PCT = 0.50  # At least 50% within fair guardrails

def run_evaluation_gate():
    logger.info("Running automated MLOps & SME evaluation quality gate on holdout test set...")

    df = load_gold_obt_dataset(limit=1000, offline_mode=True)
    _, test_df = temporal_train_test_split(df, time_col="BOOKING_DATE", train_ratio=0.8)

    # 1. Evaluate Price Regressor from MLflow Registry / fallback
    try:
        import mlflow
        mlflow.set_tracking_uri(get_default_tracking_uri())
        p_pipe = mlflow.sklearn.load_model("models:/price_regressor@champion")
        price_model = PriceRegressor({})
        price_model.pipeline = p_pipe
        price_model.is_fitted = True
        logger.info("Evaluating Price Regressor from MLflow Registry: models:/price_regressor@champion")
    except Exception:
        price_model = PriceRegressor({}).load("ml/artifacts/price_regressor.joblib")
        logger.info("Evaluating Price Regressor from local binary fallback: ml/artifacts/price_regressor.joblib")

    price_metrics = price_model.evaluate(test_df)
    logger.info("Price Regressor Quality: R2=%.4f (Min SLA=%.2f) | MAPE=%.2f%% (Max SLA=%.2f%%)",
                price_metrics["r2_score"], MIN_PRICE_R2, price_metrics["mape"] * 100, MAX_PRICE_MAPE * 100)
    logger.info("SME Pricing Impact: Guardrail Compliance=%.1f%% | Underpriced=%.1f%% | Est. Uplift=$%.2f/listing",
                price_metrics.get("sme_within_guardrails_pct", 0.0) * 100,
                price_metrics.get("sme_underpriced_listings_pct", 0.0) * 100,
                price_metrics.get("sme_estimated_monthly_uplift_per_listing_usd", 0.0))

    if price_metrics["r2_score"] < MIN_PRICE_R2:
        logger.error("GATE REJECTED: Price regressor R2 (%.4f) below SLA threshold (%.2f)", price_metrics["r2_score"], MIN_PRICE_R2)
        sys.exit(1)

    if price_metrics["mape"] > MAX_PRICE_MAPE:
        logger.error("GATE REJECTED: Price regressor MAPE (%.4f) exceeds max error threshold (%.2f)", price_metrics["mape"], MAX_PRICE_MAPE)
        sys.exit(1)

    guardrail_pct = price_metrics.get("sme_within_guardrails_pct", 0.0)
    if guardrail_pct < MIN_PRICING_GUARDRAIL_PCT:
        logger.error("GATE REJECTED: Price regressor guardrail compliance (%.2f%%) below SLA threshold (%.2f%%)",
                     guardrail_pct * 100, MIN_PRICING_GUARDRAIL_PCT * 100)
        sys.exit(1)

    # 2. Evaluate Cancellation Classifier from MLflow Registry / fallback
    try:
        import mlflow
        mlflow.set_tracking_uri(get_default_tracking_uri())
        c_pipe = mlflow.sklearn.load_model("models:/cancellation_classifier@champion")
        cancel_model = CancellationClassifier({})
        cancel_model.pipeline = c_pipe
        cancel_model.is_fitted = True
        logger.info("Evaluating Cancellation Classifier from MLflow Registry: models:/cancellation_classifier@champion")
    except Exception:
        cancel_model = CancellationClassifier({}).load("ml/artifacts/cancellation_model.joblib")
        logger.info("Evaluating Cancellation Classifier from local binary fallback: ml/artifacts/cancellation_model.joblib")

    cancel_metrics = cancel_model.evaluate(test_df)
    logger.info("Cancellation Classifier Quality: Accuracy=%.4f (Min SLA=%.2f) | PR-AUC=%.4f (Min SLA=%.2f) | Recall=%.2f%% (Min SLA=%.2f%%) | Brier=%.4f",
                cancel_metrics["accuracy"], MIN_CLASSIFIER_ACCURACY,
                cancel_metrics["pr_auc"], MIN_CLASSIFIER_PR_AUC,
                cancel_metrics["recall"] * 100, MIN_CLASSIFIER_RECALL * 100,
                cancel_metrics.get("brier_score", 0.0))
    logger.info("SME Cancellation Impact: Revenue Protected=$%.2f (%.1f%% capture) | Salvaged Yield=$%.2f",
                cancel_metrics.get("sme_revenue_protected_usd", 0.0),
                cancel_metrics.get("sme_protection_capture_rate", 0.0) * 100,
                cancel_metrics.get("sme_estimated_salvaged_revenue_usd", 0.0))

    if cancel_metrics["accuracy"] < MIN_CLASSIFIER_ACCURACY:
        logger.error("GATE REJECTED: Cancellation accuracy (%.4f) below SLA threshold (%.2f)", cancel_metrics["accuracy"], MIN_CLASSIFIER_ACCURACY)
        sys.exit(1)

    if cancel_metrics["pr_auc"] < MIN_CLASSIFIER_PR_AUC:
        logger.error("GATE REJECTED: Cancellation PR-AUC (%.4f) below SLA threshold (%.2f)", cancel_metrics["pr_auc"], MIN_CLASSIFIER_PR_AUC)
        sys.exit(1)

    recall_pct = cancel_metrics.get("recall", 0.0)
    if recall_pct < MIN_CLASSIFIER_RECALL:
        logger.error("GATE REJECTED: Cancellation recall (%.4f) below SLA threshold (%.2f)", recall_pct, MIN_CLASSIFIER_RECALL)
        sys.exit(1)

    logger.info("ALL MODEL QUALITY & SME GATES PASSED! Validated for production deployment.")
    print("ALL MODEL EVALUATION GATES PASSED (Exit 0)")

if __name__ == "__main__":
    run_evaluation_gate()
