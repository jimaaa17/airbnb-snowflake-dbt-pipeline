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

# Minimum production quality thresholds (SLA)
MIN_PRICE_R2 = 0.85
MAX_PRICE_MAPE = 0.20  # 20% max error
MIN_CLASSIFIER_ACCURACY = 0.60

def run_evaluation_gate():
    logger.info("Running automated MLOps evaluation quality gate on holdout test set...")

    df = load_gold_obt_dataset(limit=1000, offline_mode=True)
    _, test_df = temporal_train_test_split(df, time_col="BOOKING_DATE", train_ratio=0.8)

    # 1. Evaluate Price Regressor
    price_model = PriceRegressor({}).load("ml/artifacts/price_regressor.joblib")
    price_metrics = price_model.evaluate(test_df)
    logger.info("Price Regressor Quality: R2=%.4f (Min SLA=%.2f) | MAPE=%.2f%% (Max SLA=%.2f%%)",
                price_metrics["r2_score"], MIN_PRICE_R2, price_metrics["mape"] * 100, MAX_PRICE_MAPE * 100)

    if price_metrics["r2_score"] < MIN_PRICE_R2:
        logger.error("GATE REJECTED: Price regressor R2 (%.4f) below SLA threshold (%.2f)", price_metrics["r2_score"], MIN_PRICE_R2)
        sys.exit(1)

    if price_metrics["mape"] > MAX_PRICE_MAPE:
        logger.error("GATE REJECTED: Price regressor MAPE (%.4f) exceeds max error threshold (%.2f)", price_metrics["mape"], MAX_PRICE_MAPE)
        sys.exit(1)

    # 2. Evaluate Cancellation Classifier
    cancel_model = CancellationClassifier({}).load("ml/artifacts/cancellation_model.joblib")
    cancel_metrics = cancel_model.evaluate(test_df)
    logger.info("Cancellation Classifier Quality: Accuracy=%.4f (Min SLA=%.2f)",
                cancel_metrics["accuracy"], MIN_CLASSIFIER_ACCURACY)

    if cancel_metrics["accuracy"] < MIN_CLASSIFIER_ACCURACY:
        logger.error("GATE REJECTED: Cancellation accuracy (%.4f) below SLA threshold (%.2f)", cancel_metrics["accuracy"], MIN_CLASSIFIER_ACCURACY)
        sys.exit(1)

    logger.info("ALL MODEL QUALITY GATES PASSED! Safe for production deployment.")
    print("ALL MODEL EVALUATION GATES PASSED (Exit 0)")

if __name__ == "__main__":
    run_evaluation_gate()
