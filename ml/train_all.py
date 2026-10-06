import os
import sys

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
import logging
from ml.data.datasets import load_gold_obt_dataset, temporal_train_test_split
from ml.models.cancellation_classifier import CancellationClassifier
from ml.models.price_regressor import PriceRegressor
from ml.tracking.tracker import ExperimentTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def train_and_evaluate_all():
    artifacts_dir = "ml/artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)
    tracker = ExperimentTracker("airbnb_predictive_models", base_dir="ml/experiments")

    # 1. Load Configurations
    with open("ml/configs/cancellation_model_config.yaml", "r") as f:
        cancel_cfg = yaml.safe_load(f)
    with open("ml/configs/pricing_model_config.yaml", "r") as f:
        price_cfg = yaml.safe_load(f)

    # 2. Ingest Data & Temporal Split
    logger.info("Ingesting Gold OBT dataset...")
    df = load_gold_obt_dataset(limit=1500)
    logger.info("Total rows loaded: %d", len(df))

    train_df, test_df = temporal_train_test_split(df, time_col="BOOKING_DATE", train_ratio=0.8)
    logger.info("Temporal Split complete: Train rows=%d, Test rows=%d", len(train_df), len(test_df))

    # 3. Train & Evaluate Model 1: Cancellation Classifier
    logger.info("Training Model 1: Cancellation Risk Classifier...")
    cancel_model = CancellationClassifier(cancel_cfg)
    train_metrics_c = cancel_model.train(train_df)
    test_metrics_c = cancel_model.evaluate(test_df)

    logger.info("Cancellation Model Test Results: ROC-AUC=%.4f | PR-AUC=%.4f | F1=%.4f | Accuracy=%.4f",
                test_metrics_c["roc_auc"], test_metrics_c["pr_auc"], test_metrics_c["f1_score"], test_metrics_c["accuracy"])

    cancel_model_path = os.path.join(artifacts_dir, "cancellation_model.joblib")
    cancel_model.save(cancel_model_path)
    tracker.log_run("cancellation_classifier_v1", cancel_cfg, test_metrics_c, cancel_model_path)
    logger.info("Cancellation Model saved to %s", cancel_model_path)

    # 4. Train & Evaluate Model 2: Price Regressor
    logger.info("Training Model 2: Dynamic Price Regressor...")
    price_model = PriceRegressor(price_cfg)
    train_metrics_p = price_model.train(train_df)
    test_metrics_p = price_model.evaluate(test_df)

    logger.info("Price Regressor Test Results: RMSE=$%.2f | MAE=$%.2f | R2=%.4f | MAPE=%.2f%%",
                test_metrics_p["rmse"], test_metrics_p["mae"], test_metrics_p["r2_score"], test_metrics_p["mape"] * 100)

    price_model_path = os.path.join(artifacts_dir, "price_regressor.joblib")
    price_model.save(price_model_path)
    tracker.log_run("price_regressor_v1", price_cfg, test_metrics_p, price_model_path)
    logger.info("Price Regressor Model saved to %s", price_model_path)

    print("\n" + "=" * 60)
    print("ALL MODELS TRAINED & PERSISTED SUCCESSFULLY!")
    print(f"Cancellation Model: {cancel_model_path}")
    print(f"Price Regressor:    {price_model_path}")
    print("=" * 60)

if __name__ == "__main__":
    train_and_evaluate_all()
