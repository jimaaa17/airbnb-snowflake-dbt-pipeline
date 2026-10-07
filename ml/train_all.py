import os
import sys

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
import logging
from ml.data.datasets import load_gold_obt_dataset, temporal_train_test_split
from ml.models.cancellation_classifier import CancellationClassifier
from ml.models.price_regressor import PriceRegressor
from ml.tracking.tracker import MLflowTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def train_and_evaluate_all():
    artifacts_dir = "ml/artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Initialize MLflow Enterprise Tracker & Model Registry Manager
    tracker = MLflowTracker(experiment_name="airbnb_predictive_models")

    # 1. Load Configurations
    with open("ml/configs/cancellation_model_config.yaml", "r") as f:
        cancel_cfg = yaml.safe_load(f)
    with open("ml/configs/pricing_model_config.yaml", "r") as f:
        price_cfg = yaml.safe_load(f)

    # 2. Ingest Data & Temporal Split (Zero lookahead leakage)
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
    logger.info("SME Cancellation Impact: Revenue at Risk=$%.2f | Protected=$%.2f (%.1f%%) | Salvaged Revenue=$%.2f",
                test_metrics_c.get("sme_revenue_at_risk_usd", 0.0),
                test_metrics_c.get("sme_revenue_protected_usd", 0.0),
                test_metrics_c.get("sme_protection_capture_rate", 0.0) * 100,
                test_metrics_c.get("sme_estimated_salvaged_revenue_usd", 0.0))

    # Persist local artifact for fallback compatibility
    cancel_model_path = os.path.join(artifacts_dir, "cancellation_model.joblib")
    cancel_model.save(cancel_model_path)

    # Register in MLflow Model Registry and tag with '@champion' alias
    c_reg = tracker.log_and_register_model(
        run_name="cancellation_classifier_v1",
        model_name="cancellation_classifier",
        pipeline=cancel_model.pipeline,
        parameters=cancel_cfg,
        metrics=test_metrics_c,
        tags={
            "model_type": "gradient_boosting_classifier",
            "business_domain": "cancellation_prevention",
            "sme_goal": "protect_host_revenue"
        },
        alias="champion"
    )
    logger.info("Cancellation Classifier registered in MLflow: %s (v%s, @%s)",
                c_reg["model_name"], c_reg["version"], c_reg["alias"])

    # 4. Train & Evaluate Model 2: Price Regressor
    logger.info("Training Model 2: Dynamic Price Regressor...")
    price_model = PriceRegressor(price_cfg)
    train_metrics_p = price_model.train(train_df)
    test_metrics_p = price_model.evaluate(test_df)

    logger.info("Price Regressor Test Results: RMSE=$%.2f | MAE=$%.2f | R2=%.4f | MAPE=%.2f%%",
                test_metrics_p["rmse"], test_metrics_p["mae"], test_metrics_p["r2_score"], test_metrics_p["mape"] * 100)
    logger.info("SME Pricing Impact: Underpriced Listings=%.1f%% | Est. Monthly Uplift=$%.2f/listing | Guardrail Compliance=%.1f%%",
                test_metrics_p.get("sme_underpriced_listings_pct", 0.0) * 100,
                test_metrics_p.get("sme_estimated_monthly_uplift_per_listing_usd", 0.0),
                test_metrics_p.get("sme_within_guardrails_pct", 0.0) * 100)

    # Persist local artifact for fallback compatibility
    price_model_path = os.path.join(artifacts_dir, "price_regressor.joblib")
    price_model.save(price_model_path)

    # Register in MLflow Model Registry and tag with '@champion' alias
    p_reg = tracker.log_and_register_model(
        run_name="price_regressor_v1",
        model_name="price_regressor",
        pipeline=price_model.pipeline,
        parameters=price_cfg,
        metrics=test_metrics_p,
        tags={
            "model_type": "gradient_boosting_regressor",
            "business_domain": "pricing_optimization",
            "sme_goal": "maximize_host_occupancy_and_yield"
        },
        alias="champion"
    )
    logger.info("Price Regressor registered in MLflow: %s (v%s, @%s)",
                p_reg["model_name"], p_reg["version"], p_reg["alias"])

    # Executive Output for Technical & Business Stakeholders
    print("\n" + "=" * 75)
    print("🚀 MLFLOW MODEL REGISTRY & SME EVALUATION REPORT")
    print("=" * 75)
    print(f"MLflow Tracking URI: {tracker.tracking_uri}")
    print(f"Experiment Name:     {tracker.experiment_name}")
    print("-" * 75)
    print("1. CANCELLATION RISK CLASSIFIER")
    print(f"   Registry URI:     models:/cancellation_classifier@champion (v{c_reg['version']})")
    print(f"   Fallback Binary:  {cancel_model_path}")
    print(f"   [Technical ML]    ROC-AUC: {test_metrics_c['roc_auc']:.4f} | PR-AUC: {test_metrics_c['pr_auc']:.4f} | F1: {test_metrics_c['f1_score']:.4f} | Acc: {test_metrics_c['accuracy']:.4f}")
    print(f"   [SME / Business]  Evaluated Volume:     ${test_metrics_c.get('sme_total_booking_volume_usd', 0):,.2f} ({test_metrics_c.get('sme_total_bookings_evaluated', 0)} bookings)")
    print(f"                     Revenue at Risk:      ${test_metrics_c.get('sme_revenue_at_risk_usd', 0):,.2f}")
    print(f"                     Revenue Protected:    ${test_metrics_c.get('sme_revenue_protected_usd', 0):,.2f} ({test_metrics_c.get('sme_protection_capture_rate', 0)*100:.1f}% capture)")
    print(f"                     Est. Salvaged Yield:  ${test_metrics_c.get('sme_estimated_salvaged_revenue_usd', 0):,.2f} (via proactive host rebooking)")
    print(f"                     Avg. Warning Lead:    {test_metrics_c.get('sme_avg_lead_time_days_for_rebooking', 0)} days")
    print("-" * 75)
    print("2. DYNAMIC PRICE REGRESSOR")
    print(f"   Registry URI:     models:/price_regressor@champion (v{p_reg['version']})")
    print(f"   Fallback Binary:  {price_model_path}")
    print(f"   [Technical ML]    R²: {test_metrics_p['r2_score']:.4f} | MAPE: {test_metrics_p['mape']*100:.2f}% | RMSE: ${test_metrics_p['rmse']:.2f} | MAE: ${test_metrics_p['mae']:.2f}")
    print(f"   [SME / Business]  Underpriced Listings: {test_metrics_p.get('sme_underpriced_listings_pct', 0)*100:.1f}% (leaving money on table)")
    print(f"                     Overpriced Listings:  {test_metrics_p.get('sme_overpriced_listings_pct', 0)*100:.1f}% (vacancy risk)")
    print(f"                     Guardrail Compliance:{test_metrics_p.get('sme_within_guardrails_pct', 0)*100:.1f}% within ±10% fair market rate")
    print(f"                     Est. Monthly Uplift:  +${test_metrics_p.get('sme_estimated_monthly_uplift_per_listing_usd', 0):,.2f} / host listing")
    print("=" * 75)
    print("To inspect in MLflow UI run:")
    print("  uv run mlflow ui --backend-store-uri sqlite:///ml/mlruns.db --port 5001")
    print("=" * 75 + "\n")

if __name__ == "__main__":
    train_and_evaluate_all()
