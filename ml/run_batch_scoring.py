"""Scheduled batch scoring pipeline for Airbnb Gold Marts."""

import os
import sys

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import logging
from ml.data.datasets import load_gold_obt_dataset
from ml.inference.batch_predictor import BatchPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run_batch_scoring(output_path: str = "ml/artifacts/scored_bookings_batch.parquet"):
    logger.info("Initializing Batch Predictor...")
    c_path = "ml/artifacts/cancellation_model.joblib"
    p_path = "ml/artifacts/price_regressor.joblib"

    if not os.path.exists(c_path) or not os.path.exists(p_path):
        raise FileNotFoundError("Model artifacts missing. Run 'python ml/train_all.py' first.")

    predictor = BatchPredictor(c_path, p_path)

    logger.info("Extracting latest partition from AIRBNB.gold.obt...")
    df = load_gold_obt_dataset(limit=500, offline_mode=True)
    logger.info("Extract completed: %d rows to score.", len(df))

    scored_df = predictor.score_dataset(df)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    scored_df.to_parquet(output_path, index=False)
    logger.info("Batch scoring complete! Saved %d scored records to %s", len(scored_df), output_path)

    # Print summary of scoring
    high_risk_count = (scored_df["PRED_CANCEL_PROB"] >= 0.60).sum()
    avg_pred_price = scored_df["PRED_FAIR_PRICE"].mean()
    print("\n" + "=" * 60)
    print("BATCH SCORING PIPELINE SUMMARY")
    print(f"Total Bookings Scored:       {len(scored_df)}")
    print(f"High Cancellation Risk Rows: {high_risk_count} ({high_risk_count/len(scored_df)*100:.1f}%)")
    print(f"Average Predicted Fair Price: ${avg_pred_price:.2f}")
    print(f"Output File:                 {output_path}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch scoring pipeline")
    parser.add_argument("--output", default="ml/artifacts/scored_bookings_batch.parquet", help="Output parquet path")
    args = parser.parse_args()
    run_batch_scoring(args.output)
