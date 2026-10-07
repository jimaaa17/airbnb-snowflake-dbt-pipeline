"""CLI Executable: Run SHAP Analysis on Dynamic Pricing Model.

Diagnoses global feature importance and reveals why underpriced gaps occur.
"""

import os
import sys
import json
import logging
import pandas as pd

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.data.datasets import load_gold_obt_dataset, temporal_train_test_split
from ml.evaluation.shap_diagnostics import PricingShapDiagnostics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    print("\n" + "=" * 80)
    print("🔍 SHAP PRICING INTERPRETABILITY & UNDERPRICED GAP DIAGNOSTIC")
    print("=" * 80)

    # 1. Load evaluation dataset
    logger.info("Loading Gold OBT holdout evaluation dataset...")
    df = load_gold_obt_dataset(limit=1500, offline_mode=True)
    _, test_df = temporal_train_test_split(df, time_col="BOOKING_DATE", train_ratio=0.8)
    logger.info("Test split size: %d records", len(test_df))

    # 2. Run SHAP Diagnostic Engine
    diagnostics = PricingShapDiagnostics()
    results = diagnostics.run_diagnostics(test_df, output_dir="ml/artifacts/shap")

    # 3. Print SME Diagnostic Summary
    print("\n" + "-" * 80)
    print("1. SUMMARY OF UNDERPRICED LISTING METRICS")
    print("-" * 80)
    print(f"• Total Evaluated Listings:       {results['total_test_samples']}")
    print(f"• Underpriced Listings Count:     {results['underpriced_count']} ({results['underpriced_pct']*100:.1f}%)")
    print(f"• Average Underpriced Gap:        ${results['avg_underpriced_gap_usd']:.2f} / night")
    print(f"• Model Baseline Expected Value:  ${results['expected_value_base_price']:.2f} / night")
    print(f"• Estimated Monthly Host Uplift:  ${results['avg_underpriced_gap_usd'] * 15.0:,.2f} per listing")

    print("\n" + "-" * 80)
    print("2. GLOBAL FEATURE IMPORTANCE (Mean |SHAP| across all listings)")
    print("-" * 80)
    df_global = pd.DataFrame(
        list(results["global_top_drivers"].items()),
        columns=["Feature", "Mean |SHAP Impact| ($/night)"]
    )
    print(df_global.to_markdown(index=False))

    print("\n" + "-" * 80)
    print("3. ROOT-CAUSE: WHY WERE UNDERPRICED LISTINGS VALUED HIGHER BY MODEL?")
    print("-" * 80)
    df_under = pd.DataFrame(
        list(results["underpriced_positive_drivers"].items()),
        columns=["Feature", "Mean Positive Push ($/night)"]
    )
    print(df_under.to_markdown(index=False))

    print("\n" + "-" * 80)
    print("4. DEEP-DIVE CASE STUDIES: TOP UNDERPRICED LISTINGS")
    print("-" * 80)
    for i, cs in enumerate(results["case_studies"], 1):
        print(f"\n[Case Study #{i}] Listing ID: {cs['listing_id']}")
        print(f"  • City: {cs['city']} | Room Type: {cs['room_type']} | Accommodates: {cs['accommodates']}")
        print(f"  • Actual Listed Price:    ${cs['actual_price']:.2f}")
        print(f"  • Model Fair Market Price: ${cs['predicted_price']:.2f}")
        print(f"  • Money Left on Table:    +${cs['underpriced_gap_usd']:.2f} / night")
        print("  • Key Upward Price Drivers (Why it should cost more):")
        for feat, val in list(cs["top_positive_shap_drivers"].items())[:3]:
            print(f"      + ${val:.2f}  <- {feat}")
        print("  • Key Downward Dampeners:")
        for feat, val in list(cs["top_negative_shap_drivers"].items())[:2]:
            print(f"      - ${abs(val):.2f}  <- {feat}")

    print("\n" + "-" * 80)
    print("5. PERSISTED ARTIFACTS & PLOTS")
    print("-" * 80)
    for plot_name, path in results["plots"].items():
        print(f"  • {plot_name}: {path}")

    metrics_out = "ml/artifacts/shap/shap_diagnostic_summary.json"
    with open(metrics_out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  • Summary JSON: {metrics_out}")

    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
