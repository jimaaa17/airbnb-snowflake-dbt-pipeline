"""SHAP (SHapley Additive exPlanations) Diagnostic Engine for Dynamic Pricing.

Explains global model drivers and diagnoses why underpriced gaps occur
(identifying listings leaving money on the table and explaining the feature attribution).
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, Tuple

# Headless matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import joblib

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ml.data.datasets import load_gold_obt_dataset, temporal_train_test_split
from ml.tracking.tracker import MLflowTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class PricingShapDiagnostics:
    """Computes tree-based SHAP explanations for PriceRegressor models."""

    def __init__(self, model_path: Optional[str] = None):
        self.tracker = MLflowTracker(experiment_name="airbnb_predictive_models")
        self.pipeline, self.config = self._load_model_and_config(model_path)
        self.feature_engineer = self.pipeline.named_steps["feature_engineer"]
        self.preprocessor = self.pipeline.named_steps["preprocessor"]
        self.regressor = self.pipeline.named_steps["regressor"]
        self.explainer = shap.TreeExplainer(self.regressor)

    def _load_model_and_config(self, model_path: Optional[str]) -> Tuple[Any, Dict[str, Any]]:
        """Loads model pipeline and config from MLflow registry or local fallback."""
        if model_path and os.path.exists(model_path):
            logger.info("Loading model from specified path: %s", model_path)
            artifact = joblib.load(model_path)
            return artifact["pipeline"], artifact.get("config", {})

        try:
            logger.info("Attempting to load @champion price_regressor from MLflow...")
            pipeline = self.tracker.load_registered_model("price_regressor", alias_or_version="champion")
            # Config fallback
            fallback_path = "ml/artifacts/price_regressor.joblib"
            config = {}
            if os.path.exists(fallback_path):
                config = joblib.load(fallback_path).get("config", {})
            return pipeline, config
        except Exception as e:
            logger.warning("MLflow model registry load failed (%s). Falling back to local joblib...", e)
            fallback_path = "ml/artifacts/price_regressor.joblib"
            artifact = joblib.load(fallback_path)
            return artifact["pipeline"], artifact.get("config", {})

    def prepare_data(self, df_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        """Transforms raw test dataframe through feature engineering and column transformer."""
        y_test = pd.to_numeric(df_test["PRICE_PER_NIGHT"]).to_numpy()
        y_pred = self.pipeline.predict(df_test)

        df_fe = self.feature_engineer.transform(df_test)
        X_proc = self.preprocessor.transform(df_fe)
        feature_names = list(self.preprocessor.get_feature_names_out())

        return X_proc, y_test, y_pred, feature_names

    def run_diagnostics(
        self,
        df_test: pd.DataFrame,
        output_dir: str = "ml/artifacts/shap"
    ) -> Dict[str, Any]:
        """Executes full global and underpriced segment SHAP diagnostics."""
        os.makedirs(output_dir, exist_ok=True)
        X_proc, y_test, y_pred, feature_names = self.prepare_data(df_test)

        logger.info("Computing Tree SHAP values across %d test samples...", len(df_test))
        shap_explanation = self.explainer(X_proc)
        shap_values = shap_explanation.values
        ev = self.explainer.expected_value
        expected_value = float(np.ravel(ev)[0]) if hasattr(ev, "__iter__") else float(ev)

        # 1. Underpriced Segmentation
        underpriced_mask = y_test < (y_pred * 0.90)
        overpriced_mask = y_test > (y_pred * 1.10)
        guardrail_mask = (~underpriced_mask) & (~overpriced_mask)

        underpriced_indices = np.where(underpriced_mask)[0]
        num_underpriced = int(np.sum(underpriced_mask))
        underpriced_pct = float(num_underpriced / max(1, len(df_test)))

        gaps = y_pred[underpriced_mask] - y_test[underpriced_mask]
        avg_underpriced_gap = float(np.mean(gaps)) if num_underpriced > 0 else 0.0

        # 2. Global Feature Importance (Mean Absolute SHAP)
        mean_abs_global = np.abs(shap_values).mean(axis=0)
        global_importance = pd.Series(mean_abs_global, index=feature_names).sort_values(ascending=False)

        # 3. Underpriced Segment Attribution (Why did the model assign a higher price?)
        shap_under = shap_values[underpriced_mask]
        mean_shap_under = pd.Series(shap_under.mean(axis=0), index=feature_names).sort_values(ascending=False)
        mean_abs_under = pd.Series(np.abs(shap_under).mean(axis=0), index=feature_names).sort_values(ascending=False)

        # 4. Generate & Save Visualizations
        # (A) Global Summary Bar Plot
        fig_global, ax_global = plt.subplots(figsize=(10, 6))
        global_importance.head(10).sort_values().plot(
            kind="barh",
            color="#FF5A5F",
            ax=ax_global
        )
        ax_global.set_title("Global Feature Importance (Mean |SHAP Value|)", fontsize=13, fontweight="bold")
        ax_global.set_xlabel("Mean |SHAP Impact| ($/night)")
        plt.tight_layout()
        global_plot_path = os.path.join(output_dir, "shap_global_importance.png")
        fig_global.savefig(global_plot_path, dpi=200)
        plt.close(fig_global)

        # (B) Underpriced Drivers Plot (Top Features pushing price UP for underpriced cohort)
        fig_under, ax_under = plt.subplots(figsize=(10, 6))
        mean_shap_under.head(10).sort_values().plot(
            kind="barh",
            color="#00A699",
            ax=ax_under
        )
        ax_under.set_title("Top SHAP Contributors Pushing Price UP on Underpriced Listings", fontsize=13, fontweight="bold")
        ax_under.set_xlabel("Average Positive SHAP Contribution ($/night)")
        plt.tight_layout()
        under_plot_path = os.path.join(output_dir, "shap_underpriced_drivers.png")
        fig_under.savefig(under_plot_path, dpi=200)
        plt.close(fig_under)

        # 5. Top 3 Individual Underpriced Case Studies
        case_studies = []
        top_gap_sub_indices = np.argsort(gaps)[::-1][:3]
        for sub_idx in top_gap_sub_indices:
            orig_idx = underpriced_indices[sub_idx]
            listing_row = df_test.iloc[orig_idx]
            listing_shap = pd.Series(shap_values[orig_idx], index=feature_names)

            pos_contributors = listing_shap.nlargest(5).to_dict()
            neg_contributors = listing_shap.nsmallest(5).to_dict()

            case_studies.append({
                "test_index": int(orig_idx),
                "listing_id": str(listing_row.get("LISTING_ID", f"idx_{orig_idx}")),
                "city": str(listing_row.get("CITY", "Unknown")),
                "room_type": str(listing_row.get("ROOM_TYPE", "Unknown")),
                "accommodates": int(listing_row.get("ACCOMMODATES", 0)),
                "actual_price": round(float(y_test[orig_idx]), 2),
                "predicted_price": round(float(y_pred[orig_idx]), 2),
                "underpriced_gap_usd": round(float(gaps[sub_idx]), 2),
                "top_positive_shap_drivers": {k: round(float(v), 2) for k, v in pos_contributors.items()},
                "top_negative_shap_drivers": {k: round(float(v), 2) for k, v in neg_contributors.items()}
            })

        return {
            "total_test_samples": len(df_test),
            "underpriced_count": num_underpriced,
            "underpriced_pct": round(underpriced_pct, 4),
            "avg_underpriced_gap_usd": round(avg_underpriced_gap, 2),
            "expected_value_base_price": round(expected_value, 2),
            "global_top_drivers": global_importance.head(10).to_dict(),
            "underpriced_positive_drivers": mean_shap_under.head(10).to_dict(),
            "underpriced_most_impactful": mean_abs_under.head(10).to_dict(),
            "plots": {
                "global_importance": global_plot_path,
                "underpriced_drivers": under_plot_path
            },
            "case_studies": case_studies
        }
