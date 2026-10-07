"""Cancellation risk classifier pipeline."""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier

from ml.models.base import BaseAirbnbModel
from ml.features.transformers import AirbnbFeatureEngineer
from ml.features.feature_store import ZiplineFeatureStore
from ml.evaluation.metrics import evaluate_classification, evaluate_cancellation_sme_impact

class CancellationClassifier(BaseAirbnbModel):
    """Predicts probability of a booking cancellation at reservation creation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config or {})
        self.feature_store = ZiplineFeatureStore(window_days=30)
        self.decision_threshold = self.config.get("decision_threshold", 0.35)
        if "features" in self.config:
            self.build_pipeline()

    def load(self, filepath: str):
        super().load(filepath)
        self.decision_threshold = self.config.get("decision_threshold", 0.35)
        return self

    def build_pipeline(self):
        num_cols = self.config["features"]["numeric"]
        cat_cols = self.config["features"]["categorical"]

        num_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        cat_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        preprocessor = ColumnTransformer(transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols)
        ])

        hp = self.config.get("hyperparameters", {})
        estimator = GradientBoostingClassifier(
            n_estimators=hp.get("n_estimators", 120),
            max_depth=hp.get("max_depth", 6),
            learning_rate=hp.get("learning_rate", 0.08),
            subsample=hp.get("subsample", 0.85),
            random_state=hp.get("random_state", 42)
        )

        self.pipeline = Pipeline(steps=[
            ("feature_engineer", AirbnbFeatureEngineer()),
            ("preprocessor", preprocessor),
            ("classifier", estimator)
        ])

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enriches dataframe with point-in-time sliding window features."""
        return self.feature_store.compute_as_of_listing_features(df)

    def train(self, df_train: pd.DataFrame) -> Dict[str, Any]:
        enriched_df = self._prepare_data(df_train)
        y = (enriched_df["BOOKING_STATUS"] == "cancelled").astype(int).to_numpy()

        self.pipeline.fit(enriched_df, y)
        self.is_fitted = True

        y_prob = self.pipeline.predict_proba(enriched_df)[:, 1]
        y_pred = (y_prob >= self.decision_threshold).astype(int)

        return evaluate_classification(y, y_pred, y_prob)

    def evaluate(self, df_test: pd.DataFrame) -> Dict[str, Any]:
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before evaluation.")
        enriched_df = self._prepare_data(df_test)
        y = (enriched_df["BOOKING_STATUS"] == "cancelled").astype(int).to_numpy()

        y_prob = self.pipeline.predict_proba(enriched_df)[:, 1]
        y_pred = (y_prob >= self.decision_threshold).astype(int)

        metrics = evaluate_classification(y, y_pred, y_prob)
        sme_metrics = evaluate_cancellation_sme_impact(df_test, y, y_pred, y_prob)
        metrics.update(sme_metrics)
        return metrics

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before inference.")
        enriched_df = self._prepare_data(df)
        return self.pipeline.predict_proba(enriched_df)[:, 1]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(df)
        return (probs >= self.decision_threshold).astype(int)
