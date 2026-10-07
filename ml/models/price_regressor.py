"""Listing dynamic price and fair value regressor pipeline."""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor

from ml.models.base import BaseAirbnbModel
from ml.features.transformers import AirbnbFeatureEngineer
from ml.evaluation.metrics import evaluate_regression, evaluate_pricing_sme_impact

class PriceRegressor(BaseAirbnbModel):
    """Predicts fair market price per night for a listing."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config or {})
        if "features" in self.config:
            self.build_pipeline()

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
        estimator = GradientBoostingRegressor(
            n_estimators=hp.get("n_estimators", 150),
            max_depth=hp.get("max_depth", 5),
            learning_rate=hp.get("learning_rate", 0.05),
            random_state=hp.get("random_state", 42)
        )

        self.pipeline = Pipeline(steps=[
            ("feature_engineer", AirbnbFeatureEngineer()),
            ("preprocessor", preprocessor),
            ("regressor", estimator)
        ])

    def train(self, df_train: pd.DataFrame) -> Dict[str, float]:
        df = df_train.copy()
        y = pd.to_numeric(df["PRICE_PER_NIGHT"]).to_numpy()

        self.pipeline.fit(df, y)
        self.is_fitted = True

        y_pred = self.pipeline.predict(df)
        return evaluate_regression(y, y_pred)

    def evaluate(self, df_test: pd.DataFrame) -> Dict[str, Any]:
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before evaluation.")
        df = df_test.copy()
        y = pd.to_numeric(df["PRICE_PER_NIGHT"]).to_numpy()

        y_pred = self.pipeline.predict(df)
        metrics = evaluate_regression(y, y_pred)
        sme_metrics = evaluate_pricing_sme_impact(df, y, y_pred)
        metrics.update(sme_metrics)
        return metrics

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before inference.")
        return self.pipeline.predict(df)
