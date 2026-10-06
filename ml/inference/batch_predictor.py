"""Batch prediction runner for offline scoring pipelines."""

import os
from typing import Optional
import pandas as pd
from ml.models.cancellation_classifier import CancellationClassifier
from ml.models.price_regressor import PriceRegressor

class BatchPredictor:
    """Executes scalable batch scoring across large booking/listing partitions."""

    def __init__(self, cancellation_model_path: str, price_model_path: str):
        self.cancellation_model = CancellationClassifier({}).load(cancellation_model_path)
        self.price_model = PriceRegressor({}).load(price_model_path)

    def score_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enriches dataset with model predictions."""
        scored_df = df.copy()

        # Score cancellation risk
        probs = self.cancellation_model.predict_proba(scored_df)
        preds = self.cancellation_model.predict(scored_df)
        scored_df["PRED_CANCEL_PROB"] = [round(float(p), 4) for p in probs]
        scored_df["PRED_IS_CANCELLED"] = preds

        # Score fair market nightly price
        fair_prices = self.price_model.predict(scored_df)
        scored_df["PRED_FAIR_PRICE"] = [round(float(p), 2) for p in fair_prices]
        scored_df["PRICE_VARIANCE_PCT"] = round(
            (scored_df["PRICE_PER_NIGHT"] - scored_df["PRED_FAIR_PRICE"]) / scored_df["PRED_FAIR_PRICE"] * 100.0,
            2
        )

        return scored_df
