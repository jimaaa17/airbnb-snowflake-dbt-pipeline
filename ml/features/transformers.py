"""Custom scikit-learn compatible transformers for Airbnb feature pipelines."""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class AirbnbFeatureEngineer(BaseEstimator, TransformerMixin):
    """Derives domain-specific business features from raw Gold OBT fields.
    
    Adheres to scikit-learn Transformer API so it embeds directly into Pipeline.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        
        # 1. Lead time
        if "BOOKING_DATE" in df.columns and "BOOKING_CREATED_AT" in df.columns:
            b_date = pd.to_datetime(df["BOOKING_DATE"])
            b_created = pd.to_datetime(df["BOOKING_CREATED_AT"])
            df["lead_time_days"] = (b_date - b_created).dt.days.clip(lower=0)
        else:
            df["lead_time_days"] = 0

        # 2. Financial ratios
        if "TOTAL_AMOUNT" in df.columns and "CLEANING_FEE" in df.columns:
            total_amt = df["TOTAL_AMOUNT"].replace(0, np.nan)
            df["cleaning_fee_ratio"] = (df["CLEANING_FEE"] / total_amt).fillna(0.0)
            df["service_fee_ratio"] = (df.get("SERVICE_FEE", 0.0) / total_amt).fillna(0.0)
        else:
            df["cleaning_fee_ratio"] = 0.0
            df["service_fee_ratio"] = 0.0

        # 3. Capacity ratios
        accommodates = df["ACCOMMODATES"].replace(0, 1) if "ACCOMMODATES" in df.columns else pd.Series(1, index=df.index)
        if "PRICE_PER_NIGHT" in df.columns:
            df["price_per_accommodate"] = df["PRICE_PER_NIGHT"] / accommodates
        else:
            df["price_per_accommodate"] = 0.0

        if "BEDROOMS" in df.columns:
            df["bedroom_to_accommodates_ratio"] = df["BEDROOMS"] / accommodates
        else:
            df["bedroom_to_accommodates_ratio"] = 0.5

        # 4. Host features
        if "IS_SUPERHOST" in df.columns:
            df["is_superhost_binary"] = (df["IS_SUPERHOST"].astype(str).str.upper() == "TRUE").astype(int)
        else:
            df["is_superhost_binary"] = 0

        if "RESPONSE_RATE" in df.columns:
            df["host_response_rate"] = pd.to_numeric(df["RESPONSE_RATE"], errors="coerce").fillna(80.0)
        else:
            df["host_response_rate"] = 80.0

        return df
