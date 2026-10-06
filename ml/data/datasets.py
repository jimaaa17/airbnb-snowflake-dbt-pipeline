"""Dataset loading and temporal partitioning utilities for ML pipelines."""

from typing import Tuple, Optional
import pandas as pd
from ml.data.connector import SnowflakeConnector

def load_gold_obt_dataset(limit: Optional[int] = None, offline_mode: bool = False) -> pd.DataFrame:
    """Loads raw gold dataset ready for feature engineering."""
    connector = SnowflakeConnector(offline_mode=offline_mode)
    df = connector.fetch_gold_obt(limit=limit)
    df["BOOKING_DATE"] = pd.to_datetime(df["BOOKING_DATE"])
    df["BOOKING_CREATED_AT"] = pd.to_datetime(df["BOOKING_CREATED_AT"])
    return df

def temporal_train_test_split(
    df: pd.DataFrame,
    time_col: str = "BOOKING_DATE",
    train_ratio: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits dataset chronologically to avoid future-to-past data leakage.
    
    Standard in time-series and transaction machine learning.
    """
    sorted_df = df.sort_values(by=time_col).reset_index(drop=True)
    split_idx = int(len(sorted_df) * train_ratio)
    
    train_df = sorted_df.iloc[:split_idx].copy()
    test_df = sorted_df.iloc[split_idx:].copy()
    
    return train_df, test_df
