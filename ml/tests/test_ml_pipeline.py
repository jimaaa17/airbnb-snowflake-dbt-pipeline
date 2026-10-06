"""Unit tests for ML engineering components, transformers, and leakage prevention."""

import pytest
import pandas as pd
import numpy as np
from ml.data.datasets import load_gold_obt_dataset, temporal_train_test_split
from ml.features.transformers import AirbnbFeatureEngineer
from ml.features.feature_store import ZiplineFeatureStore

def test_temporal_split_order():
    """Verify temporal train/test split maintains strict chronological order."""
    df = load_gold_obt_dataset(limit=200, offline_mode=True)
    train_df, test_df = temporal_train_test_split(df, time_col="BOOKING_DATE", train_ratio=0.75)
    
    assert len(train_df) == 150
    assert len(test_df) == 50
    assert train_df["BOOKING_DATE"].max() <= test_df["BOOKING_DATE"].min()

def test_feature_engineer_transformation():
    """Verify AirbnbFeatureEngineer generates domain financial and capacity ratios."""
    df = load_gold_obt_dataset(limit=50, offline_mode=True)
    engineer = AirbnbFeatureEngineer()
    transformed = engineer.fit_transform(df)

    assert "lead_time_days" in transformed.columns
    assert "cleaning_fee_ratio" in transformed.columns
    assert "service_fee_ratio" in transformed.columns
    assert "price_per_accommodate" in transformed.columns
    assert "is_superhost_binary" in transformed.columns
    assert (transformed["lead_time_days"] >= 0).all()

def test_zipline_feature_store_no_lookahead():
    """Verify Zipline point-in-time window feature does not look into the future."""
    # Synthetic 3 events for same listing
    data = {
        "LISTING_ID": ["L1", "L1", "L1"],
        "BOOKING_CREATED_AT": [
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-01-10"),
            pd.Timestamp("2024-01-20")
        ],
        "BOOKING_STATUS": ["cancelled", "confirmed", "confirmed"]
    }
    df = pd.DataFrame(data)
    store = ZiplineFeatureStore(window_days=30)
    enriched = store.compute_as_of_listing_features(df)

    # First event has 0 prior bookings
    assert enriched.loc[0, "trailing_30d_listing_bookings"] == 0
    assert enriched.loc[0, "trailing_30d_listing_cancellations"] == 0

    # Second event sees event 0 (which was cancelled)
    assert enriched.loc[1, "trailing_30d_listing_bookings"] == 1
    assert enriched.loc[1, "trailing_30d_listing_cancellations"] == 1

    # Third event sees event 0 and event 1 (1 cancel out of 2 bookings)
    assert enriched.loc[2, "trailing_30d_listing_bookings"] == 2
    assert enriched.loc[2, "trailing_30d_listing_cancellations"] == 1
