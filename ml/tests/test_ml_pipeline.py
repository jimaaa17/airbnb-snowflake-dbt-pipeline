"""Unit tests for ML engineering components, transformers, and leakage prevention."""

import pytest
import pandas as pd
import numpy as np
from ml.data.datasets import load_gold_obt_dataset, temporal_train_test_split
from ml.features.transformers import (
    AirbnbFeatureEngineer,
    compute_lead_time_features,
    compute_calendar_seasonality_features
)
from ml.features.feature_store import ZiplineFeatureStore

def test_temporal_split_order():
    """Verify temporal train/test split maintains strict chronological order."""
    df = load_gold_obt_dataset(limit=200, offline_mode=True)
    train_df, test_df = temporal_train_test_split(df, time_col="BOOKING_DATE", train_ratio=0.75)
    
    assert len(train_df) == 150
    assert len(test_df) == 50
    assert train_df["BOOKING_DATE"].max() <= test_df["BOOKING_DATE"].min()

def test_feature_engineer_transformation():
    """Verify AirbnbFeatureEngineer generates domain financial, seasonality, and capacity ratios."""
    df = load_gold_obt_dataset(limit=50, offline_mode=True)
    engineer = AirbnbFeatureEngineer()
    transformed = engineer.fit_transform(df)

    # Lead time & behavioral bins
    assert "lead_time_days" in transformed.columns
    assert "lead_time_invalid" in transformed.columns
    assert "lead_time_missing" in transformed.columns
    assert "is_last_minute" in transformed.columns
    assert "is_short_notice" in transformed.columns
    assert "is_far_advance" in transformed.columns
    assert "lead_time_log" in transformed.columns

    # Seasonality & calendar features
    assert "arrival_month" in transformed.columns
    assert "arrival_dow" in transformed.columns
    assert "is_weekend_arrival" in transformed.columns
    assert "arrival_quarter" in transformed.columns
    assert "arrival_month_sin" in transformed.columns
    assert "arrival_month_cos" in transformed.columns
    assert "arrival_dow_sin" in transformed.columns
    assert "arrival_dow_cos" in transformed.columns

    # Financial & capacity fee ratios
    assert "cleaning_fee_ratio" in transformed.columns
    assert "service_fee_ratio" in transformed.columns
    assert "cleaning_fee_per_bedroom" in transformed.columns
    assert "cleaning_fee_per_accommodate" in transformed.columns
    assert "price_per_accommodate" in transformed.columns
    assert "is_superhost_binary" in transformed.columns

    assert (transformed["lead_time_days"] >= 0).all()
    assert (transformed["arrival_month"].between(1, 12)).all()

def test_lead_time_edge_cases():
    """Verify robust lead time handles sub-day timing, tz-mismatch, and bad inputs."""
    data = {
        "BOOKING_DATE": [
            "2024-06-10",                # Same-day booking (afternoon creation)
            "2024-06-15",                # Standard 5-day advance booking
            "2024-06-01",                # Retroactive / negative lead time
            None,                        # Missing date
            "corrupt_date_value",        # Unparseable string
            "2024-06-20T00:00:00Z",      # TZ-aware ISO string
        ],
        "BOOKING_CREATED_AT": [
            "2024-06-10 16:45:00",       # Afternoon creation on same calendar day
            "2024-06-10 09:00:00",       # Morning creation
            "2024-06-10 08:00:00",       # Created after stay
            "2024-06-10 00:00:00",       # Created valid, stay missing
            "2024-06-10 00:00:00",       # Created valid, stay corrupt
            "2024-06-15",                # Naive date string
        ]
    }
    df = pd.DataFrame(data)
    result = compute_lead_time_features(df)

    # 1. Same-day booking must NOT be negative (-1) or invalid
    assert result.loc[0, "lead_time_days"] == 0
    assert result.loc[0, "lead_time_invalid"] == 0
    assert result.loc[0, "lead_time_missing"] == 0

    # 2. Advance booking evaluates to exact calendar days
    assert result.loc[1, "lead_time_days"] == 5
    assert result.loc[1, "lead_time_invalid"] == 0

    # 3. Retroactive booking is flagged as invalid and clipped to 0
    assert result.loc[2, "lead_time_days"] == 0
    assert result.loc[2, "lead_time_invalid"] == 1
    assert result.loc[2, "lead_time_missing"] == 0

    # 4. Missing date is flagged as missing
    assert result.loc[3, "lead_time_days"] == 0
    assert result.loc[3, "lead_time_missing"] == 1

    # 5. Corrupt date is coerced to NaT without crashing
    assert result.loc[4, "lead_time_days"] == 0
    assert result.loc[4, "lead_time_missing"] == 1

    # 6. TZ-aware vs naive does not throw TypeError and computes correctly
    assert result.loc[5, "lead_time_days"] == 5
    assert result.loc[5, "lead_time_invalid"] == 0

def test_lead_time_missing_columns_validation():
    """Verify behavior when required columns are missing."""
    df_empty = pd.DataFrame({"OTHER_COL": [1, 2]})
    
    # Graceful fallback by default
    res = compute_lead_time_features(df_empty, require_columns=False)
    assert res["lead_time_days"].tolist() == [0.0, 0.0]
    assert res["lead_time_missing"].tolist() == [1, 1]

    # Strict mode raises informative ValueError
    with pytest.raises(ValueError, match="Missing required booking date column"):
        compute_lead_time_features(df_empty, require_columns=True)


def test_calendar_seasonality_features_edge_cases():
    """Verify seasonality helper handles weekend detection, cyclical math, and missing dates."""
    data = {
        "BOOKING_DATE": [
            "2024-06-14",  # Friday -> is_weekend_arrival = 1
            "2024-06-15",  # Saturday -> is_weekend_arrival = 1
            "2024-06-12",  # Wednesday -> is_weekend_arrival = 0
            "2024-01-01",  # January -> month = 1, sin = 0, cos = 1
            "2024-12-31",  # December -> month = 12
            None,          # Missing date -> missing flag = 1
            "bad_string"   # Unparseable -> missing flag = 1
        ]
    }
    df = pd.DataFrame(data)
    res = compute_calendar_seasonality_features(df)

    # 1. Friday & Saturday are weekend arrivals
    assert res.loc[0, "is_weekend_arrival"] == 1
    assert res.loc[1, "is_weekend_arrival"] == 1
    assert res.loc[2, "is_weekend_arrival"] == 0

    # 2. Cyclical month check for January (angle 0 -> sin=0, cos=1)
    assert np.isclose(res.loc[3, "arrival_month_sin"], 0.0, atol=1e-5)
    assert np.isclose(res.loc[3, "arrival_month_cos"], 1.0, atol=1e-5)

    # 3. Cyclical distance between Dec (12) and Jan (1) is close on unit circle
    dec_sin = res.loc[4, "arrival_month_sin"]
    dec_cos = res.loc[4, "arrival_month_cos"]
    jan_sin = res.loc[3, "arrival_month_sin"]
    jan_cos = res.loc[3, "arrival_month_cos"]
    euc_dist = np.sqrt((dec_sin - jan_sin)**2 + (dec_cos - jan_cos)**2)
    assert euc_dist < 0.6  # Dec and Jan are close neighbors

    # 4. Missing / Corrupt dates
    assert res.loc[5, "arrival_date_missing"] == 1
    assert res.loc[5, "is_weekend_arrival"] == 0
    assert res.loc[6, "arrival_date_missing"] == 1

    # 5. Case-insensitive lookup
    df_lower = pd.DataFrame({"booking_date": ["2024-07-04"]})
    res_lower = compute_calendar_seasonality_features(df_lower)
    assert res_lower.loc[0, "arrival_month"] == 7
    assert res_lower.loc[0, "arrival_date_missing"] == 0


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
