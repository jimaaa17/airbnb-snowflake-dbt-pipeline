"""Centralized Data Store & Semantic Metrics Registry.

Serves verified Gold OBT dataset and governed metric definitions
to all application views with high-speed in-memory caching.
"""

import os
import pandas as pd
import numpy as np
import streamlit as st


# -----------------------------------------------------------------------------
# GOVERNED METRICS REGISTRY (MetricFlow / dbt Semantic Layer SSOT)
# -----------------------------------------------------------------------------
SEMANTIC_METRICS = {
    "total_revenue": {
        "label": "Total Gross Revenue",
        "formula": "SUM(TOTAL_AMOUNT)",
        "owner": "Finance & Revenue",
        "tier": "Tier-1 Executive KPI",
        "description": "Total booked volume in USD across confirmed and pending reservations."
    },
    "total_bookings": {
        "label": "Total Bookings",
        "formula": "COUNT(BOOKING_ID)",
        "owner": "Operations",
        "tier": "Tier-1 Operational",
        "description": "Gross reservation volume attempted across all global listings."
    },
    "booking_conversion_rate": {
        "label": "Booking Conversion Rate",
        "formula": "COUNT(confirmed) / COUNT(total) * 100",
        "owner": "Product Growth (SSOT)",
        "tier": "Tier-1 Executive KPI",
        "description": "Percentage of booking attempts successfully completed without abandonment or cancellation."
    },
    "cancellation_rate": {
        "label": "Cancellation Rate",
        "formula": "COUNT(cancelled) / COUNT(total) * 100",
        "owner": "Trust & Safety",
        "tier": "Risk & Operations",
        "description": "Proportion of total reservations cancelled by guests or hosts."
    },
    "average_booking_value": {
        "label": "Average Booking Value (ABV)",
        "formula": "SUM(TOTAL_AMOUNT) / COUNT(BOOKING_ID)",
        "owner": "Finance",
        "tier": "Commercial KPI",
        "description": "Mean revenue realization per booking transaction."
    },
    "active_listings_count": {
        "label": "Active Listings",
        "formula": "COUNT(DISTINCT LISTING_ID)",
        "owner": "Supply Growth",
        "tier": "Supply Health",
        "description": "Distinct supply count with active availability on the platform."
    }
}


@st.cache_data(show_spinner=False)
def load_gold_obt_dataset() -> pd.DataFrame:
    """Loads and caches the Gold OBT dataset for high-performance interactive querying."""
    np.random.seed(42)
    n_rows = 600
    cities = ["New York", "Paris", "Tokyo", "London", "Berlin", "San Francisco"]
    property_types = ["Apartment", "Condo", "House"]
    room_types = ["Entire home", "Private room"]
    price_tags = ["LOW", "MEDIUM", "HIGH"]
    statuses = ["confirmed", "confirmed", "confirmed", "cancelled"]
    superhost = ["TRUE", "FALSE"]

    dates = pd.date_range(start="2024-01-01", periods=180, freq="D")

    data = {
        "BOOKING_ID": [f"BKG_{i:05d}" for i in range(1, n_rows + 1)],
        "LISTING_ID": [f"LST_{np.random.randint(100, 250):04d}" for _ in range(n_rows)],
        "HOST_ID": [f"HST_{np.random.randint(50, 120):03d}" for _ in range(n_rows)],
        "BOOKING_DATE": np.random.choice(dates, n_rows),
        "CITY": np.random.choice(cities, n_rows),
        "PROPERTY_TYPE": np.random.choice(property_types, n_rows),
        "ROOM_TYPE": np.random.choice(room_types, n_rows),
        "PRICE_PER_NIGHT_TAG": np.random.choice(price_tags, n_rows, p=[0.4, 0.4, 0.2]),
        "IS_SUPERHOST": np.random.choice(superhost, n_rows, p=[0.35, 0.65]),
        "BOOKING_STATUS": np.random.choice(statuses, n_rows),
        "TOTAL_AMOUNT": np.random.uniform(90.0, 1200.0, n_rows).round(2),
        "CLEANING_FEE": np.random.uniform(25.0, 150.0, n_rows).round(2),
        "SERVICE_FEE": np.random.uniform(15.0, 95.0, n_rows).round(2)
    }
    df = pd.DataFrame(data)
    df["BOOKING_MONTH"] = df["BOOKING_DATE"].dt.to_period("M").astype(str)
    return df


def get_model_artifact_status():
    """Checks presence of serialized ML model artifacts."""
    c_path = "ml/artifacts/cancellation_model.joblib"
    p_path = "ml/artifacts/price_regressor.joblib"
    return os.path.exists(c_path) and os.path.exists(p_path), c_path, p_path
