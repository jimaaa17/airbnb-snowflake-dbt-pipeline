"""Unit tests for MLflow Tracking, Model Registry lifecycle, and SME metrics."""

import pytest
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.pipeline import Pipeline
from ml.tracking.tracker import MLflowTracker, get_default_tracking_uri
from ml.evaluation.metrics import evaluate_cancellation_sme_impact, evaluate_pricing_sme_impact

@pytest.fixture
def tracker():
    return MLflowTracker(experiment_name="airbnb_test_suite")

def test_mlflow_tracker_initialization(tracker):
    """Verify MLflowTracker correctly points to SQLite backend and experiment."""
    assert "sqlite" in tracker.tracking_uri
    assert tracker.experiment_id is not None
    assert tracker.experiment_name == "airbnb_test_suite"

def test_model_registration_and_alias(tracker):
    """Verify logging a pipeline registers it in Model Registry and assigns alias."""
    # Create simple dummy model
    dummy_pipe = Pipeline([("clf", DummyClassifier(strategy="constant", constant=1))])
    X = np.array([[1], [2], [3], [4]])
    y = np.array([1, 1, 1, 1])
    dummy_pipe.fit(X, y)

    res = tracker.log_and_register_model(
        run_name="unit_test_run",
        model_name="unit_test_dummy_model",
        pipeline=dummy_pipe,
        parameters={"alpha": 0.1, "strategy": "constant"},
        metrics={"accuracy": 1.0, "roc_auc": 0.95},
        alias="champion"
    )

    assert res["model_name"] == "unit_test_dummy_model"
    assert res["alias"] == "champion"
    assert res["version"] is not None

    # Load from registry via alias
    loaded_pipe = tracker.load_registered_model("unit_test_dummy_model", alias_or_version="champion")
    preds = loaded_pipe.predict(np.array([[5]]))
    assert preds[0] == 1

def test_cancellation_sme_metrics():
    """Verify SME metrics computation for cancellation impact."""
    df_test = pd.DataFrame({
        "TOTAL_AMOUNT": [100.0, 200.0, 300.0, 400.0],
        "BOOKING_DATE": pd.to_datetime(["2024-06-20", "2024-06-25", "2024-07-01", "2024-07-10"]),
        "BOOKING_CREATED_AT": pd.to_datetime(["2024-06-01", "2024-06-05", "2024-06-10", "2024-06-15"])
    })
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0])  # Caught 1 of the 2 cancellations
    y_prob = np.array([0.9, 0.1, 0.4, 0.2])

    metrics = evaluate_cancellation_sme_impact(df_test, y_true, y_pred, y_prob)

    assert metrics["sme_total_bookings_evaluated"] == 4
    assert metrics["sme_total_booking_volume_usd"] == 1000.0
    assert metrics["sme_revenue_at_risk_usd"] == 400.0  # 100 + 300
    assert metrics["sme_revenue_protected_usd"] == 100.0  # Caught index 0
    assert metrics["sme_protection_capture_rate"] == 0.25
    assert metrics["sme_estimated_salvaged_revenue_usd"] == 35.0

def test_pricing_sme_metrics():
    """Verify SME metrics computation for pricing optimization."""
    df_test = pd.DataFrame()
    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    # Case 0: actual 100 vs fair 130 -> underpriced (>10% gap)
    # Case 1: actual 200 vs fair 205 -> within guardrail
    # Case 2: actual 300 vs fair 240 -> overpriced (>10% gap)
    # Case 3: actual 400 vs fair 410 -> within guardrail
    y_pred = np.array([130.0, 205.0, 240.0, 410.0])

    metrics = evaluate_pricing_sme_impact(df_test, y_true, y_pred)

    assert metrics["sme_underpriced_listings_pct"] == 0.25
    assert metrics["sme_overpriced_listings_pct"] == 0.25
    assert metrics["sme_within_guardrails_pct"] == 0.50
    assert metrics["sme_avg_underpriced_gap_usd"] == 30.0
    assert metrics["sme_estimated_monthly_uplift_per_listing_usd"] == 450.0  # 30 * 15 nights
