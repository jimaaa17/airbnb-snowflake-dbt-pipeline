"""Unit tests for inference services and batch predictor."""

import pytest
import pandas as pd
from ml.inference.service import (
    ModelInferenceService,
    CancellationPredictionRequest,
    PricePredictionRequest
)
from ml.inference.batch_predictor import BatchPredictor
from ml.data.datasets import load_gold_obt_dataset

@pytest.fixture
def inference_service():
    return ModelInferenceService(
        cancellation_model_path="ml/artifacts/cancellation_model.joblib",
        price_model_path="ml/artifacts/price_regressor.joblib"
    )

def test_realtime_cancellation_prediction(inference_service):
    req = CancellationPredictionRequest(
        booking_date="2024-06-15",
        booking_created_at="2024-05-15",
        total_amount=520.0,
        cleaning_fee=75.0,
        service_fee=50.0,
        accommodates=4,
        bedrooms=2,
        bathrooms=1.5,
        price_per_night=180.0,
        price_per_night_tag="MEDIUM",
        property_type="Apartment",
        room_type="Entire home",
        city="Paris",
        is_superhost="TRUE",
        response_rate=98.0,
        response_rate_band="VERY GOOD"
    )
    res = inference_service.predict_cancellation(req)

    assert 0.0 <= res.cancellation_probability <= 1.0
    assert res.cancellation_risk_level in ["LOW", "MEDIUM", "HIGH"]
    assert res.predicted_is_cancelled in [0, 1]

def test_realtime_price_prediction(inference_service):
    req = PricePredictionRequest(
        accommodates=4,
        bedrooms=2,
        bathrooms=1.5,
        cleaning_fee=60.0,
        property_type="Apartment",
        room_type="Entire home",
        city="Paris",
        is_superhost="TRUE",
        response_rate=95.0,
        response_rate_band="VERY GOOD"
    )
    res = inference_service.predict_fair_price(req)

    assert res.predicted_fair_price_per_night > 0
    assert res.recommended_min_guardrail < res.predicted_fair_price_per_night
    assert res.recommended_max_guardrail > res.predicted_fair_price_per_night

def test_batch_predictor():
    batch = BatchPredictor(
        cancellation_model_path="ml/artifacts/cancellation_model.joblib",
        price_model_path="ml/artifacts/price_regressor.joblib"
    )
    df = load_gold_obt_dataset(limit=25, offline_mode=True)
    scored = batch.score_dataset(df)

    assert "PRED_CANCEL_PROB" in scored.columns
    assert "PRED_IS_CANCELLED" in scored.columns
    assert "PRED_FAIR_PRICE" in scored.columns
    assert len(scored) == 25
