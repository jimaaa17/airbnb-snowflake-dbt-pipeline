"""Real-time scoring service with Pydantic schemas for API serving."""

from typing import Dict, Any, Optional
import pandas as pd
from pydantic import BaseModel, Field
from ml.models.cancellation_classifier import CancellationClassifier
from ml.models.price_regressor import PriceRegressor

class CancellationPredictionRequest(BaseModel):
    booking_date: str = Field(..., examples=["2024-06-15"])
    booking_created_at: str = Field(..., examples=["2024-05-20"])
    total_amount: float = Field(..., examples=[450.0])
    cleaning_fee: float = Field(..., examples=[60.0])
    service_fee: float = Field(..., examples=[45.0])
    accommodates: int = Field(..., examples=[4])
    bedrooms: int = Field(..., examples=[2])
    bathrooms: float = Field(..., examples=[1.5])
    price_per_night: float = Field(..., examples=[150.0])
    price_per_night_tag: str = Field(..., examples=["MEDIUM"])
    property_type: str = Field(..., examples=["Apartment"])
    room_type: str = Field(..., examples=["Entire home"])
    city: str = Field(..., examples=["Paris"])
    is_superhost: str = Field(..., examples=["TRUE"])
    response_rate: float = Field(..., examples=[95.0])
    response_rate_band: str = Field(..., examples=["VERY GOOD"])
    listing_id: str = Field(default="LST_0101")
    booking_id: str = Field(default="BKG_ONLINE")

class CancellationPredictionResponse(BaseModel):
    cancellation_probability: float
    cancellation_risk_level: str
    predicted_is_cancelled: int
    threshold_applied: float

class PricePredictionRequest(BaseModel):
    accommodates: int = Field(..., examples=[4])
    bedrooms: int = Field(..., examples=[2])
    bathrooms: float = Field(..., examples=[1.5])
    cleaning_fee: float = Field(..., examples=[60.0])
    property_type: str = Field(..., examples=["Apartment"])
    room_type: str = Field(..., examples=["Entire home"])
    city: str = Field(..., examples=["Paris"])
    is_superhost: str = Field(..., examples=["TRUE"])
    response_rate: float = Field(..., examples=[95.0])
    response_rate_band: str = Field(..., examples=["VERY GOOD"])
    booking_date: Optional[str] = Field(default="2024-06-15")
    booking_created_at: Optional[str] = Field(default="2024-05-20")

class PricePredictionResponse(BaseModel):
    predicted_fair_price_per_night: float
    recommended_min_guardrail: float
    recommended_max_guardrail: float

class ModelInferenceService:
    """Manages loaded models and executes real-time inference."""

    def __init__(self, cancellation_model_path: str, price_model_path: str):
        self.cancellation_model = CancellationClassifier({}).load(cancellation_model_path)
        self.price_model = PriceRegressor({}).load(price_model_path)

    def predict_cancellation(self, req: CancellationPredictionRequest) -> CancellationPredictionResponse:
        row = {
            "BOOKING_ID": req.booking_id,
            "LISTING_ID": req.listing_id,
            "BOOKING_DATE": req.booking_date,
            "BOOKING_CREATED_AT": req.booking_created_at,
            "TOTAL_AMOUNT": req.total_amount,
            "CLEANING_FEE": req.cleaning_fee,
            "SERVICE_FEE": req.service_fee,
            "ACCOMMODATES": req.accommodates,
            "BEDROOMS": req.bedrooms,
            "BATHROOMS": req.bathrooms,
            "PRICE_PER_NIGHT": req.price_per_night,
            "PRICE_PER_NIGHT_TAG": req.price_per_night_tag,
            "PROPERTY_TYPE": req.property_type,
            "ROOM_TYPE": req.room_type,
            "CITY": req.city,
            "IS_SUPERHOST": req.is_superhost,
            "RESPONSE_RATE": req.response_rate,
            "RESPONSE_RATE_BAND": req.response_rate_band,
            "BOOKING_STATUS": "confirmed"  # placeholder for feature store computation
        }
        df = pd.DataFrame([row])
        prob = float(self.cancellation_model.predict_proba(df)[0])
        threshold = self.cancellation_model.decision_threshold
        pred = int(prob >= threshold)

        risk_level = "HIGH" if prob >= 0.60 else ("MEDIUM" if prob >= threshold else "LOW")

        return CancellationPredictionResponse(
            cancellation_probability=round(prob, 4),
            cancellation_risk_level=risk_level,
            predicted_is_cancelled=pred,
            threshold_applied=threshold
        )

    def predict_fair_price(self, req: PricePredictionRequest) -> PricePredictionResponse:
        row = {
            "ACCOMMODATES": req.accommodates,
            "BEDROOMS": req.bedrooms,
            "BATHROOMS": req.bathrooms,
            "CLEANING_FEE": req.cleaning_fee,
            "PROPERTY_TYPE": req.property_type,
            "ROOM_TYPE": req.room_type,
            "CITY": req.city,
            "IS_SUPERHOST": req.is_superhost,
            "RESPONSE_RATE": req.response_rate,
            "RESPONSE_RATE_BAND": req.response_rate_band,
            "BOOKING_DATE": req.booking_date,
            "BOOKING_CREATED_AT": req.booking_created_at
        }
        df = pd.DataFrame([row])
        pred_price = float(self.price_model.predict(df)[0])
        fair_price = round(max(25.0, pred_price), 2)

        return PricePredictionResponse(
            predicted_fair_price_per_night=fair_price,
            recommended_min_guardrail=round(fair_price * 0.85, 2),
            recommended_max_guardrail=round(fair_price * 1.25, 2)
        )
