"""Real-time scoring service with Pydantic schemas for API serving."""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
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
    actual_price: Optional[float] = Field(default=None, description="Actual listed nightly price for gap diagnostics")

class PricePredictionResponse(BaseModel):
    predicted_fair_price_per_night: float
    recommended_min_guardrail: float
    recommended_max_guardrail: float
    actual_price: Optional[float] = None
    price_gap: Optional[float] = None
    pricing_status: Optional[str] = None
    monthly_opportunity_usd: Optional[float] = None

class PriceExplanationResponse(BaseModel):
    predicted_fair_price_per_night: float
    base_expected_value: float
    net_shap_adjustment: float
    recommended_min_guardrail: float
    recommended_max_guardrail: float
    contributions: list[Dict[str, Any]]
    actual_price: Optional[float] = None
    price_gap: Optional[float] = None
    pricing_status: Optional[str] = None
    monthly_opportunity_usd: Optional[float] = None

class ModelInferenceService:
    """Manages loaded models and executes real-time inference via MLflow Registry or local binaries."""

    def __init__(
        self,
        cancellation_model_path: str = "models:/cancellation_classifier@champion",
        price_model_path: str = "models:/price_regressor@champion"
    ):
        # 1. Load Cancellation Model
        try:
            if cancellation_model_path.startswith("models:/"):
                import mlflow
                from ml.tracking.tracker import get_default_tracking_uri
                mlflow.set_tracking_uri(get_default_tracking_uri())
                c_pipe = mlflow.sklearn.load_model(cancellation_model_path)
                self.cancellation_model = CancellationClassifier({})
                self.cancellation_model.pipeline = c_pipe
                self.cancellation_model.is_fitted = True
            else:
                self.cancellation_model = CancellationClassifier({}).load(cancellation_model_path)
        except Exception:
            fallback = "ml/artifacts/cancellation_model.joblib"
            self.cancellation_model = CancellationClassifier({}).load(fallback)

        # 2. Load Price Regressor Model
        try:
            if price_model_path.startswith("models:/"):
                import mlflow
                from ml.tracking.tracker import get_default_tracking_uri
                mlflow.set_tracking_uri(get_default_tracking_uri())
                p_pipe = mlflow.sklearn.load_model(price_model_path)
                self.price_model = PriceRegressor({})
                self.price_model.pipeline = p_pipe
                self.price_model.is_fitted = True
            else:
                self.price_model = PriceRegressor({}).load(price_model_path)
        except Exception:
            fallback = "ml/artifacts/price_regressor.joblib"
            self.price_model = PriceRegressor({}).load(fallback)

        self.price_explainer = None

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
        # Ensure all expected seasonality columns exist (they will be generated by the feature engineer, but guarantee preprocessor sees them)
        for col in ["arrival_month", "arrival_dow", "arrival_quarter", "arrival_month_sin", "arrival_month_cos", "arrival_dow_sin", "arrival_dow_cos", "is_weekend_arrival", "arrival_date_missing"]:
            if col not in df.columns:
                df[col] = 0.0
        pred_price = float(self.price_model.predict(df)[0])
        fair_price = round(max(25.0, pred_price), 2)

        price_gap = None
        pricing_status = None
        monthly_opp = None
        if req.actual_price is not None:
            price_gap = round(fair_price - req.actual_price, 2)
            if req.actual_price < (fair_price * 0.90):
                pricing_status = "UNDERPRICED"
                monthly_opp = round(price_gap * 15.0, 2)
            elif req.actual_price > (fair_price * 1.10):
                pricing_status = "OVERPRICED"
            else:
                pricing_status = "WITHIN_GUARDRAILS"

        return PricePredictionResponse(
            predicted_fair_price_per_night=fair_price,
            recommended_min_guardrail=round(fair_price * 0.85, 2),
            recommended_max_guardrail=round(fair_price * 1.25, 2),
            actual_price=req.actual_price,
            price_gap=price_gap,
            pricing_status=pricing_status,
            monthly_opportunity_usd=monthly_opp
        )

    def explain_fair_price(self, req: PricePredictionRequest) -> PriceExplanationResponse:
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
        # Ensure seasonality columns for SHAP explanation
        for col in ["arrival_month", "arrival_dow", "arrival_quarter", "arrival_month_sin", "arrival_month_cos", "arrival_dow_sin", "arrival_dow_cos", "is_weekend_arrival", "arrival_date_missing"]:
            if col not in df.columns:
                df[col] = 0.0
        pred_price = float(self.price_model.predict(df)[0])
        fair_price = round(max(25.0, pred_price), 2)

        if self.price_explainer is None:
            import shap
            regressor = self.price_model.pipeline.named_steps["regressor"]
            self.price_explainer = shap.TreeExplainer(regressor)

        fe = self.price_model.pipeline.named_steps["feature_engineer"]
        prep = self.price_model.pipeline.named_steps["preprocessor"]
        df_fe = fe.transform(df)
        X_proc = prep.transform(df_fe)
        feat_names = list(prep.get_feature_names_out())
        shap_vals = self.price_explainer(X_proc)[0].values
        ev = float(np.ravel(self.price_explainer.expected_value)[0])

        friendly_labels = {
            "num__ACCOMMODATES": f"Guest Capacity (Accommodates: {req.accommodates})",
            "num__BEDROOMS": f"Bedrooms Count ({req.bedrooms})",
            "num__BATHROOMS": f"Bathrooms Count ({req.bathrooms})",
            "num__CLEANING_FEE": f"Cleaning Fee (${req.cleaning_fee:.0f})",
            "num__bedroom_to_accommodates_ratio": "Bedrooms-to-Capacity Ratio",
            "num__cleaning_fee_per_bedroom": "Cleaning Fee per Bedroom",
            "num__cleaning_fee_per_accommodate": "Cleaning Fee per Guest",
            "num__arrival_month": f"Arrival Month ({req.booking_date[:7] if req.booking_date else 'N/A'})",
            "num__arrival_dow": "Arrival Day of Week",
            "num__is_weekend_arrival": "Weekend Check-in Premium",
            "num__arrival_quarter": "Arrival Quarter",
            "num__arrival_month_sin": "Summer Peak Seasonality (Sin)",
            "num__arrival_month_cos": "Spring/Fall Seasonality (Cos)",
            "num__arrival_dow_sin": "Weekly Cycle (Sin)",
            "num__arrival_dow_cos": "Weekly Cycle (Cos)",
            "num__host_response_rate": f"Host Responsiveness ({req.response_rate}%)",
            "num__is_superhost_binary": f"Superhost Tier ({req.is_superhost})"
        }

        contributions = []
        for feat_name, impact in zip(feat_names, shap_vals):
            if abs(impact) < 0.05:
                continue
            if feat_name.startswith("cat__CITY_"):
                city_val = feat_name.replace("cat__CITY_", "")
                display = f"Market Location: {city_val}"
            elif feat_name.startswith("cat__ROOM_TYPE_"):
                rtype_val = feat_name.replace("cat__ROOM_TYPE_", "")
                display = f"Room Category: {rtype_val}"
            elif feat_name.startswith("cat__PROPERTY_TYPE_"):
                ptype_val = feat_name.replace("cat__PROPERTY_TYPE_", "")
                display = f"Property Type: {ptype_val}"
            elif feat_name.startswith("cat__RESPONSE_RATE_BAND_"):
                rband_val = feat_name.replace("cat__RESPONSE_RATE_BAND_", "")
                display = f"Response Band: {rband_val}"
            else:
                display = friendly_labels.get(feat_name, feat_name.replace("num__", "").replace("_", " ").title())

            contributions.append({
                "feature": feat_name,
                "display_name": display,
                "impact": round(float(impact), 2),
                "direction": "Increases Price" if impact > 0 else "Reduces Price"
            })

        contributions.sort(key=lambda x: abs(x["impact"]), reverse=True)

        price_gap = None
        pricing_status = None
        monthly_opp = None
        if req.actual_price is not None:
            price_gap = round(fair_price - req.actual_price, 2)
            if req.actual_price < (fair_price * 0.90):
                pricing_status = "UNDERPRICED"
                monthly_opp = round(price_gap * 15.0, 2)
            elif req.actual_price > (fair_price * 1.10):
                pricing_status = "OVERPRICED"
            else:
                pricing_status = "WITHIN_GUARDRAILS"

        return PriceExplanationResponse(
            predicted_fair_price_per_night=fair_price,
            base_expected_value=round(ev, 2),
            net_shap_adjustment=round(float(shap_vals.sum()), 2),
            recommended_min_guardrail=round(fair_price * 0.85, 2),
            recommended_max_guardrail=round(fair_price * 1.25, 2),
            contributions=contributions,
            actual_price=req.actual_price,
            price_gap=price_gap,
            pricing_status=pricing_status,
            monthly_opportunity_usd=monthly_opp
        )
