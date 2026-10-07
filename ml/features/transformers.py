"""Custom scikit-learn compatible transformers for Airbnb feature pipelines."""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

def compute_lead_time_features(
    df: pd.DataFrame,
    booking_date_col: str = "BOOKING_DATE",
    created_at_col: str = "BOOKING_CREATED_AT",
    require_columns: bool = False,
    fill_na_value: float = 0.0,
    max_lead_time_days: int = 730,
) -> pd.DataFrame:
    """Robust calculation of booking lead time and date validation flags.
    
    Handles:
    - Timezone differences (naive vs aware, ISO8601 formats)
    - Mixed date string formats without parsing crashes
    - Sub-day truncation trap: prevents valid same-day reservations from
      evaluating to -1 days due to Python timedelta integer floor division
    - Corrupt or missing dates via safe coercion to NaT
    - Retroactive / invalid booking dates (lead_time_invalid flag)
    - Case-insensitive column name resolution
    """
    df = df.copy()

    # Case-insensitive column lookup if exact match not found
    cols_lookup = {str(c).upper(): c for c in df.columns}
    b_date_key = booking_date_col if booking_date_col in df.columns else cols_lookup.get(booking_date_col.upper())
    b_created_key = created_at_col if created_at_col in df.columns else cols_lookup.get(created_at_col.upper())

    if not b_date_key or not b_created_key:
        if require_columns:
            missing = [c for c, k in [(booking_date_col, b_date_key), (created_at_col, b_created_key)] if not k]
            raise ValueError(f"Missing required booking date column(s): {missing}")
        df["lead_time_days"] = fill_na_value
        df["lead_time_invalid"] = 0
        df["lead_time_missing"] = 1
        return df

    # Safe datetime parsing in UTC to avoid tz-naive vs tz-aware subtraction errors
    b_date = pd.to_datetime(df[b_date_key], format="mixed", errors="coerce", utc=True)
    b_created = pd.to_datetime(df[b_created_key], format="mixed", errors="coerce", utc=True)

    # Normalize to midnight calendar day to eliminate sub-day floor division bugs
    b_date_cal = b_date.dt.tz_localize(None).dt.normalize()
    b_created_cal = b_created.dt.tz_localize(None).dt.normalize()

    delta_days = (b_date_cal - b_created_cal).dt.days

    # Identify missing vs genuinely negative/retroactive dates
    is_missing = delta_days.isna()
    is_negative = delta_days < 0

    df["lead_time_missing"] = is_missing.astype(int)
    df["lead_time_invalid"] = (is_negative.fillna(False)).astype(int)

    # Clip lead time to non-negative range and realistic ceiling; fill NaNs
    clipped = delta_days.clip(lower=0, upper=max_lead_time_days)
    df["lead_time_days"] = clipped.fillna(fill_na_value) if fill_na_value is not None else clipped

    return df


def compute_calendar_seasonality_features(
    df: pd.DataFrame,
    booking_date_col: str = "BOOKING_DATE",
    fill_na: bool = True,
) -> pd.DataFrame:
    """Derives cyclical seasonality and calendar window features from arrival date.
    
    Generates:
    - arrival_month (1-12)
    - arrival_dow (0=Monday, 6=Sunday)
    - is_weekend_arrival (Friday/Saturday check-in binary: 1 or 0)
    - arrival_quarter (1-4)
    - arrival_month_sin, arrival_month_cos (12-month annual cyclical wave)
    - arrival_dow_sin, arrival_dow_cos (7-day weekly cyclical wave)
    - arrival_date_missing indicator flag
    """
    df = df.copy()

    # Case-insensitive column lookup
    cols_lookup = {str(c).upper(): c for c in df.columns}
    date_key = booking_date_col if booking_date_col in df.columns else cols_lookup.get(booking_date_col.upper())

    if not date_key:
        df["arrival_date_missing"] = 1
        for col in [
            "arrival_month", "arrival_dow", "is_weekend_arrival", "arrival_quarter",
            "arrival_month_sin", "arrival_month_cos", "arrival_dow_sin", "arrival_dow_cos"
        ]:
            df[col] = 0.0 if fill_na else np.nan
        return df

    arrival = pd.to_datetime(df[date_key], format="mixed", errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
    is_missing = arrival.isna()

    month = arrival.dt.month
    dow = arrival.dt.weekday
    quarter = arrival.dt.quarter

    df["arrival_date_missing"] = is_missing.astype(int)

    # Raw calendar units (default to median baseline when missing)
    df["arrival_month"] = month.fillna(6).astype(float)
    df["arrival_dow"] = dow.fillna(2).astype(float)
    df["arrival_quarter"] = quarter.fillna(2).astype(float)

    # Friday (4) and Saturday (5) arrivals represent leisure weekend check-ins
    df["is_weekend_arrival"] = dow.isin([4, 5]).astype(int).where(~is_missing, 0)

    # Cyclical Annual Seasonality (Period = 12 months)
    # Mapping Jan(1)->0 and Dec(12)->11/12 ensures smooth circular continuity
    month_rad = 2 * np.pi * (month - 1) / 12.0
    df["arrival_month_sin"] = np.sin(month_rad).fillna(0.0)
    df["arrival_month_cos"] = np.cos(month_rad).fillna(1.0)

    # Cyclical Weekly Cadence (Period = 7 days)
    dow_rad = 2 * np.pi * dow / 7.0
    df["arrival_dow_sin"] = np.sin(dow_rad).fillna(0.0)
    df["arrival_dow_cos"] = np.cos(dow_rad).fillna(1.0)

    return df


class AirbnbFeatureEngineer(BaseEstimator, TransformerMixin):
    """Derives domain-specific business features from raw Gold OBT fields.
    
    Adheres to scikit-learn Transformer API so it embeds directly into Pipeline.
    """

    def __init__(self, require_booking_dates: bool = False, max_lead_time_days: int = 730):
        self.require_booking_dates = require_booking_dates
        self.max_lead_time_days = max_lead_time_days

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        
        # 1. Robust Lead Time & Booking Date Features (backward compatible with unpickled models)
        require_dates = getattr(self, "require_booking_dates", False)
        max_lead_time = getattr(self, "max_lead_time_days", 730)

        df = compute_lead_time_features(
            df,
            booking_date_col="BOOKING_DATE",
            created_at_col="BOOKING_CREATED_AT",
            require_columns=require_dates,
            max_lead_time_days=max_lead_time,
        )

        # 2. Behavioral Lead Time Bins & Non-linear scaling
        df["is_last_minute"] = (df["lead_time_days"] <= 3).astype(int)
        df["is_short_notice"] = ((df["lead_time_days"] > 3) & (df["lead_time_days"] <= 7)).astype(int)
        df["is_far_advance"] = (df["lead_time_days"] >= 45).astype(int)
        df["lead_time_log"] = np.log1p(df["lead_time_days"])

        # 3. Seasonality & Calendar Windows (Cyclical Waves)
        df = compute_calendar_seasonality_features(
            df,
            booking_date_col="BOOKING_DATE"
        )

        # 4. Financial ratios & Fee Discrepancy Signals (Robust & Defensively Typed)
        if "TOTAL_AMOUNT" in df.columns:
            total_raw = pd.to_numeric(df["TOTAL_AMOUNT"], errors="coerce")
            is_missing = total_raw.isna()
            is_non_positive = total_raw <= 0

            df["total_amount_invalid"] = (is_missing | is_non_positive).astype(int)
            # Mask amounts <= 0 to NaN to avoid division by zero or negative ratio inversion
            total_valid = total_raw.where(total_raw > 0)
        else:
            df["total_amount_invalid"] = 1
            total_valid = pd.Series(np.nan, index=df.index)

        # Safe extraction of cleaning fee
        if "CLEANING_FEE" in df.columns:
            clean_fee = pd.to_numeric(df["CLEANING_FEE"], errors="coerce").fillna(0.0).clip(lower=0.0)
        else:
            clean_fee = pd.Series(0.0, index=df.index)

        # Safe extraction of service fee
        if "SERVICE_FEE" in df.columns:
            svc_fee = pd.to_numeric(df["SERVICE_FEE"], errors="coerce").fillna(0.0).clip(lower=0.0)
        else:
            svc_fee = pd.Series(0.0, index=df.index)

        # Calculate ratios bounded strictly between [0.0, 1.0] and fill NaNs
        df["cleaning_fee_ratio"] = (clean_fee / total_valid).clip(lower=0.0, upper=1.0).fillna(0.0)
        df["service_fee_ratio"] = (svc_fee / total_valid).clip(lower=0.0, upper=1.0).fillna(0.0)

        # 5. Capacity & Relative Pricing Signals (Robust & Defensively Typed)
        if "ACCOMMODATES" in df.columns:
            acc_raw = pd.to_numeric(df["ACCOMMODATES"], errors="coerce")
            acc_valid = acc_raw.where(acc_raw > 0)
            df["accommodates_invalid"] = (acc_raw.isna() | (acc_raw <= 0)).astype(int)
        else:
            acc_valid = pd.Series(np.nan, index=df.index)
            df["accommodates_invalid"] = 1

        if "BEDROOMS" in df.columns:
            bed_raw = pd.to_numeric(df["BEDROOMS"], errors="coerce").clip(lower=0.0)
            bed_valid = bed_raw.where(bed_raw > 0)
            df["bedroom_to_accommodates_ratio"] = (bed_raw / acc_valid).clip(lower=0.0, upper=2.0).fillna(0.5)
        else:
            bed_valid = pd.Series(np.nan, index=df.index)
            df["bedroom_to_accommodates_ratio"] = 0.5

        # Fee discrepancy relative to capacity
        df["cleaning_fee_per_bedroom"] = (clean_fee / bed_valid.fillna(1.0)).fillna(0.0)
        df["cleaning_fee_per_accommodate"] = (clean_fee / acc_valid.fillna(1.0)).fillna(0.0)

        # Price per accommodate (clipped at 0, defaulted to 0.0)
        if "PRICE_PER_NIGHT" in df.columns:
            price_raw = pd.to_numeric(df["PRICE_PER_NIGHT"], errors="coerce").clip(lower=0.0)
            df["price_per_accommodate"] = (price_raw / acc_valid).fillna(0.0)
        else:
            df["price_per_accommodate"] = 0.0

        # 6. Host features (Robust & Defensively Typed)
        if "IS_SUPERHOST" in df.columns:
            s = df["IS_SUPERHOST"].astype(str).str.strip().str.lower()
            df["is_superhost_binary"] = s.isin({"true", "t", "1", "yes", "y"}).astype(int)
        else:
            df["is_superhost_binary"] = 0

        if "RESPONSE_RATE" in df.columns:
            # Clean whitespace, strip percent signs, and parse numeric
            rr_str = df["RESPONSE_RATE"].astype(str).str.strip().str.rstrip("%").str.strip()
            rr_numeric = pd.to_numeric(rr_str, errors="coerce").clip(lower=0.0, upper=100.0)
            
            df["response_rate_missing"] = rr_numeric.isna().astype(int)
            # Default to 80.0 median baseline for missing values
            df["host_response_rate"] = rr_numeric.fillna(80.0)
        else:
            df["response_rate_missing"] = 1
            df["host_response_rate"] = 80.0

        return df
