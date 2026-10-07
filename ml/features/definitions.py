"""Feature registry and schema specifications for ML pipelines."""

from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class FeatureMeta:
    name: str
    entity: str
    data_type: str
    description: str
    window_days: Optional[int] = None
    imputation_strategy: str = "median"

FEATURE_REGISTRY: Dict[str, FeatureMeta] = {
    # Booking transaction features
    "lead_time_days": FeatureMeta(
        name="lead_time_days",
        entity="booking",
        data_type="int",
        description="Days between booking creation and check-in date."
    ),
    "lead_time_invalid": FeatureMeta(
        name="lead_time_invalid",
        entity="booking",
        data_type="int",
        description="Binary flag (1 if booking date preceded creation timestamp, indicating invalid/retroactive record)."
    ),
    "lead_time_missing": FeatureMeta(
        name="lead_time_missing",
        entity="booking",
        data_type="int",
        description="Binary flag (1 if either booking date or creation timestamp was missing or unparseable)."
    ),
    "cleaning_fee_ratio": FeatureMeta(
        name="cleaning_fee_ratio",
        entity="booking",
        data_type="float",
        description="Ratio of cleaning fee relative to gross total booking amount."
    ),
    "service_fee_ratio": FeatureMeta(
        name="service_fee_ratio",
        entity="booking",
        data_type="float",
        description="Ratio of platform service fee relative to gross total amount."
    ),

    # Listing supply features
    "price_per_accommodate": FeatureMeta(
        name="price_per_accommodate",
        entity="listing",
        data_type="float",
        description="Nightly price divided by accommodation capacity."
    ),
    "bedroom_to_accommodates_ratio": FeatureMeta(
        name="bedroom_to_accommodates_ratio",
        entity="listing",
        data_type="float",
        description="Bedrooms divided by accommodates."
    ),
    "trailing_30d_listing_bookings": FeatureMeta(
        name="trailing_30d_listing_bookings",
        entity="listing",
        data_type="int",
        description="Sliding 30-day point-in-time count of prior bookings.",
        window_days=30
    ),
    "trailing_30d_listing_cancellations": FeatureMeta(
        name="trailing_30d_listing_cancellations",
        entity="listing",
        data_type="int",
        description="Sliding 30-day point-in-time count of prior cancellations.",
        window_days=30
    ),

    # Host features
    "is_superhost_binary": FeatureMeta(
        name="is_superhost_binary",
        entity="host",
        data_type="int",
        description="Binary indicator (1 for Superhost, 0 otherwise)."
    ),
    "host_response_rate": FeatureMeta(
        name="host_response_rate",
        entity="host",
        data_type="float",
        description="Point-in-time host inquiry response rate (0-100)."
    ),

    # Behavioral lead time bins
    "is_last_minute": FeatureMeta(
        name="is_last_minute",
        entity="booking",
        data_type="int",
        description="Binary flag (1 if booking lead time <= 3 days, indicating last-minute reservation)."
    ),
    "is_short_notice": FeatureMeta(
        name="is_short_notice",
        entity="booking",
        data_type="int",
        description="Binary flag (1 if booking lead time is between 4 and 7 days)."
    ),
    "is_far_advance": FeatureMeta(
        name="is_far_advance",
        entity="booking",
        data_type="int",
        description="Binary flag (1 if booking lead time >= 45 days, carrying higher cancellation risk)."
    ),
    "lead_time_log": FeatureMeta(
        name="lead_time_log",
        entity="booking",
        data_type="float",
        description="Log-transformed lead time log1p(days) for variance stabilization."
    ),

    # Seasonality and calendar features
    "arrival_month": FeatureMeta(
        name="arrival_month",
        entity="booking",
        data_type="float",
        description="Calendar arrival month (1-12)."
    ),
    "arrival_dow": FeatureMeta(
        name="arrival_dow",
        entity="booking",
        data_type="float",
        description="Day of week for arrival (0=Monday, 6=Sunday)."
    ),
    "is_weekend_arrival": FeatureMeta(
        name="is_weekend_arrival",
        entity="booking",
        data_type="int",
        description="Binary flag (1 for Friday/Saturday arrival, representing leisure trip check-in)."
    ),
    "arrival_quarter": FeatureMeta(
        name="arrival_quarter",
        entity="booking",
        data_type="float",
        description="Calendar arrival quarter (1-4)."
    ),
    "arrival_month_sin": FeatureMeta(
        name="arrival_month_sin",
        entity="booking",
        data_type="float",
        description="Annual cyclical seasonal sine projection."
    ),
    "arrival_month_cos": FeatureMeta(
        name="arrival_month_cos",
        entity="booking",
        data_type="float",
        description="Annual cyclical seasonal cosine projection."
    ),
    "arrival_dow_sin": FeatureMeta(
        name="arrival_dow_sin",
        entity="booking",
        data_type="float",
        description="Weekly cyclical day-of-week sine projection."
    ),
    "arrival_dow_cos": FeatureMeta(
        name="arrival_dow_cos",
        entity="booking",
        data_type="float",
        description="Weekly cyclical day-of-week cosine projection."
    ),

    # Relative fee and discrepancy features
    "cleaning_fee_per_bedroom": FeatureMeta(
        name="cleaning_fee_per_bedroom",
        entity="listing",
        data_type="float",
        description="Cleaning fee normalized per bedroom."
    ),
    "cleaning_fee_per_accommodate": FeatureMeta(
        name="cleaning_fee_per_accommodate",
        entity="listing",
        data_type="float",
        description="Cleaning fee normalized per guest accommodate capacity."
    )
}
