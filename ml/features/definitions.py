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
    )
}
