"""Dual-mode data connector for Snowflake Gold Marts with deterministic offline fallback."""

import os
import logging
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class SnowflakeConnector:
    """Connects to Snowflake with seamless fallback to offline mock data.
    
    Ensures local testing, development, and CI/CD pipelines run without requiring
    active cloud warehouse credentials.
    """

    def __init__(
        self,
        account: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        role: Optional[str] = None,
        warehouse: Optional[str] = None,
        database: Optional[str] = None,
        schema: Optional[str] = None,
        offline_mode: bool = False
    ):
        self.account = account or os.getenv("SNOWFLAKE_ACCOUNT")
        self.user = user or os.getenv("SNOWFLAKE_USER")
        self.password = password or os.getenv("SNOWFLAKE_PASSWORD")
        self.role = role or os.getenv("SNOWFLAKE_ROLE", "AIRBNB_TRANSFORMER_ROLE")
        self.warehouse = warehouse or os.getenv("SNOWFLAKE_WAREHOUSE", "SNOWFLAKE_LEARNING_WH")
        self.database = database or os.getenv("SNOWFLAKE_DATABASE", "AIRBNB")
        self.schema = schema or os.getenv("SNOWFLAKE_SCHEMA", "gold")
        
        # Explicit or environment-driven offline override
        self.offline_mode = offline_mode or (os.getenv("AIRBNB_ML_OFFLINE", "1") == "1")

    def fetch_gold_obt(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Fetches AIRBNB.gold.obt data from Snowflake or offline generator."""
        if not self.offline_mode and self.account and self.user and self.password:
            try:
                import snowflake.connector
                conn = snowflake.connector.connect(
                    user=self.user,
                    password=self.password,
                    account=self.account,
                    warehouse=self.warehouse,
                    database=self.database,
                    schema=self.schema,
                    role=self.role
                )
                query = "SELECT * FROM AIRBNB.gold.obt"
                if limit:
                    query += f" LIMIT {limit}"
                logger.info("Executing query on Snowflake: %s", query)
                df = pd.read_sql(query, conn)
                conn.close()
                df.columns = [c.upper() for c in df.columns]
                return df
            except Exception as e:
                logger.warning("Snowflake query failed (%s). Falling back to deterministic offline fixture.", e)

        return self._generate_synthetic_gold_obt(n_rows=limit or 1200)

    @staticmethod
    def _generate_synthetic_gold_obt(n_rows: int = 1200, seed: int = 42) -> pd.DataFrame:
        """Generates realistic, schema-accurate Airbnb Gold OBT dataset for offline execution."""
        rng = np.random.default_rng(seed)

        cities = ["New York", "Paris", "Tokyo", "London", "Berlin", "San Francisco"]
        property_types = ["Apartment", "Condo", "House"]
        room_types = ["Entire home", "Private room"]
        price_tags = ["LOW", "MEDIUM", "HIGH"]
        response_bands = ["VERY GOOD", "GOOD", "AVERAGE", "POOR"]
        statuses = ["confirmed", "confirmed", "confirmed", "cancelled"]

        dates = pd.date_range(start="2024-01-01", periods=180, freq="D")
        booking_dates = rng.choice(dates, size=n_rows)

        listing_ids = [f"LST_{i:04d}" for i in range(101, 301)]
        host_ids = [f"HST_{i:03d}" for i in range(51, 151)]

        rows = []
        for i in range(1, n_rows + 1):
            b_date = booking_dates[i - 1]
            b_created_at = b_date - pd.Timedelta(days=int(rng.integers(1, 45)))
            
            p_type = rng.choice(property_types)
            r_type = rng.choice(room_types)
            city = rng.choice(cities)
            
            accommodates = int(rng.integers(1, 8))
            bedrooms = max(1, int(round(accommodates / 2.0)))
            bathrooms = float(max(1.0, round(bedrooms * 0.75, 1)))

            # Base nightly price correlated with capacity and city
            city_multiplier = {"New York": 1.4, "Paris": 1.2, "Tokyo": 1.1, "London": 1.3, "Berlin": 0.9, "San Francisco": 1.5}[city]
            base_price = (40.0 + accommodates * 35.0 + (50.0 if r_type == "Entire home" else 0.0)) * city_multiplier
            price_per_night = round(float(rng.normal(base_price, 25.0)), 2)
            price_per_night = max(35.0, price_per_night)

            price_tag = "LOW" if price_per_night < 100 else ("MEDIUM" if price_per_night < 200 else "HIGH")
            nights = int(rng.integers(1, 7))
            cleaning_fee = round(float(rng.uniform(25.0, 120.0)), 2)
            service_fee = round(float((price_per_night * nights + cleaning_fee) * 0.12), 2)
            total_amount = round(float(price_per_night * nights + cleaning_fee + service_fee), 2)

            is_superhost = rng.choice(["TRUE", "FALSE"], p=[0.35, 0.65])
            response_rate = float(rng.uniform(70.0, 100.0) if is_superhost == "TRUE" else rng.uniform(40.0, 95.0))
            response_band = "VERY GOOD" if response_rate >= 90 else ("GOOD" if response_rate >= 80 else ("AVERAGE" if response_rate >= 60 else "POOR"))

            # Cancellation probability higher for expensive listings and longer lead time
            lead_time_days = (b_date - b_created_at).days
            cancel_prob = 0.15 + (0.10 if lead_time_days > 20 else 0.0) + (0.08 if price_tag == "HIGH" else 0.0) - (0.07 if is_superhost == "TRUE" else 0.0)
            status = "cancelled" if rng.uniform() < cancel_prob else "confirmed"
            if status == "cancelled":
                cancel_lead = max(1, lead_time_days)
                cancel_offset = int(rng.integers(1, max(2, cancel_lead)))
                cancelled_at = b_created_at + pd.Timedelta(days=cancel_offset)
            else:
                cancelled_at = pd.NaT

            rows.append({
                "BOOKING_ID": f"BKG_{i:05d}",
                "BOOKING_DATE": b_date,
                "CLEANING_FEE": cleaning_fee,
                "SERVICE_FEE": service_fee,
                "TOTAL_AMOUNT": total_amount,
                "BOOKING_STATUS": status,
                "BOOKING_CREATED_AT": b_created_at,
                "CANCELLED_AT": cancelled_at,
                "LISTING_ID": rng.choice(listing_ids),
                "PROPERTY_TYPE": p_type,
                "ROOM_TYPE": r_type,
                "CITY": city,
                "COUNTRY": "USA" if city in ["New York", "San Francisco"] else "International",
                "ACCOMMODATES": accommodates,
                "BEDROOMS": bedrooms,
                "BATHROOMS": bathrooms,
                "PRICE_PER_NIGHT": price_per_night,
                "PRICE_PER_NIGHT_TAG": price_tag,
                "LISTING_CREATED_AT": b_date - pd.Timedelta(days=int(rng.integers(100, 500))),
                "HOST_ID": rng.choice(host_ids),
                "HOST_NAME": f"Host_{rng.choice(host_ids)}",
                "IS_SUPERHOST": is_superhost,
                "HOST_SINCE": pd.Timestamp("2020-01-01") + pd.Timedelta(days=int(rng.integers(0, 1000))),
                "RESPONSE_RATE": response_rate,
                "RESPONSE_RATE_BAND": response_band,
                "HOST_CREATED_AT": b_date - pd.Timedelta(days=int(rng.integers(200, 600)))
            })

        df = pd.DataFrame(rows)
        return df
