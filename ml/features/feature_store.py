"""Point-in-Time Feature Store Engine inspired by Airbnb's Zipline architecture."""

import pandas as pd
import numpy as np
from typing import Optional

class ZiplineFeatureStore:
    """Computes point-in-time correct historical features without data leakage.
    
    Adheres to the core design principles of Airbnb's Zipline:
    1. As-Of Joins: Only observations prior to event timestamp are aggregated.
    2. Zero Target Leakage: Current observation's outcome is excluded from historical windows.
    """

    def __init__(self, window_days: int = 30):
        self.window_days = window_days

    def compute_as_of_listing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes point-in-time trailing booking and cancellation counts per listing."""
        df = df.copy()
        df["BOOKING_CREATED_AT"] = pd.to_datetime(df["BOOKING_CREATED_AT"])
        
        # Sort chronologically to enable sliding as-of window computations
        sorted_df = df.sort_values(by="BOOKING_CREATED_AT").reset_index(drop=True)
        
        n_rows = len(sorted_df)
        trailing_bookings = np.zeros(n_rows, dtype=int)
        trailing_cancellations = np.zeros(n_rows, dtype=int)
        
        if "CANCELLED_AT" not in sorted_df.columns:
            raise ValueError(
                "Missing required column 'CANCELLED_AT'. Point-in-time cancellation history requires "
                "explicit outcome event timestamps to prevent future-informed target leakage. "
                "Falling back to terminal 'BOOKING_STATUS' is strictly prohibited."
            )

        cancelled_missing_ts = (sorted_df["BOOKING_STATUS"] == "cancelled") & (sorted_df["CANCELLED_AT"].isna())
        if cancelled_missing_ts.any():
            raise ValueError(
                f"Detected {cancelled_missing_ts.sum()} cancelled reservations missing explicit 'CANCELLED_AT' event timestamps. "
                "All cancellation outcomes must provide occurrence timestamps for leakage-free point-in-time computation."
            )

        for listing_id, group in sorted_df.groupby("LISTING_ID"):
            idxs = group.index.to_numpy()
            timestamps = group["BOOKING_CREATED_AT"].to_numpy()
            cancel_ts_arr = pd.to_datetime(group["CANCELLED_AT"]).to_numpy()

            for i, curr_idx in enumerate(idxs):
                curr_time = timestamps[i]
                window_start = curr_time - np.timedelta64(self.window_days, 'D')

                # Strictly look back: t < curr_time and t >= window_start
                mask = (timestamps[:i] >= window_start) & (timestamps[:i] < curr_time)
                trailing_bookings[curr_idx] = int(np.sum(mask))

                # Point-in-time outcome availability guarantee:
                # A prior booking's cancellation is ONLY observable at curr_time if the cancellation
                # event actually occurred strictly prior to curr_time (cancel_ts < curr_time)
                # and within the lookback window (cancel_ts >= window_start).
                cancel_mask = (
                    (~pd.isna(cancel_ts_arr[:i])) &
                    (cancel_ts_arr[:i] < curr_time) &
                    (cancel_ts_arr[:i] >= window_start)
                )
                trailing_cancellations[curr_idx] = int(np.sum(cancel_mask))

        sorted_df["trailing_30d_listing_bookings"] = trailing_bookings
        sorted_df["trailing_30d_listing_cancellations"] = trailing_cancellations
        
        # Derived point-in-time cancellation velocity
        sorted_df["trailing_30d_cancellation_rate"] = np.where(
            sorted_df["trailing_30d_listing_bookings"] > 0,
            sorted_df["trailing_30d_listing_cancellations"] / sorted_df["trailing_30d_listing_bookings"],
            0.0
        )
        
        return sorted_df
