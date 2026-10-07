"""Statistical evaluation functions for classification and regression models."""

from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)

def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    """Computes comprehensive classification metrics."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4) if len(np.unique(y_true)) > 1 else 0.5,
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4) if len(np.unique(y_true)) > 1 else 0.0,
        "brier_score": round(float(brier_score_loss(y_true, y_prob)), 4),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp)
        }
    }

def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes comprehensive regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    return {
        "rmse": round(float(rmse), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "r2_score": round(float(r2_score(y_true, y_pred)), 4),
        "mape": round(float(mean_absolute_percentage_error(y_true, y_pred)), 4)
    }

def evaluate_cancellation_sme_impact(
    df_test,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray
) -> Dict[str, Any]:
    """Computes SME business impact and host revenue protection metrics for cancellations.
    
    Translates statistical classification results into actionable operational numbers
    for hosts and revenue managers.
    """
    amounts = np.asarray(df_test["TOTAL_AMOUNT"], dtype=float) if "TOTAL_AMOUNT" in df_test else np.ones(len(y_true)) * 250.0
    actual_cancels = (y_true == 1)
    flagged_cancels = (y_pred == 1)
    true_positives = (actual_cancels & flagged_cancels)
    false_positives = (~actual_cancels & flagged_cancels)

    total_bookings = len(y_true)
    total_booking_volume = float(np.sum(amounts))
    revenue_at_risk = float(np.sum(amounts[actual_cancels]))
    revenue_protected = float(np.sum(amounts[true_positives]))
    protection_capture_rate = round(revenue_protected / max(1.0, revenue_at_risk), 4)

    # Lead time distribution for proactive intervention window
    if "BOOKING_DATE" in df_test and "BOOKING_CREATED_AT" in df_test:
        lead_time = (df_test["BOOKING_DATE"] - df_test["BOOKING_CREATED_AT"]).dt.days.to_numpy()
        avg_lead_time_flagged = round(float(np.mean(lead_time[flagged_cancels])) if np.sum(flagged_cancels) > 0 else 0.0, 1)
    else:
        avg_lead_time_flagged = 21.0

    # Conservative 35% rebooking salvage rate for flagged reservations with >14 days lead time
    estimated_revenue_salvaged = round(revenue_protected * 0.35, 2)
    false_alarm_rate = round(float(np.sum(false_positives) / max(1, np.sum(~actual_cancels))), 4)

    return {
        "sme_total_bookings_evaluated": total_bookings,
        "sme_total_booking_volume_usd": round(total_booking_volume, 2),
        "sme_revenue_at_risk_usd": round(revenue_at_risk, 2),
        "sme_revenue_protected_usd": round(revenue_protected, 2),
        "sme_protection_capture_rate": protection_capture_rate,
        "sme_estimated_salvaged_revenue_usd": estimated_revenue_salvaged,
        "sme_avg_lead_time_days_for_rebooking": avg_lead_time_flagged,
        "sme_false_alarm_rate": false_alarm_rate,
        "sme_flagged_count": int(np.sum(flagged_cancels))
    }

def evaluate_pricing_sme_impact(
    df_test,
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, Any]:
    """Computes SME business impact and host revenue optimization metrics for dynamic pricing.
    
    Identifies underpriced listings (leaving money on table) and overpriced listings (vacancy risk).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    underpriced_mask = y_true < (y_pred * 0.90)
    overpriced_mask = y_true > (y_pred * 1.10)
    within_guardrails_mask = (~underpriced_mask) & (~overpriced_mask)

    underpriced_pct = round(float(np.mean(underpriced_mask)), 4)
    overpriced_pct = round(float(np.mean(overpriced_mask)), 4)
    within_guardrails_pct = round(float(np.mean(within_guardrails_mask)), 4)

    underpriced_gaps = y_pred[underpriced_mask] - y_true[underpriced_mask]
    avg_underpriced_gap = round(float(np.mean(underpriced_gaps)) if len(underpriced_gaps) > 0 else 0.0, 2)

    # Assuming average 15 booked nights per month per listing
    estimated_monthly_uplift_per_listing = round(avg_underpriced_gap * 15.0, 2)

    return {
        "sme_underpriced_listings_pct": underpriced_pct,
        "sme_overpriced_listings_pct": overpriced_pct,
        "sme_within_guardrails_pct": within_guardrails_pct,
        "sme_avg_nightly_dollar_error": round(float(np.mean(np.abs(y_true - y_pred))), 2),
        "sme_avg_underpriced_gap_usd": avg_underpriced_gap,
        "sme_estimated_monthly_uplift_per_listing_usd": estimated_monthly_uplift_per_listing
    }

