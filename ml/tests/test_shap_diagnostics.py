"""Unit tests for SHAP dynamic pricing diagnostics."""

import os
import pytest
import pandas as pd
from ml.data.datasets import load_gold_obt_dataset, temporal_train_test_split
from ml.evaluation.shap_diagnostics import PricingShapDiagnostics

def test_pricing_shap_diagnostics_execution(tmp_path):
    # Load small test partition
    df = load_gold_obt_dataset(limit=100, offline_mode=True)
    _, test_df = temporal_train_test_split(df, time_col="BOOKING_DATE", train_ratio=0.8)

    diagnostics = PricingShapDiagnostics()
    results = diagnostics.run_diagnostics(test_df, output_dir=str(tmp_path / "shap"))

    assert "total_test_samples" in results
    assert results["total_test_samples"] == len(test_df)
    assert "underpriced_count" in results
    assert "avg_underpriced_gap_usd" in results
    assert "global_top_drivers" in results
    assert len(results["global_top_drivers"]) > 0
    assert "case_studies" in results
    assert os.path.exists(results["plots"]["global_importance"])
    assert os.path.exists(results["plots"]["underpriced_drivers"])
