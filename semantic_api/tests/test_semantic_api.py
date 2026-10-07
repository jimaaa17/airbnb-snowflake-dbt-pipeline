"""Unit and integration tests for the FastAPI Semantic Metric Layer & SQL Compiler."""

import pytest
import math
from fastapi import HTTPException
from fastapi.testclient import TestClient

from semantic_api.main import (
    app,
    load_canonical_registry,
    compile_semantic_sql,
    MetricQueryRequest,
    GOVERNED_METRICS,
    ALLOWED_DIMENSIONS,
    CANONICAL_SEMANTIC_YAML
)

client = TestClient(app)


def test_canonical_metricflow_yaml_loading():
    """Verify metrics and measures are dynamically loaded from MetricFlow YAML."""
    metrics, dims, entities, source_path = load_canonical_registry()
    
    assert len(metrics) >= 5
    assert "total_revenue" in metrics
    assert "total_bookings" in metrics
    assert "booking_confirmation_rate" in metrics
    assert "cancellation_rate" in metrics
    assert "average_booking_value" in metrics

    # Simple metrics
    assert "SUM(" in metrics["total_revenue"].formula_expression
    assert "COUNT(" in metrics["total_bookings"].formula_expression

    # Ratio metrics with display_format: percent
    assert "* 100.0" in metrics["booking_confirmation_rate"].formula_expression
    assert "* 100.0" in metrics["cancellation_rate"].formula_expression

    # Ratio metrics with display_format: currency (no 100x scaling)
    assert "* 100.0" not in metrics["average_booking_value"].formula_expression
    assert "NULLIF" in metrics["average_booking_value"].formula_expression


def test_missing_yaml_raises_runtime_error(monkeypatch):
    """Verify the semantic layer fails fast without silent fallback drift if YAML is missing."""
    import semantic_api.main as sm_main
    monkeypatch.setattr(sm_main, "CANONICAL_SEMANTIC_YAML", "/nonexistent/path/semantic_models.yml")
    with pytest.raises(RuntimeError, match="not found"):
        sm_main.load_canonical_registry()


def test_true_sql_parameter_binding():
    """Verify that filter values are bound as parameters rather than string-interpolated."""
    req = MetricQueryRequest(
        metrics=["total_revenue", "booking_confirmation_rate"],
        dimensions=["city"],
        filters={"country": "USA", "price_per_night_tag": "HIGH"},
        limit=25
    )
    sql, params = compile_semantic_sql(req)

    # Assert SQL contains named parameter placeholders
    assert "%(filter_country_1)s" in sql
    assert "%(filter_price_per_night_tag_2)s" in sql
    assert "%(limit_val)s" in sql

    # Assert params dict contains the actual bound values
    assert params["filter_country_1"] == "USA"
    assert params["filter_price_per_night_tag_2"] == "HIGH"
    assert params["limit_val"] == 25

    # Ensure raw filter values are NOT interpolated directly in the SQL text
    assert "'USA'" not in sql
    assert "'HIGH'" not in sql


def test_list_filter_parameter_binding():
    """Verify multi-select filter lists bind individual parameter placeholders."""
    req = MetricQueryRequest(
        metrics=["total_bookings"],
        dimensions=["property_type"],
        filters={"city": ["Paris", "Tokyo", "London"]},
        limit=10
    )
    sql, params = compile_semantic_sql(req)

    assert "obt.CITY IN (%(filter_city_1)s, %(filter_city_2)s, %(filter_city_3)s)" in sql
    assert params["filter_city_1"] == "Paris"
    assert params["filter_city_2"] == "Tokyo"
    assert params["filter_city_3"] == "London"
    assert params["limit_val"] == 10


def test_hostile_sql_injection_defense():
    """Verify hostile SQL injection strings are safely bound into params and never executed as code."""
    hostile_input = "USA'; DROP TABLE AIRBNB.gold.obt; --"
    req = MetricQueryRequest(
        metrics=["total_revenue"],
        filters={"country": hostile_input}
    )
    sql, params = compile_semantic_sql(req)

    # SQL structure remains completely intact with pyformat placeholder
    assert "WHERE obt.COUNTRY = %(filter_country_1)s" in sql
    assert "DROP TABLE" not in sql

    # Value is safely stored in the bound parameters map
    assert params["filter_country_1"] == hostile_input


def test_reject_disallowed_filter_dimension():
    """Verify unapproved dimension names are rejected with HTTP 400."""
    req = MetricQueryRequest(
        metrics=["total_revenue"],
        filters={"unapproved_column; DROP TABLE;": "test"}
    )
    with pytest.raises(HTTPException) as exc_info:
        compile_semantic_sql(req)
    assert exc_info.value.status_code == 400
    assert "not a permitted dimension" in exc_info.value.detail


def test_reject_non_finite_numeric_filter():
    """Verify NaN and Inf numeric filters are rejected with HTTP 400."""
    req_nan = MetricQueryRequest(
        metrics=["total_revenue"],
        filters={"price_per_night": float("nan")}
    )
    with pytest.raises(HTTPException) as exc_info:
        compile_semantic_sql(req_nan)
    assert exc_info.value.status_code == 400
    assert "Non-finite numeric values" in exc_info.value.detail


def test_api_catalog_endpoint():
    """Verify the /api/v1/catalog endpoint returns governed metrics with metadata."""
    response = client.get("/api/v1/catalog")
    assert response.status_code == 200
    data = response.json()
    assert data["total_metrics"] >= 5
    metric_names = [m["name"] for m in data["metrics"]]
    assert "total_revenue" in metric_names
    assert "booking_confirmation_rate" in metric_names


def test_api_query_metrics_endpoint():
    """Verify the /api/v1/metrics/query endpoint returns compiled SQL and bound params."""
    payload = {
        "metrics": ["total_revenue", "booking_confirmation_rate"],
        "dimensions": ["city"],
        "filters": {"country": "USA"},
        "limit": 50
    }
    response = client.post("/api/v1/metrics/query", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "compiled_sql" in body
    assert "params" in body
    assert body["params"]["filter_country_1"] == "USA"
    assert "%(filter_country_1)s" in body["compiled_sql"]
    assert body["row_count"] > 0
    assert len(body["data"]) > 0
