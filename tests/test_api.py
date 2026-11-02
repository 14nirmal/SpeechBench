"""
Tests for FastAPI Endpoints and Route Handlers
Validates API contract, status codes, query filters, and health responses.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.main import root, healthcheck
from app.api.routes_benchmark import list_runs, list_records


def test_root_endpoint():
    data = root()
    assert data["service"] == "SpeechBench"
    assert data["version"] == "1.0.0"
    assert data["docs"] == "/docs"


def test_healthcheck():
    data = healthcheck()
    assert data["status"] == "healthy"


def test_benchmark_runs_endpoint():
    data = list_runs()
    assert data["status"] == "success"
    assert "runs" in data
    assert isinstance(data["runs"], list)


def test_benchmark_records_endpoint():
    data = list_records()
    assert data["status"] == "success"
    assert "count" in data
    assert "records" in data
    assert data["count"] == len(data["records"])


def test_benchmark_records_filtered():
    # Filter by model_name
    data_tiny = list_records(model_name="tiny")
    assert data_tiny["status"] == "success"
    for r in data_tiny["records"]:
        assert r["model_name"] == "tiny"

    # Filter by condition
    data_clean = list_records(condition="clean")
    assert data_clean["status"] == "success"
    for r in data_clean["records"]:
        assert r["condition"] == "clean"


if __name__ == "__main__":
    print("Testing root...")
    test_root_endpoint()
    print("Testing healthcheck...")
    test_healthcheck()
    print("Testing runs...")
    test_benchmark_runs_endpoint()
    print("Testing records...")
    test_benchmark_records_endpoint()
    print("Testing filtered...")
    test_benchmark_records_filtered()
    print("All API handler tests passed successfully!")

