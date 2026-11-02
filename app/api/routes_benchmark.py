"""
Benchmark History and Query API Routes
"""

from typing import Optional
from fastapi import APIRouter, Query
from app.database.db import BenchmarkRepository

router = APIRouter(prefix="/api/benchmark", tags=["Benchmark History"])
repo = BenchmarkRepository()


@router.get("/runs")
def list_runs():
    """Returns all recorded benchmark sweeps."""
    return {"status": "success", "runs": repo.get_runs()}


@router.get("/records")
def list_records(
    model_name: Optional[str] = Query(None),
    condition: Optional[str] = Query(None),
):
    """Returns evaluation records with optional filters."""
    all_records = repo.get_all_records()
    filtered = all_records

    if model_name:
        filtered = [r for r in filtered if r["model_name"] == model_name]
    if condition:
        filtered = [r for r in filtered if r["condition"] == condition]

    return {"status": "success", "count": len(filtered), "records": filtered}
