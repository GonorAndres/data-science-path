"""Overview KPI endpoint."""

from fastapi import APIRouter
from pricing_backend.data_loader import overview_kpis

router = APIRouter(tags=["overview"])


@router.get("/overview")
def get_overview():
    """Return dataset and model KPIs."""
    return overview_kpis or {"error": "Data not loaded"}
