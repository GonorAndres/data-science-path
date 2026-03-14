"""EDA endpoints: distributions, correlations, exposure, claim frequency."""

from fastapi import APIRouter, Query
from pricing_backend.data_loader import (
    eda_distributions,
    eda_correlations,
    eda_exposure,
    eda_claim_frequency,
)

router = APIRouter(prefix="/eda", tags=["eda"])


@router.get("/distributions")
def get_distributions(feature: str | None = Query(None, description="Specific feature")):
    """Return distribution data, optionally filtered by feature."""
    if eda_distributions is None:
        return {"error": "Data not loaded"}
    if feature:
        data = eda_distributions.get(feature)
        if data is None:
            return {"error": f"Feature '{feature}' not found"}
        return {feature: data}
    return eda_distributions


@router.get("/correlations")
def get_correlations():
    """Return correlation matrix."""
    return eda_correlations or {"error": "Data not loaded"}


@router.get("/exposure")
def get_exposure():
    """Return exposure analysis."""
    return eda_exposure or {"error": "Data not loaded"}


@router.get("/claim-frequency")
def get_claim_frequency():
    """Return claim frequency heatmap data."""
    return eda_claim_frequency or {"error": "Data not loaded"}
