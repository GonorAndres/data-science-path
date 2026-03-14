"""Model comparison, SHAP, fairness, and evaluation endpoints."""

from fastapi import APIRouter, Query
from pricing_backend.data_loader import (
    model_comparison,
    lift_chart,
    calibration,
    lorenz_curves,
    glm_coefficients,
    shap_summary,
    shap_dependence,
    pure_premium,
    double_lift,
    fairness_audit,
    feature_importance,
    mlflow_summary,
)

router = APIRouter(prefix="/models", tags=["models"])


@router.get("/comparison")
def get_comparison():
    """Return model comparison metrics."""
    return model_comparison or {"error": "Data not loaded"}


@router.get("/lift-chart")
def get_lift_chart():
    """Return lift chart data for all models."""
    return lift_chart or {"error": "Data not loaded"}


@router.get("/calibration")
def get_calibration():
    """Return calibration data for all models."""
    return calibration or {"error": "Data not loaded"}


@router.get("/lorenz-curves")
def get_lorenz_curves():
    """Return Lorenz curve data for all models."""
    return lorenz_curves or {"error": "Data not loaded"}


@router.get("/coefficients")
def get_coefficients():
    """Return GLM coefficient tables."""
    return glm_coefficients or {"error": "Data not loaded"}


@router.get("/shap-summary")
def get_shap_summary():
    """Return SHAP feature importance summary."""
    return shap_summary or {"error": "Data not loaded"}


@router.get("/shap-dependence")
def get_shap_dependence(
    feature: str = Query(..., description="Feature name"),
    model: str = Query("xgb", description="Model prefix: xgb or lgb"),
):
    """Return SHAP dependence data for a specific feature and model."""
    key = f"{model}_{feature}"
    data = shap_dependence.get(key)
    if data is None:
        available = list(shap_dependence.keys())
        return {"error": f"Key '{key}' not found", "available": available}
    return data


@router.get("/pure-premium")
def get_pure_premium():
    """Return pure premium comparison by segment."""
    return pure_premium or {"error": "Data not loaded"}


@router.get("/double-lift")
def get_double_lift():
    """Return double lift analysis (GLM vs GBM divergence)."""
    return double_lift or {"error": "Data not loaded"}


@router.get("/fairness-audit")
def get_fairness_audit():
    """Return fairness and proxy discrimination audit."""
    return fairness_audit or {"error": "Data not loaded"}


@router.get("/feature-importance")
def get_feature_importance():
    """Return GBM native feature importance."""
    return feature_importance or {"error": "Data not loaded"}


@router.get("/mlflow-summary")
def get_mlflow_summary():
    """Return MLflow experiment summary."""
    return mlflow_summary or {"error": "Data not loaded"}
