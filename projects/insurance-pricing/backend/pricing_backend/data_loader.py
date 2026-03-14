"""
Load all pre-computed JSON files at import time.
Module-level dicts are shared across all request handlers.
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _load(filename: str) -> dict | list | None:
    """Load a JSON file from the data directory, return None if missing."""
    filepath = DATA_DIR / filename
    if filepath.exists():
        with open(filepath) as f:
            return json.load(f)
    print(f"Warning: {filepath} not found")
    return None


# Load all datasets at module level
overview_kpis = _load("overview_kpis.json")
eda_distributions = _load("eda_distributions.json")
eda_correlations = _load("eda_correlations.json")
eda_exposure = _load("eda_exposure.json")
eda_claim_frequency = _load("eda_claim_frequency.json")
model_comparison = _load("model_comparison.json")
lift_chart = _load("lift_chart.json")
calibration = _load("calibration.json")
lorenz_curves = _load("lorenz_curves.json")
glm_coefficients = _load("glm_coefficients.json")
shap_summary = _load("shap_summary.json")
pure_premium = _load("pure_premium_comparison.json")
double_lift = _load("double_lift.json")
fairness_audit = _load("fairness_audit.json")
feature_importance = _load("feature_importance.json")
mlflow_summary = _load("mlflow_summary.json")


def _load_shap_dependence() -> dict:
    """Load all SHAP dependence files from subdirectory."""
    dep_dir = DATA_DIR / "shap_dependence"
    result = {}
    if dep_dir.exists():
        for f in dep_dir.glob("*.json"):
            with open(f) as fh:
                result[f.stem] = json.load(fh)
    return result


shap_dependence = _load_shap_dependence()
