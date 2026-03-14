"""
SHAP analysis for tree-based insurance pricing models.
Extracts summary and dependence data as JSON-serializable structures.
"""

import numpy as np
import pandas as pd
import shap


def compute_shap_values(
    model,
    X: pd.DataFrame,
    feature_names: list[str],
    model_type: str = "xgb",
    sample_size: int = 15_000,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Compute TreeSHAP values on a subsample.

    Returns (shap_values array, X_sample DataFrame).
    """
    if len(X) > sample_size:
        X_sample = X[feature_names].sample(n=sample_size, random_state=42)
    else:
        X_sample = X[feature_names].copy()

    if model_type == "xgb":
        import xgboost as xgb
        explainer = shap.TreeExplainer(model)
        dmatrix = xgb.DMatrix(X_sample)
        shap_values = explainer.shap_values(dmatrix)
    elif model_type == "lgb":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return np.asarray(shap_values), X_sample


def shap_summary_data(shap_values: np.ndarray, feature_names: list[str], top_n: int = 15) -> list[dict]:
    """
    Extract SHAP summary bar chart data.

    Returns list of dicts sorted by mean |SHAP| descending:
    [{feature, mean_abs_shap, rank}, ...]
    """
    mean_abs = np.abs(shap_values).mean(axis=0)

    results = []
    for i, fname in enumerate(feature_names):
        results.append({
            "feature": fname,
            "mean_abs_shap": round(float(mean_abs[i]), 6),
        })

    results.sort(key=lambda x: x["mean_abs_shap"], reverse=True)
    for i, r in enumerate(results[:top_n]):
        r["rank"] = i + 1

    return results[:top_n]


def shap_dependence_data(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    feature_name: str,
    feature_names: list[str],
    n_points: int = 2000,
) -> dict:
    """
    Extract SHAP dependence plot data for a single feature.

    Returns dict with: feature, points (list of {x, shap_value}).
    Subsampled to n_points for frontend performance.
    """
    if feature_name not in feature_names:
        raise ValueError(f"Feature '{feature_name}' not in feature_names")

    idx = feature_names.index(feature_name)
    feature_values = X_sample[feature_name].values
    shap_vals = shap_values[:, idx]

    # Subsample if needed
    if len(feature_values) > n_points:
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(len(feature_values), n_points, replace=False)
        feature_values = feature_values[sample_idx]
        shap_vals = shap_vals[sample_idx]

    points = []
    for x, s in zip(feature_values, shap_vals):
        points.append({
            "x": round(float(x), 4),
            "shap_value": round(float(s), 6),
        })

    return {
        "feature": feature_name,
        "n_points": len(points),
        "points": points,
    }


def compute_all_shap_outputs(
    model,
    X: pd.DataFrame,
    feature_names: list[str],
    model_type: str = "xgb",
    dependence_features: list[str] | None = None,
    sample_size: int = 15_000,
) -> dict:
    """
    Compute all SHAP outputs for export.

    Returns dict with:
    - summary: list of feature importance dicts
    - dependence: dict mapping feature -> dependence data
    """
    shap_values, X_sample = compute_shap_values(
        model, X, feature_names, model_type, sample_size,
    )

    summary = shap_summary_data(shap_values, feature_names)

    # Default dependence features: top 6 by importance
    if dependence_features is None:
        dependence_features = [s["feature"] for s in summary[:6]]

    dependence = {}
    for feat in dependence_features:
        if feat in feature_names:
            dependence[feat] = shap_dependence_data(
                shap_values, X_sample, feat, feature_names,
            )

    return {
        "summary": summary,
        "dependence": dependence,
    }
