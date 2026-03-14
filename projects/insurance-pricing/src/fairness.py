"""
Fairness and proxy discrimination audit for insurance pricing models.

Checks whether geographic features (Area, Density) act as proxies for
socioeconomic class -- directly relevant for Mexico's inequality context.
"""

import numpy as np
import pandas as pd


def proxy_discrimination_audit(
    df: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    group_col: str = "Area",
) -> dict:
    """
    Compare model predictions across demographic/geographic segments.

    For each model, compute average predicted frequency by group_col,
    then compute relative ratio vs the overall mean.

    Returns dict with: group_col, groups (list of dicts with segment metrics).
    """
    groups = []
    overall_actual = float(df["ClaimNb"].mean())

    for group_val in sorted(df[group_col].unique()):
        mask = df[group_col] == group_val
        n_policies = int(mask.sum())
        actual_freq = float(df.loc[mask, "ClaimNb"].mean())

        group_data = {
            "group": str(group_val),
            "n_policies": n_policies,
            "pct_portfolio": round(n_policies / len(df) * 100, 2),
            "actual_frequency": round(actual_freq, 6),
            "actual_ratio_to_mean": round(actual_freq / overall_actual if overall_actual > 0 else 0, 4),
        }

        for model_name, preds in predictions.items():
            avg_pred = float(preds[mask].mean())
            group_data[f"{model_name}_predicted"] = round(avg_pred, 6)
            group_data[f"{model_name}_ratio_to_mean"] = round(
                avg_pred / preds.mean() if preds.mean() > 0 else 0, 4,
            )

        groups.append(group_data)

    return {
        "group_col": group_col,
        "overall_actual_frequency": round(overall_actual, 6),
        "groups": groups,
        "models": list(predictions.keys()),
    }


def double_lift_analysis(
    df: pd.DataFrame,
    glm_preds: np.ndarray,
    gbm_preds: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Identify where GBM diverges most from GLM -- the 'double lift'.

    Bins by GLM prediction, then compares GBM-vs-GLM ratio within each bin.
    High divergence bins indicate where GLM fails to capture patterns that GBM finds.

    Returns dict with: bins (list of dicts).
    """
    temp = pd.DataFrame({
        "actual": df["ClaimNb"].values,
        "glm_pred": glm_preds,
        "gbm_pred": gbm_preds,
        "exposure": df["Exposure"].values,
    })

    # Bin by GLM prediction
    try:
        temp["glm_bin"] = pd.qcut(temp["glm_pred"], n_bins, labels=False, duplicates="drop")
    except ValueError:
        temp["glm_bin"] = pd.cut(temp["glm_pred"], n_bins, labels=False)

    bins = []
    for b in sorted(temp["glm_bin"].unique()):
        mask = temp["glm_bin"] == b
        subset = temp[mask]

        avg_actual = float(np.average(subset["actual"], weights=subset["exposure"]))
        avg_glm = float(np.average(subset["glm_pred"], weights=subset["exposure"]))
        avg_gbm = float(np.average(subset["gbm_pred"], weights=subset["exposure"]))

        # Divergence: how much does GBM disagree with GLM?
        divergence = avg_gbm / avg_glm if avg_glm > 0 else 1.0

        bins.append({
            "bin": int(b + 1),
            "n_policies": int(mask.sum()),
            "avg_actual": round(avg_actual, 6),
            "avg_glm": round(avg_glm, 6),
            "avg_gbm": round(avg_gbm, 6),
            "gbm_to_glm_ratio": round(divergence, 4),
            "glm_error": round(abs(avg_actual - avg_glm) / avg_actual if avg_actual > 0 else 0, 4),
            "gbm_error": round(abs(avg_actual - avg_gbm) / avg_actual if avg_actual > 0 else 0, 4),
        })

    return {
        "n_bins": len(bins),
        "bins": bins,
        "summary": {
            "max_divergence_bin": max(bins, key=lambda x: abs(x["gbm_to_glm_ratio"] - 1))["bin"],
            "avg_glm_error": round(float(np.mean([b["glm_error"] for b in bins])), 4),
            "avg_gbm_error": round(float(np.mean([b["gbm_error"] for b in bins])), 4),
        },
    }


def density_impact_analysis(
    df: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    n_quantiles: int = 5,
) -> dict:
    """
    Analyze how Density (population density) affects predictions.

    Relevant for Mexico: urban vs rural pricing disparities.
    """
    temp = df[["Density", "ClaimNb", "Exposure"]].copy()
    for name, preds in predictions.items():
        temp[f"{name}_pred"] = preds

    # Create density quantile bins
    try:
        temp["density_q"] = pd.qcut(temp["Density"], n_quantiles, labels=False, duplicates="drop")
    except ValueError:
        temp["density_q"] = pd.cut(temp["Density"], n_quantiles, labels=False)

    quantiles = []
    for q in sorted(temp["density_q"].unique()):
        mask = temp["density_q"] == q
        subset = temp[mask]

        q_data = {
            "quantile": int(q + 1),
            "density_min": round(float(subset["Density"].min()), 1),
            "density_max": round(float(subset["Density"].max()), 1),
            "density_median": round(float(subset["Density"].median()), 1),
            "n_policies": int(mask.sum()),
            "actual_frequency": round(float(np.average(subset["ClaimNb"], weights=subset["Exposure"])), 6),
        }

        for name in predictions:
            q_data[f"{name}_predicted"] = round(
                float(np.average(subset[f"{name}_pred"], weights=subset["Exposure"])), 6,
            )

        quantiles.append(q_data)

    return {
        "n_quantiles": len(quantiles),
        "quantiles": quantiles,
        "models": list(predictions.keys()),
    }
