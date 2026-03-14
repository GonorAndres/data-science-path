"""
Orchestrator: load data -> features -> train GLMs -> tune GBMs -> evaluate -> SHAP -> fairness -> export JSON.

Outputs ~15 JSON files to backend/data/ for the FastAPI server.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add paths for imports (directory has hyphen, can't use as Python module)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))

from scripts.utils.data_loader import load_merged, split_data
from scripts.utils.evaluation import (
    model_comparison_table,
    lorenz_curve,
    lift_chart,
    calibration_data,
)

from feature_engineering import (
    create_features,
    get_glm_feature_names,
    get_gbm_feature_names,
)
from glm_models import (
    train_poisson_glm,
    train_gamma_glm,
    predict_glm,
    extract_coefficients,
    glm_summary_stats,
)
from gbm_models import (
    train_xgb_frequency,
    train_lgb_frequency,
    train_xgb_severity,
    predict_xgb,
    predict_lgb,
    get_feature_importance,
)
from shap_analysis import compute_all_shap_outputs
from fairness import (
    proxy_discrimination_audit,
    double_lift_analysis,
    density_impact_analysis,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "backend" / "data"


def save_json(data: dict | list, filename: str) -> None:
    """Save data as JSON to the output directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filepath = OUTPUT_DIR / filename
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved {filepath.name} ({filepath.stat().st_size / 1024:.1f} KB)")


def run_pipeline(n_optuna_trials: int = 100) -> None:
    """Run the full training and evaluation pipeline."""
    start = time.time()

    # ===== 1. LOAD DATA =====
    print("=" * 60)
    print("STEP 1: Loading freMTPL2 dataset")
    print("=" * 60)

    merged = load_merged()
    train, val, test = split_data(merged)

    # ===== 2. FEATURE ENGINEERING =====
    print("\n" + "=" * 60)
    print("STEP 2: Feature engineering")
    print("=" * 60)

    train_glm = create_features(train, for_glm=True)
    val_glm = create_features(val, for_glm=True)
    test_glm = create_features(test, for_glm=True)

    train_gbm = create_features(train, for_glm=False)
    val_gbm = create_features(val, for_glm=False)
    test_gbm = create_features(test, for_glm=False)

    glm_features = get_glm_feature_names(train_glm)
    gbm_features = get_gbm_feature_names(train_gbm)

    print(f"GLM features: {len(glm_features)}")
    print(f"GBM features: {len(gbm_features)}")

    # ===== 3. TRAIN GLMs =====
    print("\n" + "=" * 60)
    print("STEP 3: Training GLMs")
    print("=" * 60)

    # Poisson GLM (frequency)
    print("\n--- Poisson GLM (frequency) ---")
    poisson_glm = train_poisson_glm(
        train_glm, train_glm["ClaimNb"], train_glm["Exposure"], glm_features,
    )

    # Gamma GLM (severity)
    print("\n--- Gamma GLM (severity) ---")
    gamma_glm = train_gamma_glm(
        train_glm, train_glm["AvgClaimAmount"], glm_features,
    )

    # GLM predictions on test set
    glm_freq_pred = predict_glm(poisson_glm, test_glm, glm_features, test_glm["Exposure"])
    glm_sev_pred = predict_glm(gamma_glm, test_glm, glm_features)

    # ===== 4. TRAIN GBMs =====
    print("\n" + "=" * 60)
    print("STEP 4: Training GBMs with Optuna")
    print("=" * 60)

    # XGBoost frequency
    print("\n--- XGBoost (frequency) ---")
    xgb_freq, xgb_freq_params = train_xgb_frequency(
        train_gbm, train_gbm["ClaimNb"], train_gbm["Exposure"],
        val_gbm, val_gbm["ClaimNb"], val_gbm["Exposure"],
        gbm_features, n_trials=n_optuna_trials,
    )

    # LightGBM frequency
    print("\n--- LightGBM (frequency) ---")
    lgb_freq, lgb_freq_params = train_lgb_frequency(
        train_gbm, train_gbm["ClaimNb"], train_gbm["Exposure"],
        val_gbm, val_gbm["ClaimNb"], val_gbm["Exposure"],
        gbm_features, n_trials=n_optuna_trials,
    )

    # XGBoost severity
    print("\n--- XGBoost (severity) ---")
    xgb_sev, xgb_sev_params = train_xgb_severity(
        train_gbm, train_gbm["AvgClaimAmount"],
        val_gbm, val_gbm["AvgClaimAmount"],
        gbm_features, n_trials=n_optuna_trials // 2,
    )

    # GBM predictions on test set
    xgb_freq_pred = predict_xgb(xgb_freq, test_gbm, gbm_features)
    lgb_freq_pred = predict_lgb(lgb_freq, test_gbm, gbm_features)
    xgb_sev_pred = predict_xgb(xgb_sev, test_gbm, gbm_features)

    # ===== 5. EVALUATE ALL MODELS =====
    print("\n" + "=" * 60)
    print("STEP 5: Evaluating models")
    print("=" * 60)

    y_test = test["ClaimNb"].values
    w_test = test["Exposure"].values

    # Frequency model comparison
    freq_predictions = {
        "Poisson GLM": glm_freq_pred,
        "XGBoost": xgb_freq_pred,
        "LightGBM": lgb_freq_pred,
    }

    freq_comparison = model_comparison_table(y_test, freq_predictions, w_test, family="poisson")
    print("\nFrequency model comparison:")
    for row in freq_comparison:
        print(f"  {row['model']}: Gini={row['gini']:.4f}, "
              f"Deviance={row.get('poisson_deviance', 'N/A')}")

    # Lift charts
    freq_lift = {}
    for name, pred in freq_predictions.items():
        freq_lift[name] = lift_chart(y_test, pred, w_test)

    # Calibration
    freq_calibration = {}
    for name, pred in freq_predictions.items():
        freq_calibration[name] = calibration_data(y_test, pred, w_test)

    # Lorenz curves
    freq_lorenz = {}
    for name, pred in freq_predictions.items():
        freq_lorenz[name] = lorenz_curve(y_test, pred, w_test)

    # ===== 6. SHAP ANALYSIS =====
    print("\n" + "=" * 60)
    print("STEP 6: SHAP analysis")
    print("=" * 60)

    xgb_shap = compute_all_shap_outputs(
        xgb_freq, test_gbm, gbm_features, model_type="xgb",
    )
    lgb_shap = compute_all_shap_outputs(
        lgb_freq, test_gbm, gbm_features, model_type="lgb",
    )

    # ===== 7. FAIRNESS AUDIT =====
    print("\n" + "=" * 60)
    print("STEP 7: Fairness and proxy discrimination audit")
    print("=" * 60)

    # Need Area column from original test set
    test_with_area = test.copy()
    test_with_area["Area"] = test["Area"].astype(str)

    area_audit = proxy_discrimination_audit(
        test_with_area, freq_predictions, group_col="Area",
    )

    # Double lift: where does GBM diverge from GLM?
    double_lift = double_lift_analysis(test, glm_freq_pred, xgb_freq_pred)

    # Density impact
    density_audit = density_impact_analysis(test, freq_predictions)

    # ===== 8. EXPORT JSON FILES =====
    print("\n" + "=" * 60)
    print("STEP 8: Exporting JSON files")
    print("=" * 60)

    # EDA data
    _export_eda_data(merged, test)

    # Overview KPIs
    save_json({
        "dataset": {
            "n_policies": int(len(merged)),
            "n_claims": int(merged["HasClaim"].sum()),
            "claim_rate": round(float(merged["HasClaim"].mean()), 4),
            "avg_claim_amount": round(float(merged.loc[merged["TotalClaimAmount"] > 0, "TotalClaimAmount"].mean()), 2),
            "total_exposure": round(float(merged["Exposure"].sum()), 2),
        },
        "models": {
            "n_frequency_models": 3,
            "n_severity_models": 2,
            "best_frequency_model": freq_comparison[0]["model"],
            "best_frequency_gini": freq_comparison[0]["gini"],
        },
        "split": {
            "train_size": int(len(train)),
            "val_size": int(len(val)),
            "test_size": int(len(test)),
        },
    }, "overview_kpis.json")

    # Model comparison
    save_json({
        "frequency": freq_comparison,
        "glm_summary": glm_summary_stats(poisson_glm),
    }, "model_comparison.json")

    # Lift charts
    save_json(freq_lift, "lift_chart.json")

    # Calibration
    save_json(freq_calibration, "calibration.json")

    # Lorenz curves
    save_json(freq_lorenz, "lorenz_curves.json")

    # GLM coefficients
    save_json({
        "frequency": extract_coefficients(poisson_glm, top_n=40),
        "severity": extract_coefficients(gamma_glm, top_n=40),
    }, "glm_coefficients.json")

    # SHAP
    save_json({
        "xgboost": xgb_shap["summary"],
        "lightgbm": lgb_shap["summary"],
    }, "shap_summary.json")

    # SHAP dependence (separate files per feature)
    shap_dep_dir = OUTPUT_DIR / "shap_dependence"
    shap_dep_dir.mkdir(parents=True, exist_ok=True)
    for feat, data in xgb_shap["dependence"].items():
        with open(shap_dep_dir / f"xgb_{feat}.json", "w") as f:
            json.dump(data, f, indent=2)
    for feat, data in lgb_shap["dependence"].items():
        with open(shap_dep_dir / f"lgb_{feat}.json", "w") as f:
            json.dump(data, f, indent=2)
    print(f"  Saved {len(xgb_shap['dependence']) + len(lgb_shap['dependence'])} SHAP dependence files")

    # Pure premium comparison
    _export_pure_premium(test, glm_freq_pred, glm_sev_pred, xgb_freq_pred, xgb_sev_pred)

    # Double lift
    save_json(double_lift, "double_lift.json")

    # Fairness audit
    save_json({
        "area_audit": area_audit,
        "density_impact": density_audit,
    }, "fairness_audit.json")

    # Feature importance (GBM native)
    save_json({
        "xgboost": get_feature_importance(xgb_freq, gbm_features, "xgb"),
        "lightgbm": get_feature_importance(lgb_freq, gbm_features, "lgb"),
    }, "feature_importance.json")

    # MLflow summary
    save_json({
        "xgb_freq_params": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in xgb_freq_params.items()},
        "lgb_freq_params": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in lgb_freq_params.items()},
        "xgb_sev_params": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in xgb_sev_params.items()},
        "n_optuna_trials": n_optuna_trials,
    }, "mlflow_summary.json")

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"Pipeline complete in {elapsed / 60:.1f} minutes")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


def _export_eda_data(merged: pd.DataFrame, test: pd.DataFrame) -> None:
    """Export EDA data for the dashboard."""
    # Distributions
    distributions = {}
    for col in ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]:
        hist, edges = np.histogram(merged[col].dropna(), bins=30)
        distributions[col] = {
            "type": "numeric",
            "bins": [round(float(e), 2) for e in edges[:-1]],
            "counts": [int(c) for c in hist],
            "mean": round(float(merged[col].mean()), 2),
            "median": round(float(merged[col].median()), 2),
            "std": round(float(merged[col].std()), 2),
        }

    for col in ["VehBrand", "VehGas", "Area", "Region"]:
        vc = merged[col].value_counts().head(15)
        distributions[col] = {
            "type": "categorical",
            "categories": [str(c) for c in vc.index],
            "counts": [int(c) for c in vc.values],
        }

    save_json(distributions, "eda_distributions.json")

    # Exposure analysis
    exposure_by_area = merged.groupby("Area").agg(
        total_exposure=("Exposure", "sum"),
        n_policies=("IDpol", "count"),
        avg_exposure=("Exposure", "mean"),
        claim_rate=("HasClaim", "mean"),
    ).reset_index()

    save_json({
        "by_area": exposure_by_area.round(4).to_dict(orient="records"),
        "exposure_distribution": {
            "mean": round(float(merged["Exposure"].mean()), 4),
            "median": round(float(merged["Exposure"].median()), 4),
            "std": round(float(merged["Exposure"].std()), 4),
        },
    }, "eda_exposure.json")

    # Claim frequency heatmap (VehAge x DrivAge)
    freq_pivot = merged.groupby(
        [pd.cut(merged["VehAge"], bins=range(0, 35, 5)),
         pd.cut(merged["DrivAge"], bins=range(18, 100, 10))],
    )["ClaimNb"].mean()

    heatmap_data = []
    for (veh_age, driv_age), freq in freq_pivot.items():
        heatmap_data.append({
            "veh_age": str(veh_age),
            "driv_age": str(driv_age),
            "frequency": round(float(freq), 6),
        })

    save_json(heatmap_data, "eda_claim_frequency.json")

    # Correlations
    numeric_cols = ["ClaimNb", "Exposure", "VehPower", "VehAge", "DrivAge",
                    "BonusMalus", "Density", "TotalClaimAmount"]
    corr = merged[numeric_cols].corr().round(4)
    save_json({
        "columns": list(corr.columns),
        "matrix": corr.values.tolist(),
    }, "eda_correlations.json")


def _export_pure_premium(
    test: pd.DataFrame,
    glm_freq: np.ndarray,
    glm_sev: np.ndarray,
    xgb_freq: np.ndarray,
    xgb_sev: np.ndarray,
) -> None:
    """Export pure premium comparison by segment."""
    glm_pp = glm_freq * glm_sev
    xgb_pp = xgb_freq * xgb_sev

    segments = []
    for area in sorted(test["Area"].astype(str).unique()):
        mask = test["Area"].astype(str) == area
        segments.append({
            "segment": f"Area {area}",
            "n_policies": int(mask.sum()),
            "actual_pp": round(float(test.loc[mask, "PurePremium"].mean()), 2),
            "glm_pp": round(float(glm_pp[mask].mean()), 2),
            "xgb_pp": round(float(xgb_pp[mask].mean()), 2),
        })

    save_json({"segments": segments}, "pure_premium_comparison.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train insurance pricing models")
    parser.add_argument("--trials", type=int, default=100, help="Optuna trials per model")
    args = parser.parse_args()

    run_pipeline(n_optuna_trials=args.trials)
