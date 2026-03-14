"""
Poisson and Gamma GLMs for insurance frequency-severity pricing.
Uses statsmodels with log link and Exposure offset.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm


def train_poisson_glm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    exposure_train: pd.Series,
    feature_names: list[str],
) -> sm.GLM:
    """
    Train Poisson GLM for claim frequency.

    Uses log(Exposure) as offset so the model predicts claim rate per unit exposure.
    """
    X = X_train[feature_names].astype(float)
    X = sm.add_constant(X)

    model = sm.GLM(
        y_train,
        X,
        family=sm.families.Poisson(link=sm.families.links.Log()),
        offset=np.log(exposure_train.clip(lower=1e-9)),
    )

    result = model.fit(maxiter=100, method="IRLS")
    print(f"Poisson GLM converged: {result.converged}")
    print(f"  AIC: {result.aic:.1f}")
    print(f"  BIC: {result.bic:.1f}")
    print(f"  Log-likelihood: {result.llf:.1f}")

    return result


def train_gamma_glm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_names: list[str],
) -> sm.GLM:
    """
    Train Gamma GLM for claim severity (conditional on claim > 0).

    No offset needed -- severity is not exposure-dependent.
    """
    # Filter to positive claims only
    mask = y_train > 0
    X = X_train.loc[mask, feature_names].astype(float)
    X = sm.add_constant(X)
    y = y_train[mask]

    model = sm.GLM(
        y,
        X,
        family=sm.families.Gamma(link=sm.families.links.Log()),
    )

    result = model.fit(maxiter=100, method="IRLS")
    print(f"Gamma GLM converged: {result.converged}")
    print(f"  AIC: {result.aic:.1f}")

    return result


def predict_glm(
    model_result,
    X: pd.DataFrame,
    feature_names: list[str],
    exposure: pd.Series | None = None,
) -> np.ndarray:
    """
    Generate predictions from a fitted GLM.

    For Poisson: predictions are claim rates; multiply by exposure to get expected counts.
    For Gamma: predictions are expected severity (no exposure adjustment).
    """
    X_input = X[feature_names].astype(float)
    X_input = sm.add_constant(X_input)

    if exposure is not None:
        # Poisson: predict with offset
        offset = np.log(exposure.clip(lower=1e-9))
        pred = model_result.predict(X_input, offset=offset)
    else:
        pred = model_result.predict(X_input)

    return np.asarray(pred)


def extract_coefficients(model_result, top_n: int = 30) -> list[dict]:
    """
    Extract GLM coefficient table as JSON-serializable list.

    Returns list of dicts with: feature, coefficient, std_error, z_value, p_value, significant.
    Sorted by absolute coefficient value descending.
    """
    summary = pd.DataFrame({
        "feature": model_result.params.index,
        "coefficient": model_result.params.values,
        "std_error": model_result.bse.values,
        "z_value": model_result.tvalues.values,
        "p_value": model_result.pvalues.values,
    })

    # Add significance flag
    summary["significant"] = summary["p_value"] < 0.05

    # Add relativity (exp(coefficient) for log-link GLMs)
    summary["relativity"] = np.exp(summary["coefficient"])

    # Sort by absolute coefficient
    summary["abs_coef"] = summary["coefficient"].abs()
    summary = summary.sort_values("abs_coef", ascending=False).head(top_n)
    summary = summary.drop(columns=["abs_coef"])

    # Round for JSON
    for col in ["coefficient", "std_error", "z_value", "p_value", "relativity"]:
        summary[col] = summary[col].round(6)

    return summary.to_dict(orient="records")


def glm_summary_stats(model_result) -> dict:
    """Extract summary statistics from a GLM fit."""
    return {
        "aic": round(float(model_result.aic), 2),
        "bic": round(float(model_result.bic), 2),
        "log_likelihood": round(float(model_result.llf), 2),
        "deviance": round(float(model_result.deviance), 2),
        "pearson_chi2": round(float(model_result.pearson_chi2), 2),
        "df_model": int(model_result.df_model),
        "df_residuals": int(model_result.df_resid),
        "n_observations": int(model_result.nobs),
        "converged": bool(model_result.converged),
    }
