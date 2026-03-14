"""
Feature engineering for freMTPL2 insurance pricing models.
Handles binning, transformations, and encoding for both GLM and GBM pipelines.
"""

import numpy as np
import pandas as pd


# --- Binning definitions ---

DRIVAGE_BINS = [17, 22, 26, 32, 42, 52, 62, 72, 100]
DRIVAGE_LABELS = ["18-22", "23-26", "27-32", "33-42", "43-52", "53-62", "63-72", "73+"]

VEHAGE_BINS = [-1, 0, 1, 3, 5, 10, 15, 100]
VEHAGE_LABELS = ["0", "1", "2-3", "4-5", "6-10", "11-15", "16+"]

BONUSMALUS_BINS = [49, 55, 60, 68, 76, 90, 100, 120, 150, 230]
BONUSMALUS_LABELS = ["50-55", "56-60", "61-68", "69-76", "77-90", "91-100",
                     "101-120", "121-150", "151+"]

# Columns used by all models
FEATURE_COLS_BASE = [
    "VehPower", "VehAge", "DrivAge", "BonusMalus",
    "VehBrand", "VehGas", "Area", "Density", "Region",
]


def _bin_column(series: pd.Series, bins: list, labels: list) -> pd.Series:
    """Bin a numeric column into categories."""
    return pd.cut(series, bins=bins, labels=labels, right=True).astype(str)


def _log_density(df: pd.DataFrame) -> pd.Series:
    """Log-transform Density (right-skewed, spans 1 to 27,000+)."""
    return np.log1p(df["Density"].clip(lower=0))


def _cap_vehpower(series: pd.Series, cap: int = 15) -> pd.Series:
    """Cap VehPower at 15 (standard practice, few obs above)."""
    return series.clip(upper=cap)


def _simplify_region(series: pd.Series, top_n: int = 10) -> pd.Series:
    """Keep top N regions, group rest as 'Other'."""
    top = series.value_counts().head(top_n).index
    return series.where(series.isin(top), other="R_Other")


def _simplify_vehbrand(series: pd.Series, top_n: int = 8) -> pd.Series:
    """Keep top N vehicle brands, group rest as 'Other'."""
    top = series.value_counts().head(top_n).index
    return series.where(series.isin(top), other="B_Other")


def create_features(df: pd.DataFrame, for_glm: bool = True) -> pd.DataFrame:
    """
    Create model-ready features from raw freMTPL2 data.

    Parameters
    ----------
    df : DataFrame with raw freMTPL2 columns
    for_glm : If True, create dummy variables (one-hot encoding).
              If False, use label encoding (for tree models).

    Returns
    -------
    DataFrame with engineered features, plus Exposure, ClaimNb, and target columns.
    """
    out = pd.DataFrame(index=df.index)

    # Preserve identifiers and targets
    for col in ["IDpol", "ClaimNb", "Exposure", "HasClaim",
                 "TotalClaimAmount", "AvgClaimAmount", "PurePremium"]:
        if col in df.columns:
            out[col] = df[col].values

    # --- Numeric features ---

    # VehPower: cap at 15
    out["VehPower"] = _cap_vehpower(df["VehPower"])

    # Log Density
    out["LogDensity"] = _log_density(df)

    # BonusMalus (keep raw for GBM, binned for GLM)
    out["BonusMalus"] = df["BonusMalus"].values

    # DrivAge, VehAge (keep raw for GBM)
    out["DrivAge"] = df["DrivAge"].values
    out["VehAge"] = df["VehAge"].values

    # --- Binned features (used by GLM) ---
    out["DrivAge_bin"] = _bin_column(df["DrivAge"], DRIVAGE_BINS, DRIVAGE_LABELS)
    out["VehAge_bin"] = _bin_column(df["VehAge"], VEHAGE_BINS, VEHAGE_LABELS)
    out["BonusMalus_bin"] = _bin_column(df["BonusMalus"], BONUSMALUS_BINS, BONUSMALUS_LABELS)

    # --- Categorical features ---
    out["VehGas"] = df["VehGas"].astype(str)
    out["Area"] = df["Area"].astype(str)
    out["Region"] = _simplify_region(df["Region"].astype(str))
    out["VehBrand"] = _simplify_vehbrand(df["VehBrand"].astype(str))

    if for_glm:
        return _encode_for_glm(out)
    else:
        return _encode_for_gbm(out)


def _encode_for_glm(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical/binned features for GLM."""
    dummy_cols = ["DrivAge_bin", "VehAge_bin", "BonusMalus_bin",
                  "VehGas", "Area", "Region", "VehBrand"]

    # Keep non-dummy columns
    non_dummy = [c for c in df.columns if c not in dummy_cols]
    result = df[non_dummy].copy()

    # Create dummies with drop_first to avoid multicollinearity
    for col in dummy_cols:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=float)
            result = pd.concat([result, dummies], axis=1)

    return result


def _encode_for_gbm(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode categorical features for tree models."""
    result = df.copy()

    cat_cols = ["DrivAge_bin", "VehAge_bin", "BonusMalus_bin",
                "VehGas", "Area", "Region", "VehBrand"]

    for col in cat_cols:
        if col in result.columns:
            result[col] = result[col].astype("category").cat.codes

    return result


def get_glm_feature_names(df_glm: pd.DataFrame) -> list[str]:
    """Get the list of feature column names from a GLM-encoded DataFrame."""
    exclude = {"IDpol", "ClaimNb", "Exposure", "HasClaim",
               "TotalClaimAmount", "AvgClaimAmount", "PurePremium"}
    return [c for c in df_glm.columns if c not in exclude]


def get_gbm_feature_names(df_gbm: pd.DataFrame) -> list[str]:
    """Get the list of feature column names from a GBM-encoded DataFrame."""
    exclude = {"IDpol", "ClaimNb", "Exposure", "HasClaim",
               "TotalClaimAmount", "AvgClaimAmount", "PurePremium"}
    return [c for c in df_gbm.columns if c not in exclude]


# Feature metadata for dashboard display
FEATURE_DESCRIPTIONS = {
    "VehPower": "Vehicle power rating (1-15, capped)",
    "VehAge": "Vehicle age in years",
    "DrivAge": "Driver age in years",
    "BonusMalus": "Bonus-malus coefficient (50=best, 350=worst)",
    "VehBrand": "Vehicle brand (anonymized)",
    "VehGas": "Fuel type (Diesel/Regular)",
    "Area": "Population density area (A-F, rural to urban)",
    "Density": "Number of inhabitants per km2 in the city",
    "Region": "French administrative region (anonymized)",
    "LogDensity": "Log-transformed population density",
}
