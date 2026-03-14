"""Shared fixtures for insurance pricing tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_freq_data():
    """Small synthetic frequency dataset mimicking freMTPL2 structure."""
    rng = np.random.RandomState(42)
    n = 500

    df = pd.DataFrame({
        "IDpol": range(1, n + 1),
        "ClaimNb": rng.poisson(0.04, n),
        "Exposure": rng.uniform(0.1, 1.0, n),
        "VehPower": rng.randint(4, 12, n),
        "VehAge": rng.randint(0, 20, n),
        "DrivAge": rng.randint(18, 80, n),
        "BonusMalus": rng.randint(50, 150, n),
        "VehBrand": rng.choice(["B1", "B2", "B3", "B4", "B5"], n),
        "VehGas": rng.choice(["Diesel", "Regular"], n),
        "Area": rng.choice(["A", "B", "C", "D", "E", "F"], n),
        "Density": rng.lognormal(5, 2, n).astype(int),
        "Region": rng.choice(["R11", "R24", "R53", "R82", "R93"], n),
    })

    df["ClaimNb"] = df["ClaimNb"].clip(upper=4)
    df["HasClaim"] = (df["ClaimNb"] > 0).astype(int)

    return df


@pytest.fixture
def sample_predictions(sample_freq_data):
    """Sample model predictions for evaluation tests."""
    rng = np.random.RandomState(42)
    n = len(sample_freq_data)
    y_true = sample_freq_data["ClaimNb"].values.astype(float)

    # GLM-like predictions (slightly noisy version of true)
    glm_pred = np.clip(y_true + rng.normal(0, 0.02, n), 0.001, 1.0)

    # GBM-like predictions (closer to true)
    gbm_pred = np.clip(y_true + rng.normal(0, 0.01, n), 0.001, 1.0)

    # Random baseline
    random_pred = rng.uniform(0.01, 0.1, n)

    return {
        "y_true": y_true,
        "glm_pred": glm_pred,
        "gbm_pred": gbm_pred,
        "random_pred": random_pred,
        "weights": sample_freq_data["Exposure"].values,
    }
