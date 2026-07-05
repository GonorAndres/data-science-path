"""Feature engineering smoke tests."""

import numpy as np
import pandas as pd

from src.features import build_features


def test_build_features_produces_lags():
    dates = pd.date_range("2024-01-01", periods=400)
    df = pd.DataFrame({
        "id": ["a"] * 400,
        "date": dates,
        "sales": np.random.randint(0, 10, size=400),
        "sell_price": np.random.uniform(1, 5, size=400),
    })
    out = build_features(df)
    assert "lag_7" in out.columns
    assert "lag_365" in out.columns
    assert "roll_mean_28" in out.columns
    assert "dow" in out.columns
    assert "price_change_7" in out.columns


def test_lag_values_are_shifted():
    dates = pd.date_range("2024-01-01", periods=400)
    df = pd.DataFrame({
        "id": ["a"] * 400,
        "date": dates,
        "sales": list(range(400)),
        "sell_price": [1.0] * 400,
    })
    out = build_features(df)
    out = out.sort_values("date").reset_index(drop=True)
    assert out.loc[7, "lag_7"] == 0
    assert out.loc[14, "lag_7"] == 7
