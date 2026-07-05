"""TFT on a synthetic dataset via `pytorch_forecasting`.

Run with GPU for reasonable speed. This is a minimal sketch; see the
pytorch_forecasting tutorial at
pytorch-forecasting.readthedocs.io for production-ready guidance.
"""

import numpy as np
import pandas as pd


def make_series(n_series: int = 30, length: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_series):
        base = rng.normal(0, 1)
        trend = np.linspace(0, rng.uniform(-1, 1), length)
        season = np.sin(np.arange(length) * 2 * np.pi / 7)
        noise = rng.normal(0, 0.3, size=length)
        y = 5 + base + trend + season + noise
        rows.append(pd.DataFrame({
            "series_id": [i] * length,
            "time_idx": np.arange(length),
            "y": y,
        }))
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    df = make_series()
    print(df.head())
    print(df.shape)
    print("To fit TFT on this frame, see docs at pytorch-forecasting.readthedocs.io")
    print(
        "Quickstart imports: "
        "`from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer`"
    )


if __name__ == "__main__":
    main()
