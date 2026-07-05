"""Feature engineering for the global LightGBM model."""

import pandas as pd


LAG_DAYS = [7, 14, 28, 365]
ROLLING_WINDOWS = [7, 28, 56]


def add_lag_features(df: pd.DataFrame, target: str = "sales") -> pd.DataFrame:
    df = df.sort_values(["id", "date"]).copy()
    for lag in LAG_DAYS:
        df[f"lag_{lag}"] = df.groupby("id")[target].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, target: str = "sales") -> pd.DataFrame:
    for window in ROLLING_WINDOWS:
        df[f"roll_mean_{window}"] = (
            df.groupby("id")[target]
            .shift(7)
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(drop=True)
        )
        df[f"roll_std_{window}"] = (
            df.groupby("id")[target]
            .shift(7)
            .rolling(window=window, min_periods=1)
            .std()
            .reset_index(drop=True)
        )
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df["dow"] = df["date"].dt.dayofweek
    df["dom"] = df["date"].dt.day
    df["woy"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["id", "date"]).copy()
    df["price_lag_7"] = df.groupby("id")["sell_price"].shift(7)
    df["price_change_7"] = (df["sell_price"] - df["price_lag_7"]) / df["price_lag_7"].replace(0, 1)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_calendar_features(df)
    df = add_price_features(df)
    return df
