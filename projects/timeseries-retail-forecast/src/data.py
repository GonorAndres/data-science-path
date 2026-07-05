"""M5 data loading and reshaping to long form."""

from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sales = pd.read_csv(DATA_DIR / "sales_train_validation.csv")
    calendar = pd.read_csv(DATA_DIR / "calendar.csv")
    prices = pd.read_csv(DATA_DIR / "sell_prices.csv")
    return sales, calendar, prices


def to_long(sales: pd.DataFrame) -> pd.DataFrame:
    id_vars = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    long_df = sales.melt(
        id_vars=id_vars,
        value_vars=day_cols,
        var_name="d",
        value_name="sales",
    )
    return long_df


def attach_calendar(long_df: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    cal = calendar.copy()
    cal["date"] = pd.to_datetime(cal["date"])
    return long_df.merge(cal, on="d", how="left")


def attach_prices(df: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    return df.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")


def load_prepared() -> pd.DataFrame:
    sales, calendar, prices = load_raw()
    long_df = to_long(sales)
    with_cal = attach_calendar(long_df, calendar)
    out = attach_prices(with_cal, prices)
    out["sell_price"] = out["sell_price"].fillna(0)
    return out


if __name__ == "__main__":
    df = load_prepared()
    print(df.shape)
    print(df.head())
