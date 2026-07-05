"""Load marts from the sql-analytics-ecommerce DuckDB."""

from pathlib import Path

import duckdb
import pandas as pd


DEFAULT_DB = (
    Path(__file__).resolve().parent.parent.parent
    / "sql-analytics-ecommerce"
    / "data"
    / "olist.duckdb"
)


def _conn(db_path: Path | None = None) -> duckdb.DuckDBPyConnection:
    path = db_path or DEFAULT_DB
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run the sql-analytics-ecommerce pipeline first."
        )
    return duckdb.connect(str(path), read_only=True)


def load_monthly_revenue(db_path: Path | None = None) -> pd.DataFrame:
    with _conn(db_path) as c:
        return c.execute("SELECT * FROM mart_monthly_revenue ORDER BY month").df()


def load_retention_curve(db_path: Path | None = None) -> pd.DataFrame:
    with _conn(db_path) as c:
        return c.execute("SELECT * FROM mart_retention_curve").df()


def load_rolling_revenue(db_path: Path | None = None) -> pd.DataFrame:
    with _conn(db_path) as c:
        return c.execute("SELECT * FROM mart_rolling_revenue").df()


def load_product_performance(db_path: Path | None = None) -> pd.DataFrame:
    with _conn(db_path) as c:
        return c.execute(
            "SELECT * FROM mart_product_performance WHERE rank_in_category <= 5 "
            "ORDER BY product_category_name, rank_in_category"
        ).df()


def load_ltv_by_channel(db_path: Path | None = None) -> pd.DataFrame:
    with _conn(db_path) as c:
        return c.execute("SELECT * FROM mart_ltv_by_channel").df()
