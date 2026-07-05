"""Read-only SQL tool over the Olist DuckDB warehouse."""

import re
from pathlib import Path

import duckdb


DB_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "sql-analytics-ecommerce"
    / "data"
    / "olist.duckdb"
)

FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|GRANT|REVOKE|ATTACH|DETACH|COPY|EXPORT|IMPORT)\b",
    re.IGNORECASE,
)


def sql_query(query: str, row_cap: int = 50) -> list[dict]:
    if FORBIDDEN.search(query):
        raise PermissionError("Only SELECT statements are allowed.")
    if not DB_PATH.exists():
        raise FileNotFoundError(f"{DB_PATH} not found. Run the sql-analytics-ecommerce pipeline.")
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        result = conn.execute(query).fetchdf()
        if len(result) > row_cap:
            result = result.head(row_cap)
        return result.to_dict(orient="records")
    finally:
        conn.close()
