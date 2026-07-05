"""Orchestrate DuckDB to run staging -> intermediate -> marts in order."""

from pathlib import Path

import duckdb


PROJECT_DIR = Path(__file__).resolve().parent.parent
SQL_DIR = PROJECT_DIR / "sql"
DB_PATH = PROJECT_DIR / "data" / "olist.duckdb"


LAYERS = ["01_staging", "02_intermediate", "03_marts"]


def run() -> None:
    conn = duckdb.connect(str(DB_PATH))
    try:
        conn.execute(f"SET work_mem = '1GB';")
    except Exception:
        pass

    for layer in LAYERS:
        layer_dir = SQL_DIR / layer
        files = sorted(layer_dir.glob("*.sql"))
        print(f"--- {layer} ({len(files)} models) ---")
        for f in files:
            sql = f.read_text()
            print(f"running {f.name}")
            conn.execute(sql)

    print("--- summary ---")
    for row in conn.execute("SHOW TABLES").fetchall():
        (name,) = row
        count = conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
        print(f"  {name}: {count:,}")

    conn.close()


if __name__ == "__main__":
    run()
