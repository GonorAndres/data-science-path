"""Run SQL-level tests. A test 'passes' if the query returns 0 violations."""

import sys
from pathlib import Path

import duckdb


PROJECT_DIR = Path(__file__).resolve().parent.parent
TEST_DIR = PROJECT_DIR / "sql" / "tests"
DB_PATH = PROJECT_DIR / "data" / "olist.duckdb"


def run() -> int:
    if not DB_PATH.exists():
        print(f"ERROR: {DB_PATH} not found. Run `python -m src.pipeline` first.")
        return 1

    conn = duckdb.connect(str(DB_PATH))
    failures = 0
    for f in sorted(TEST_DIR.glob("*.sql")):
        sql = f.read_text()
        n = conn.execute(sql).fetchone()[0]
        status = "PASS" if n == 0 else "FAIL"
        print(f"[{status}] {f.name}: violations={n}")
        if n != 0:
            failures += 1
    conn.close()
    if failures:
        print(f"{failures} test(s) failed.")
        return 1
    print("All tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
