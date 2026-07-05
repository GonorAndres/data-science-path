"""Download Criteo Uplift Prediction dataset.

Expects the user to download the file manually due to Criteo's agreement,
OR points at a public Kaggle mirror if available.
"""

from pathlib import Path


DATA_DIR = Path(__file__).parent / "data"
URL_HINT = (
    "https://ailab.criteo.com/criteo-uplift-prediction-dataset/\n"
    "Or check Kaggle for a redistribution of criteo-uplift-v2."
)


def check() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    files = list(DATA_DIR.glob("*.csv*"))
    if not files:
        print(f"No Criteo file at {DATA_DIR}. Download from:\n{URL_HINT}")
        return
    print("Found:")
    for f in files:
        print(f"  {f.name} ({f.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    check()
