"""Fetch MovieLens 25M dataset."""

import urllib.request
import zipfile
from pathlib import Path


DATA_DIR = Path(__file__).parent / "data"
URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"


def download() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    zip_path = DATA_DIR / "ml-25m.zip"
    if not zip_path.exists():
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, zip_path)
    print(f"extracting {zip_path}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(DATA_DIR)
    print("done")


if __name__ == "__main__":
    download()
