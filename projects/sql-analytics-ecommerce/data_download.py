"""Download Olist Brazilian E-commerce dataset via Kaggle API."""

import subprocess
import zipfile
from pathlib import Path


DATA_DIR = Path(__file__).parent / "data"
DATASET = "olistbr/brazilian-ecommerce"


def download() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    zip_path = DATA_DIR / "brazilian-ecommerce.zip"
    if not zip_path.exists():
        print(f"downloading {DATASET} ...")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", DATASET, "-p", str(DATA_DIR)],
            check=True,
        )
    print(f"extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(DATA_DIR)
    print("done. Files:")
    for p in sorted(DATA_DIR.glob("*.csv")):
        print(f"  {p.name}")


if __name__ == "__main__":
    download()
