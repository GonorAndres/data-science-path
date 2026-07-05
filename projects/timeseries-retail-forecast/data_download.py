"""Download M5 dataset via Kaggle API.

Prerequisite: `~/.kaggle/kaggle.json` with a valid API token.
"""

import subprocess
import zipfile
from pathlib import Path


DATA_DIR = Path(__file__).parent / "data"
COMPETITION = "m5-forecasting-accuracy"


def download() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    zip_path = DATA_DIR / f"{COMPETITION}.zip"
    if not zip_path.exists():
        print(f"downloading {COMPETITION} ...")
        subprocess.run(
            ["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(DATA_DIR)],
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
