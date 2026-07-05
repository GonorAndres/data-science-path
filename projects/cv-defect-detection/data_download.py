"""Download MVTec AD (requires ~5 GB disk). Requires separate license agreement.

NOTE: MVTec AD distribution requires a separate EULA acceptance. Visit
https://www.mvtec.com/company/research/datasets/mvtec-ad and download manually,
then extract under ./data/mvtec_ad/. This script verifies the layout.
"""

from pathlib import Path


DATA_DIR = Path(__file__).parent / "data" / "mvtec_ad"


def verify() -> None:
    if not DATA_DIR.exists():
        print(
            "MVTec AD not found at "
            f"{DATA_DIR}.\n"
            "1) Go to https://www.mvtec.com/company/research/datasets/mvtec-ad\n"
            "2) Accept the license and download mvtec_anomaly_detection.tar.xz.\n"
            "3) Extract into the 'data/mvtec_ad' folder of this project.\n"
            "4) Expected layout: data/mvtec_ad/<category>/train/good/*.png\n"
            "                    data/mvtec_ad/<category>/test/<defect_type>/*.png"
        )
        return
    categories = [p.name for p in DATA_DIR.iterdir() if p.is_dir()]
    print(f"Found {len(categories)} categories: {categories}")


if __name__ == "__main__":
    verify()
