"""Download Amazon Reviews Multi (Spanish + English) and cache locally.

Run once. Produces Parquet files under ./data/.
"""

from pathlib import Path

from datasets import load_dataset


DATA_DIR = Path(__file__).parent / "data"
LANGUAGES = ("es", "en")
SPLITS = ("train", "validation", "test")


def star_to_label(star: int) -> str:
    if star <= 2:
        return "negative"
    if star == 3:
        return "neutral"
    return "positive"


def download() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    for lang in LANGUAGES:
        for split in SPLITS:
            out_path = DATA_DIR / f"{lang}_{split}.parquet"
            if out_path.exists():
                print(f"exists: {out_path}")
                continue
            print(f"downloading {lang} {split}")
            ds = load_dataset("amazon_reviews_multi", lang, split=split)
            df = ds.to_pandas()
            df["label"] = df["stars"].map(star_to_label)
            keep = ["review_id", "review_body", "stars", "label", "product_category"]
            df[keep].to_parquet(out_path, index=False)
            print(f"wrote {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    download()
