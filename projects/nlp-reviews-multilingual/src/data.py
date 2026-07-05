"""Load and prepare review data for training and evaluation."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LABELS = ["negative", "neutral", "positive"]
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}
ID_TO_LABEL = {i: name for name, i in LABEL_TO_ID.items()}


@dataclass
class Split:
    texts: list[str]
    labels: list[int]
    languages: list[str]


def load_split(split: str, languages: tuple[str, ...] = ("es", "en")) -> Split:
    frames = []
    for lang in languages:
        path = DATA_DIR / f"{lang}_{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run `python data_download.py` first."
            )
        df = pd.read_parquet(path)
        df["language"] = lang
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    return Split(
        texts=all_df["review_body"].astype(str).tolist(),
        labels=all_df["label"].map(LABEL_TO_ID).tolist(),
        languages=all_df["language"].tolist(),
    )


def summarize() -> pd.DataFrame:
    rows = []
    for split in ("train", "validation", "test"):
        try:
            s = load_split(split)
        except FileNotFoundError:
            continue
        df = pd.DataFrame({"label": s.labels, "language": s.languages})
        counts = df.groupby(["language", "label"]).size().unstack(fill_value=0)
        counts["total"] = counts.sum(axis=1)
        counts.insert(0, "split", split)
        rows.append(counts.reset_index())
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out


if __name__ == "__main__":
    print(summarize())
