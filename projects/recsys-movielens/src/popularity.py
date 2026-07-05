"""Popularity baseline: recommend the globally most popular unseen movies."""

from pathlib import Path

from .data import leave_last_out, load_ratings


MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def run() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    ratings = load_ratings()
    train, _ = leave_last_out(ratings)
    popularity = train.groupby("movieId").size().sort_values(ascending=False)
    popularity.to_frame("score").to_csv(MODEL_DIR / "popularity.csv")
    print(f"Top 10 most popular: {popularity.head(10).index.tolist()}")
    print(f"Saved popularity scores for {len(popularity)} items.")


if __name__ == "__main__":
    run()
