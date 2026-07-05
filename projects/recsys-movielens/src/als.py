"""ALS on implicit feedback via the `implicit` library."""

from pathlib import Path

import implicit
import mlflow
import numpy as np
from scipy.sparse import csr_matrix

from .data import encode_ids, leave_last_out, load_ratings


MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def run() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    ratings = load_ratings()
    train, _ = leave_last_out(ratings)
    train, user_map, item_map = encode_ids(train)

    n_users = len(user_map)
    n_items = len(item_map)

    matrix = csr_matrix(
        (np.ones(len(train)), (train["u_idx"].values, train["i_idx"].values)),
        shape=(n_users, n_items),
    )

    model = implicit.als.AlternatingLeastSquares(
        factors=64,
        regularization=0.01,
        iterations=20,
    )

    mlflow.set_experiment("recsys-movielens")
    with mlflow.start_run(run_name="als"):
        mlflow.log_params({
            "model": "als",
            "factors": 64,
            "regularization": 0.01,
            "iterations": 20,
        })
        model.fit(matrix)

        np.save(MODEL_DIR / "als_user_factors.npy", model.user_factors)
        np.save(MODEL_DIR / "als_item_factors.npy", model.item_factors)
        print(f"ALS done. n_users={n_users}, n_items={n_items}")


if __name__ == "__main__":
    run()
