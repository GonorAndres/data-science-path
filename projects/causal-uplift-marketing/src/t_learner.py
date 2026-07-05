"""T-learner: two LightGBM classifiers (treated / control), difference is uplift."""

from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
import numpy as np

from .data import load, split


MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def run() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    df = load()
    x_train, x_test, y_train, y_test, t_train, t_test = split(df)

    x1, y1 = x_train[t_train == 1], y_train[t_train == 1]
    x0, y0 = x_train[t_train == 0], y_train[t_train == 0]

    params = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63,
              "verbosity": -1, "num_iterations": 300}

    m1 = lgb.train(params, lgb.Dataset(x1, y1))
    m0 = lgb.train(params, lgb.Dataset(x0, y0))

    mlflow.set_experiment("causal-uplift-marketing")
    with mlflow.start_run(run_name="t-learner"):
        mlflow.log_params(params)
        uplift_test = m1.predict(x_test) - m0.predict(x_test)
        np.save(MODEL_DIR / "uplift_t.npy", uplift_test)
        joblib.dump({"m1": m1, "m0": m0}, MODEL_DIR / "t_learner.joblib")
    print(f"uplift mean {uplift_test.mean():.4f}, std {uplift_test.std():.4f}")


if __name__ == "__main__":
    run()
