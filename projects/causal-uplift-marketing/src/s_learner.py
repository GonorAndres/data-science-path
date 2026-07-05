"""S-learner: single model with treatment as a feature."""

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

    xt_train = x_train.copy()
    xt_train["_treatment"] = t_train.values
    xt_test = x_test.copy()
    xt_test["_treatment"] = t_test.values

    params = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63,
              "verbosity": -1, "num_iterations": 300}
    model = lgb.train(params, lgb.Dataset(xt_train, y_train))

    xt1 = x_test.assign(_treatment=1)
    xt0 = x_test.assign(_treatment=0)
    uplift = model.predict(xt1) - model.predict(xt0)

    mlflow.set_experiment("causal-uplift-marketing")
    with mlflow.start_run(run_name="s-learner"):
        mlflow.log_params(params)
        np.save(MODEL_DIR / "uplift_s.npy", uplift)
        joblib.dump(model, MODEL_DIR / "s_learner.joblib")
    print(f"uplift mean {uplift.mean():.4f}, std {uplift.std():.4f}")


if __name__ == "__main__":
    run()
