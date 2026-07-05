"""Simplified R-learner: outcome and propensity residualization."""

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

    params = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63,
              "verbosity": -1, "num_iterations": 300}

    m_y = lgb.train(params, lgb.Dataset(x_train, y_train))
    m_t = lgb.train(params, lgb.Dataset(x_train, t_train))

    y_hat_train = m_y.predict(x_train)
    t_hat_train = m_t.predict(x_train)

    y_resid = y_train.values - y_hat_train
    t_resid = t_train.values - t_hat_train

    weights = t_resid ** 2 + 1e-6
    target = y_resid / (t_resid + 1e-6)

    params_tau = {"objective": "regression", "learning_rate": 0.05,
                  "num_leaves": 31, "verbosity": -1, "num_iterations": 300}
    tau = lgb.train(params_tau, lgb.Dataset(x_train, target, weight=weights))

    uplift = tau.predict(x_test)

    mlflow.set_experiment("causal-uplift-marketing")
    with mlflow.start_run(run_name="r-learner"):
        mlflow.log_params(params)
        np.save(MODEL_DIR / "uplift_r.npy", uplift)
        joblib.dump({"m_y": m_y, "m_t": m_t, "tau": tau}, MODEL_DIR / "r_learner.joblib")
    print(f"uplift mean {uplift.mean():.4f}, std {uplift.std():.4f}")


if __name__ == "__main__":
    run()
