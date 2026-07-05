"""Global LightGBM forecaster using lag and calendar features."""

from pathlib import Path

import lightgbm as lgb
import mlflow
import pandas as pd

from .data import load_prepared
from .evaluate import wape
from .features import build_features


MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
HORIZON = 28

FEATURES = [
    "lag_7", "lag_14", "lag_28", "lag_365",
    "roll_mean_7", "roll_std_7",
    "roll_mean_28", "roll_std_28",
    "roll_mean_56", "roll_std_56",
    "dow", "dom", "woy", "month", "quarter", "is_weekend",
    "sell_price", "price_change_7",
]

CATEGORICAL = []


def run() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    df = load_prepared()
    df = build_features(df)
    df = df.dropna(subset=FEATURES + ["sales"])

    max_date = df["date"].max()
    split_date = max_date - pd.Timedelta(days=HORIZON)
    train = df[df["date"] <= split_date]
    test = df[df["date"] > split_date]

    x_train, y_train = train[FEATURES], train["sales"]
    x_test, y_test = test[FEATURES], test["sales"]

    params = {
        "objective": "tweedie",
        "tweedie_variance_power": 1.1,
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 128,
        "min_child_samples": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbosity": -1,
        "num_iterations": 2000,
    }

    mlflow.set_experiment("timeseries-retail-forecast")
    with mlflow.start_run(run_name="lightgbm-global"):
        mlflow.log_params(params)
        ds = lgb.Dataset(x_train, label=y_train, categorical_feature=CATEGORICAL)
        val_ds = lgb.Dataset(x_test, label=y_test, reference=ds)
        model = lgb.train(
            params,
            ds,
            valid_sets=[val_ds],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
        )

        preds = model.predict(x_test)
        score = wape(y_test.values, preds)
        print(f"LightGBM global WAPE: {score:.4f}")
        mlflow.log_metric("test_wape", score)

        out = test.copy()
        out["pred"] = preds
        out.to_parquet(MODEL_DIR / "gbm_predictions.parquet", index=False)
        model.save_model(str(MODEL_DIR / "lgb_global.txt"))


if __name__ == "__main__":
    run()
