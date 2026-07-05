"""Tune LightGBM on Iris with Optuna."""

import lightgbm as lgb
import optuna
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def objective(trial: optuna.Trial) -> float:
    X, y = load_iris(return_X_y=True)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.3, random_state=0)

    params = {
        "objective": "multiclass",
        "num_class": 3,
        "learning_rate": trial.suggest_float("lr", 1e-3, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 7, 63),
        "min_child_samples": trial.suggest_int("min_child_samples", 2, 40),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
        "verbosity": -1,
        "num_iterations": 200,
    }
    model = lgb.train(params, lgb.Dataset(X_tr, y_tr))
    preds = model.predict(X_val).argmax(axis=-1)
    return accuracy_score(y_val, preds)


def main() -> None:
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)
    print(f"best accuracy: {study.best_value:.4f}")
    print(f"best params: {study.best_params}")


if __name__ == "__main__":
    main()
