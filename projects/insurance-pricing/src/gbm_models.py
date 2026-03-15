"""
XGBoost and LightGBM models for insurance frequency-severity pricing.
Optuna hyperparameter tuning with MLflow tracking.
"""

import mlflow
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance


# Suppress Optuna info logs during tuning
optuna.logging.set_verbosity(optuna.logging.WARNING)


def train_xgb_frequency(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    exposure_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    exposure_val: pd.Series,
    feature_names: list[str],
    n_trials: int = 100,
    experiment_name: str = "insurance_pricing_xgb_freq",
) -> tuple[xgb.Booster, dict]:
    """
    Train XGBoost Poisson model with Optuna tuning.

    Uses sample_weight=Exposure for proper exposure handling in tree models.
    Returns (best_booster, best_params).
    """
    dtrain = xgb.DMatrix(
        X_train[feature_names], label=y_train,
        weight=exposure_train,
    )
    dval = xgb.DMatrix(
        X_val[feature_names], label=y_val,
        weight=exposure_val,
    )

    mlflow.set_experiment(experiment_name)

    best_model = None
    best_score = float("inf")
    best_params = {}

    def objective(trial):
        nonlocal best_model, best_score, best_params

        params = {
            "objective": "count:poisson",
            "eval_metric": "poisson-nloglik",
            "tree_method": "hist",
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 100, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
            "verbosity": 0,
        }

        with mlflow.start_run(nested=True):
            mlflow.log_params(params)

            model = xgb.train(
                params, dtrain,
                num_boost_round=500,
                evals=[(dval, "val")],
                early_stopping_rounds=50,
                verbose_eval=False,
            )

            val_pred = model.predict(dval)
            deviance = float(mean_poisson_deviance(y_val, val_pred, sample_weight=exposure_val))

            mlflow.log_metric("val_poisson_deviance", deviance)
            mlflow.log_metric("best_iteration", model.best_iteration)

            if deviance < best_score:
                best_score = deviance
                best_model = model
                best_params = params.copy()
                best_params["best_iteration"] = model.best_iteration

        return deviance

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"XGBoost Frequency -- Best Poisson deviance: {best_score:.6f}")
    print(f"  Best params: max_depth={best_params.get('max_depth')}, "
          f"lr={best_params.get('learning_rate', 0):.4f}")

    return best_model, best_params


def train_lgb_frequency(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    exposure_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    exposure_val: pd.Series,
    feature_names: list[str],
    n_trials: int = 100,
    experiment_name: str = "insurance_pricing_lgb_freq",
) -> tuple[lgb.Booster, dict]:
    """
    Train LightGBM Poisson model with Optuna tuning.
    Returns (best_booster, best_params).
    """
    dtrain = lgb.Dataset(
        X_train[feature_names], label=y_train,
        weight=exposure_train, free_raw_data=False,
    )
    dval = lgb.Dataset(
        X_val[feature_names], label=y_val,
        weight=exposure_val, reference=dtrain, free_raw_data=False,
    )

    mlflow.set_experiment(experiment_name)

    best_model = None
    best_score = float("inf")
    best_params = {}

    def objective(trial):
        nonlocal best_model, best_score, best_params

        params = {
            "objective": "poisson",
            "metric": "poisson",
            "boosting_type": "gbdt",
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 1e-8, 1.0, log=True),
            "verbose": -1,
            "feature_pre_filter": False,
        }

        with mlflow.start_run(nested=True):
            mlflow.log_params(params)

            callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
            model = lgb.train(
                params, dtrain,
                num_boost_round=500,
                valid_sets=[dval],
                callbacks=callbacks,
            )

            val_pred = model.predict(X_val[feature_names])
            deviance = float(mean_poisson_deviance(y_val, val_pred, sample_weight=exposure_val))

            mlflow.log_metric("val_poisson_deviance", deviance)
            mlflow.log_metric("best_iteration", model.best_iteration)

            if deviance < best_score:
                best_score = deviance
                best_model = model
                best_params = params.copy()
                best_params["best_iteration"] = model.best_iteration

        return deviance

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"LightGBM Frequency -- Best Poisson deviance: {best_score:.6f}")
    print(f"  Best params: num_leaves={best_params.get('num_leaves')}, "
          f"lr={best_params.get('learning_rate', 0):.4f}")

    return best_model, best_params


def train_xgb_severity(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_names: list[str],
    n_trials: int = 50,
    experiment_name: str = "insurance_pricing_xgb_sev",
) -> tuple[xgb.Booster, dict]:
    """
    Train XGBoost Gamma model for claim severity (y > 0 only).
    Returns (best_booster, best_params).
    """
    # Filter to positive claims
    mask_train = y_train > 0
    mask_val = y_val > 0

    dtrain = xgb.DMatrix(X_train.loc[mask_train, feature_names], label=y_train[mask_train])
    dval = xgb.DMatrix(X_val.loc[mask_val, feature_names], label=y_val[mask_val])

    mlflow.set_experiment(experiment_name)

    best_model = None
    best_score = float("inf")
    best_params = {}

    def objective(trial):
        nonlocal best_model, best_score, best_params

        params = {
            "objective": "reg:gamma",
            "eval_metric": "gamma-nloglik",
            "tree_method": "hist",
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 5, 200, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "verbosity": 0,
        }

        with mlflow.start_run(nested=True):
            mlflow.log_params(params)

            model = xgb.train(
                params, dtrain,
                num_boost_round=300,
                evals=[(dval, "val")],
                early_stopping_rounds=30,
                verbose_eval=False,
            )

            val_pred = model.predict(dval).clip(min=1.0)
            deviance = float(mean_gamma_deviance(y_val[mask_val], val_pred))

            mlflow.log_metric("val_gamma_deviance", deviance)

            if deviance < best_score:
                best_score = deviance
                best_model = model
                best_params = params.copy()

        return deviance

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"XGBoost Severity -- Best Gamma deviance: {best_score:.6f}")
    return best_model, best_params


def predict_xgb(model: xgb.Booster, X: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    """Generate predictions from an XGBoost booster."""
    dmatrix = xgb.DMatrix(X[feature_names])
    return model.predict(dmatrix)


def predict_lgb(model: lgb.Booster, X: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    """Generate predictions from a LightGBM booster."""
    return model.predict(X[feature_names])


def get_feature_importance(model, feature_names: list[str], model_type: str = "xgb") -> list[dict]:
    """
    Extract feature importance as JSON-serializable list.
    Returns list of dicts with: feature, importance, rank.
    """
    if model_type == "xgb":
        importance = model.get_score(importance_type="gain")
        imp_list = [{"feature": f, "importance": round(float(importance.get(f, 0)), 4)}
                    for f in feature_names]
    elif model_type == "lgb":
        raw_importance = model.feature_importance(importance_type="gain")
        names = model.feature_name()
        imp_list = [{"feature": names[i], "importance": round(float(raw_importance[i]), 4)}
                    for i in range(len(names))]
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    imp_list.sort(key=lambda x: x["importance"], reverse=True)
    for i, item in enumerate(imp_list):
        item["rank"] = i + 1

    return imp_list
