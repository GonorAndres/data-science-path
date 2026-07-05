"""Minimal MLflow experiment tracking + model registry example."""

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


def main() -> None:
    X, y = load_iris(return_X_y=True)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.3, random_state=0)

    mlflow.set_experiment("mlops-fundamentals-demo")
    with mlflow.start_run():
        model = LogisticRegression(max_iter=500, C=1.0)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)

        mlflow.log_param("max_iter", 500)
        mlflow.log_param("C", 1.0)
        mlflow.log_metric("accuracy", accuracy_score(y_val, preds))
        mlflow.log_metric("macro_f1", f1_score(y_val, preds, average="macro"))

        mlflow.sklearn.log_model(model, "model")
        print("Logged run. Inspect with `mlflow ui` on http://localhost:5000.")


if __name__ == "__main__":
    main()
