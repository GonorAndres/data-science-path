"""Track a small grid of Logistic Regression C values on Iris with MLflow."""

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


def main() -> None:
    X, y = load_iris(return_X_y=True)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.3, random_state=0)

    mlflow.set_experiment("experiment-tracking-demo")
    for c in [0.01, 0.1, 1.0, 10.0]:
        with mlflow.start_run(run_name=f"C={c}"):
            model = LogisticRegression(C=c, max_iter=500)
            model.fit(X_tr, y_tr)
            score = f1_score(y_val, model.predict(X_val), average="macro")
            mlflow.log_param("C", c)
            mlflow.log_metric("val_macro_f1", score)
            mlflow.sklearn.log_model(model, "model")
            print(f"C={c}: macro F1 {score:.4f}")


if __name__ == "__main__":
    main()
