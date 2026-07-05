"""TF-IDF + logistic regression baseline. Logs to MLflow."""

from pathlib import Path

import mlflow
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline

from .data import LABELS, load_split


MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def train_baseline() -> None:
    train = load_split("train")
    val = load_split("validation")

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=5,
            max_features=50_000,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            C=1.0,
            class_weight="balanced",
            n_jobs=-1,
        )),
    ])

    mlflow.set_experiment("nlp-reviews-multilingual")
    with mlflow.start_run(run_name="tfidf-logreg-baseline"):
        mlflow.log_params({
            "model": "tfidf+logreg",
            "ngram": "1-2",
            "min_df": 5,
            "max_features": 50_000,
            "class_weight": "balanced",
        })
        pipe.fit(train.texts, train.labels)

        preds = pipe.predict(val.texts)
        macro_f1 = f1_score(val.labels, preds, average="macro")
        mlflow.log_metric("val_macro_f1", macro_f1)

        report = classification_report(
            val.labels,
            preds,
            target_names=LABELS,
            digits=3,
        )
        print(report)
        MODEL_DIR.mkdir(exist_ok=True)
        report_path = MODEL_DIR / "baseline_report.txt"
        report_path.write_text(report)
        mlflow.log_artifact(str(report_path))

        import joblib
        model_path = MODEL_DIR / "baseline_tfidf_logreg.joblib"
        joblib.dump(pipe, model_path)
        mlflow.log_artifact(str(model_path))

        print(f"val macro F1: {macro_f1:.3f}")
        print(f"model saved to {model_path}")


if __name__ == "__main__":
    train_baseline()
