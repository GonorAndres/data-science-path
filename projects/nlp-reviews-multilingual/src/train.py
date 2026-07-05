"""Fine-tune a multilingual transformer on Amazon Reviews Multi."""

from pathlib import Path

import mlflow
import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from .data import LABELS, ID_TO_LABEL, LABEL_TO_ID, load_split


MODEL_NAME = "distilbert-base-multilingual-cased"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "transformer"
MAX_LENGTH = 128


def to_hf_dataset(split) -> Dataset:
    return Dataset.from_dict({
        "text": split.texts,
        "label": split.labels,
        "language": split.languages,
    })


def tokenize_fn(batch, tokenizer):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"macro_f1": f1_score(labels, preds, average="macro")}


def train() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    train_ds = to_hf_dataset(load_split("train")).map(
        lambda b: tokenize_fn(b, tokenizer),
        batched=True,
    )
    val_ds = to_hf_dataset(load_split("validation")).map(
        lambda b: tokenize_fn(b, tokenizer),
        batched=True,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(MODEL_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=100,
        fp16=torch.cuda.is_available(),
        report_to="mlflow",
        run_name="distilbert-multi-sentiment",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    mlflow.set_experiment("nlp-reviews-multilingual")
    with mlflow.start_run(run_name="distilbert-multi-sentiment"):
        trainer.train()
        metrics = trainer.evaluate()
        mlflow.log_metrics({f"final_{k}": v for k, v in metrics.items() if isinstance(v, (int, float))})
        trainer.save_model(str(MODEL_DIR / "best"))
        tokenizer.save_pretrained(MODEL_DIR / "best")
        print(f"best model saved to {MODEL_DIR / 'best'}")


if __name__ == "__main__":
    train()
