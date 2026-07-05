"""Train the defect classifier for one MVTec category."""

import argparse
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from .data import (
    MVTecDataset,
    eval_transforms,
    labels_for_category,
    train_transforms,
)
from .model import build_model


MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def train(category: str, epochs: int = 10, batch: int = 16, lr: float = 1e-4) -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = labels_for_category(category)
    num_classes = len(labels)
    print(f"{category}: {num_classes} classes: {labels}")

    train_ds = MVTecDataset(category, "test", labels, train_transforms())
    val_ds = MVTecDataset(category, "test", labels, eval_transforms())

    n = len(train_ds)
    indices = np.random.permutation(n)
    split = int(0.2 * n)
    val_idx = set(indices[:split].tolist())
    train_idx = set(indices[split:].tolist())

    train_ds = torch.utils.data.Subset(train_ds, sorted(train_idx))
    val_ds = torch.utils.data.Subset(val_ds, sorted(val_idx))
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=2)

    model = build_model(num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    best_f1 = -1.0
    ckpt_path = MODEL_DIR / f"{category}.pt"

    mlflow.set_experiment("cv-defect-detection")
    with mlflow.start_run(run_name=f"resnet50-{category}"):
        mlflow.log_params({"category": category, "epochs": epochs, "lr": lr, "batch": batch})
        for epoch in range(epochs):
            model.train()
            for x, y in train_dl:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = loss_fn(logits, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()

            model.eval()
            ys, preds = [], []
            with torch.no_grad():
                for x, y in val_dl:
                    x = x.to(device)
                    p = model(x).argmax(-1).cpu().numpy()
                    ys.extend(y.numpy().tolist())
                    preds.extend(p.tolist())
            f1 = f1_score(ys, preds, average="macro")
            print(f"epoch {epoch + 1}: val macro F1 {f1:.4f}")
            mlflow.log_metric("val_macro_f1", f1, step=epoch)
            if f1 > best_f1:
                best_f1 = f1
                torch.save({"model": model.state_dict(), "labels": labels}, ckpt_path)
        mlflow.log_metric("best_val_macro_f1", best_f1)
        print(f"best macro F1: {best_f1:.4f}. Saved to {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    train(args.category, epochs=args.epochs)
