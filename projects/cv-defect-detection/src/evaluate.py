"""Evaluate a trained classifier on the MVTec category's test set."""

import argparse
from pathlib import Path

import torch
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader

from .data import MVTecDataset, eval_transforms, labels_for_category
from .model import build_model


MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def evaluate(category: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = labels_for_category(category)
    ds = MVTecDataset(category, "test", labels, eval_transforms())
    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2)

    ckpt = torch.load(MODEL_DIR / f"{category}.pt", map_location=device)
    model = build_model(len(labels)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ys, preds = [], []
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            p = model(x).argmax(-1).cpu().numpy()
            ys.extend(y.numpy().tolist())
            preds.extend(p.tolist())

    print(f"macro F1: {f1_score(ys, preds, average='macro'):.4f}")
    print(classification_report(ys, preds, target_names=labels, digits=3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True)
    args = parser.parse_args()
    evaluate(args.category)
