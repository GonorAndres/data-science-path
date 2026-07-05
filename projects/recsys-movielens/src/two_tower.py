"""Small PyTorch two-tower recommender."""

from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .data import encode_ids, leave_last_out, load_ratings, sample_negatives


MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
EMBED_DIM = 64
EPOCHS = 3
BATCH = 4096
LR = 1e-3


class TwoTower(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int = EMBED_DIM):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, u, i):
        u_vec = nn.functional.normalize(self.user_emb(u), dim=-1)
        i_vec = nn.functional.normalize(self.item_emb(i), dim=-1)
        return (u_vec * i_vec).sum(dim=-1)


def run() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ratings = load_ratings()
    train, _ = leave_last_out(ratings)
    train, user_map, item_map = encode_ids(train)
    n_users, n_items = len(user_map), len(item_map)

    sample = sample_negatives(train, n_items=n_items, k=5)
    ds = TensorDataset(
        torch.tensor(sample["u_idx"].values, dtype=torch.long),
        torch.tensor(sample["i_idx"].values, dtype=torch.long),
        torch.tensor(sample["label"].values, dtype=torch.float32),
    )
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True)

    model = TwoTower(n_users, n_items).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    mlflow.set_experiment("recsys-movielens")
    with mlflow.start_run(run_name="two-tower"):
        mlflow.log_params({
            "model": "two-tower",
            "embed_dim": EMBED_DIM,
            "epochs": EPOCHS,
            "batch": BATCH,
            "lr": LR,
        })
        for epoch in range(EPOCHS):
            total = 0.0
            n = 0
            for u, i, y in dl:
                u, i, y = u.to(device), i.to(device), y.to(device)
                logits = model(u, i)
                loss = loss_fn(logits, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.item()) * u.size(0)
                n += u.size(0)
            avg = total / max(n, 1)
            print(f"epoch {epoch + 1}: loss {avg:.4f}")
            mlflow.log_metric("train_loss", avg, step=epoch)

        np.save(MODEL_DIR / "tt_user_emb.npy", model.user_emb.weight.detach().cpu().numpy())
        np.save(MODEL_DIR / "tt_item_emb.npy", model.item_emb.weight.detach().cpu().numpy())


if __name__ == "__main__":
    run()
