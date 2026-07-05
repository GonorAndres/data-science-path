"""Minimal end-to-end PyTorch training loop on a synthetic classification task."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def make_data(n: int = 2000, d: int = 10, k: int = 3, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    w = torch.randn(d, k, generator=g)
    logits = x @ w
    y = logits.argmax(dim=-1)
    return x, y


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y = make_data()
    split = int(0.8 * len(x))
    train_ds = TensorDataset(x[:split], y[:split])
    val_ds = TensorDataset(x[split:], y[split:])
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=128)

    model = MLP(10, 32, 3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(20):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(dim=-1)
                correct += int((preds == yb).sum().item())
                total += yb.numel()
        print(f"epoch {epoch + 1}: val acc {correct / total:.3f}")


if __name__ == "__main__":
    main()
