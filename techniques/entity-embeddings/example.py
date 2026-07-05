"""Entity embeddings + MLP for a toy categorical prediction task."""

import torch
import torch.nn as nn


class EmbedMLP(nn.Module):
    def __init__(self, cardinalities: list[int], embed_dims: list[int], out_dim: int):
        super().__init__()
        self.embeds = nn.ModuleList([
            nn.Embedding(c, d) for c, d in zip(cardinalities, embed_dims)
        ])
        total = sum(embed_dims)
        self.mlp = nn.Sequential(
            nn.Linear(total, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim),
        )

    def forward(self, cats):
        parts = [emb(cats[:, i]) for i, emb in enumerate(self.embeds)]
        x = torch.cat(parts, dim=-1)
        return self.mlp(x)


def main() -> None:
    torch.manual_seed(0)
    n = 500
    c1 = torch.randint(0, 10, size=(n,))
    c2 = torch.randint(0, 50, size=(n,))
    y = ((c1 % 2) + (c2 % 3) > 2).long()
    cats = torch.stack([c1, c2], dim=-1)

    model = EmbedMLP([10, 50], [4, 8], out_dim=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(20):
        logits = model(cats)
        loss = loss_fn(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        acc = (logits.argmax(-1) == y).float().mean().item()
        if (epoch + 1) % 5 == 0:
            print(f"epoch {epoch + 1}: loss {loss.item():.4f}  acc {acc:.3f}")


if __name__ == "__main__":
    main()
