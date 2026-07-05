"""GCN on a toy citation-like graph using PyTorch Geometric."""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


def make_graph():
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 4, 5],
        [1, 0, 2, 1, 3, 2, 5, 4],
    ], dtype=torch.long)
    x = torch.eye(6, dtype=torch.float)
    y = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    train_mask = torch.tensor([True, False, True, False, True, False])
    return Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask)


class GCN(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, out_dim)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        return self.conv2(h, edge_index)


def main() -> None:
    data = make_graph()
    model = GCN(6, 8, 3)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(60):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        preds = model(data.x, data.edge_index).argmax(dim=-1)
    print("predictions:", preds.tolist())
    print("labels:     ", data.y.tolist())


if __name__ == "__main__":
    main()
