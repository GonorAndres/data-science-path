"""Tiny CNN on synthetic 28x28 images."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.flatten(1)
        return self.fc(x)


def main() -> None:
    torch.manual_seed(0)
    x = torch.randn(200, 1, 28, 28)
    y = torch.randint(0, 3, size=(200,))
    model = SmallCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(5):
        logits = model(x)
        loss = loss_fn(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        acc = (logits.argmax(-1) == y).float().mean().item()
        print(f"epoch {epoch + 1}: loss {loss.item():.4f}  acc {acc:.3f}")


if __name__ == "__main__":
    main()
