"""Feature extraction with a pretrained ResNet18 head on synthetic images."""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
    for p in backbone.parameters():
        p.requires_grad = False

    in_features = backbone.fc.in_features
    backbone.fc = nn.Linear(in_features, 5)
    model = backbone.to(device)

    opt = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    x = torch.randn(8, 3, 224, 224).to(device)
    y = torch.randint(0, 5, size=(8,)).to(device)

    for epoch in range(3):
        logits = model(x)
        loss = loss_fn(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"epoch {epoch + 1}: loss {loss.item():.4f}")


if __name__ == "__main__":
    main()
