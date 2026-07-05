"""PyTorch Dataset for MVTec AD defect classification."""

from pathlib import Path

import albumentations as A
import numpy as np
from PIL import Image
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "mvtec_ad"


def list_categories() -> list[str]:
    if not DATA_DIR.exists():
        return []
    return sorted([p.name for p in DATA_DIR.iterdir() if p.is_dir()])


def _image_records(category: str, split: str) -> list[tuple[Path, str]]:
    root = DATA_DIR / category / split
    if not root.exists():
        raise FileNotFoundError(f"{root} not found")
    records = []
    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue
        for p in sorted(subdir.glob("*.png")):
            records.append((p, subdir.name))
    return records


def labels_for_category(category: str) -> list[str]:
    records = _image_records(category, "test")
    return sorted(set(label for _, label in records))


class MVTecDataset(Dataset):
    def __init__(self, category: str, split: str, labels: list[str], transform):
        self.records = _image_records(category, split)
        self.labels = labels
        self.label_to_id = {lb: i for i, lb in enumerate(labels)}
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx):
        path, label = self.records[idx]
        img = np.array(Image.open(path).convert("RGB"))
        transformed = self.transform(image=img)
        return transformed["image"], self.label_to_id[label]


def train_transforms():
    return A.Compose([
        A.RandomResizedCrop(256, 256, scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.ColorJitter(0.2, 0.2, 0.2, 0.1, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def eval_transforms():
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
