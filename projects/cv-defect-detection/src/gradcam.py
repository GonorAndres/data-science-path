"""Generate a Grad-CAM overlay for a single sample."""

import argparse
from pathlib import Path

import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from .data import MVTecDataset, eval_transforms, labels_for_category
from .model import build_model


MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def run(category: str, sample: int) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = labels_for_category(category)
    ds = MVTecDataset(category, "test", labels, eval_transforms())
    x, y = ds[sample]
    x = x.unsqueeze(0).to(device)

    ckpt = torch.load(MODEL_DIR / f"{category}.pt", map_location=device)
    model = build_model(len(labels)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    target_layer = model.layer4[-1] if hasattr(model, "layer4") else list(model.children())[-2]
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale = cam(input_tensor=x, targets=[ClassifierOutputTarget(int(y))])[0]

    img = x.squeeze(0).cpu().permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    overlay = show_cam_on_image(img, grayscale, use_rgb=True)

    path = OUT_DIR / f"gradcam_{category}_{sample}.png"
    Image.fromarray(overlay).save(path)
    print(f"true label: {labels[y]}; saved {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True)
    parser.add_argument("--sample", type=int, default=0)
    args = parser.parse_args()
    run(args.category, args.sample)
