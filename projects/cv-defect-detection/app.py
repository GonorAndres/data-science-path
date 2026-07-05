"""Gradio demo. Upload an image, see prediction and Grad-CAM overlay."""

from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.data import eval_transforms, labels_for_category, list_categories
from src.model import build_model


MODEL_DIR = Path(__file__).parent / "models"


def predict(image: Image.Image, category: str):
    if image is None or category is None:
        return None, "Select a category and upload an image."
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = labels_for_category(category)
    transform = eval_transforms()
    arr = np.array(image.convert("RGB"))
    t = transform(image=arr)["image"].unsqueeze(0).to(device)

    ckpt = torch.load(MODEL_DIR / f"{category}.pt", map_location=device)
    model = build_model(len(labels)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    with torch.no_grad():
        logits = model(t)[0]
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    idx = int(np.argmax(probs))

    target_layer = model.layer4[-1] if hasattr(model, "layer4") else list(model.children())[-2]
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale = cam(input_tensor=t, targets=[ClassifierOutputTarget(idx)])[0]

    img = t.squeeze(0).cpu().permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    overlay = show_cam_on_image(img, grayscale, use_rgb=True)

    label_text = f"Predicted: {labels[idx]} ({probs[idx]:.2f})"
    return Image.fromarray(overlay), label_text


with gr.Blocks(title="CV Defect Detection") as demo:
    gr.Markdown("# CV Defect Detection (MVTec AD)")
    with gr.Row():
        image_in = gr.Image(label="Input", type="pil")
        category_in = gr.Dropdown(choices=list_categories(), label="Category")
    btn = gr.Button("Analyze")
    cam_out = gr.Image(label="Grad-CAM overlay")
    label_out = gr.Textbox(label="Prediction")
    btn.click(predict, inputs=[image_in, category_in], outputs=[cam_out, label_out])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)
