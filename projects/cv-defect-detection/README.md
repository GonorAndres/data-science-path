# cv-defect-detection

## Resumen (ES)

**Problema:** clasificacion de defectos en imagenes industriales. Dado una imagen, identificar si tiene defecto y, si si, de que clase.

**Metodo:** transfer learning con un backbone preentrenado (ResNet50 o ConvNeXt-tiny) via `timm`. Fine-tuning en un dataset publico (MVTec AD). Grad-CAM para interpretabilidad visual.

**Resultado objetivo:** F1 macro mayor a 0.85 por categoria, heatmaps de Grad-CAM interpretables, demo Gradio local.

## Summary (EN)

**Problem:** defect classification on industrial images. Given an image, identify whether it is defective and, if so, which class of defect.

**Method:** transfer learning with a pretrained backbone (ResNet50 or ConvNeXt-tiny) via `timm`. Fine-tune on a public dataset (MVTec AD). Grad-CAM for visual interpretability.

**Target result:** macro F1 above 0.85 per category, interpretable Grad-CAM heatmaps, local Gradio demo.

---

## Dataset

MVTec Anomaly Detection (https://www.mvtec.com/company/research/datasets/mvtec-ad). 15 object/texture categories, train set has only normal samples; test set has normal + defective with pixel-level masks.

For a classification framing, we use the test set's defect-type labels: within each category, predict the defect class.

## Structure

```
cv-defect-detection/
├── README.md
├── theory/
│   ├── 00_problem_framing.md
│   ├── 01_cnns_intuition.md
│   ├── 02_transfer_learning.md
│   ├── 03_data_augmentation.md
│   └── 04_grad_cam_interpretation.md
├── data_download.py
├── src/
│   ├── __init__.py
│   ├── data.py              # PyTorch Dataset + augmentations
│   ├── model.py             # timm backbone + head
│   ├── train.py
│   ├── evaluate.py
│   ├── gradcam.py
│   └── predict.py
├── tests/
├── notebooks/
├── app.py                   # Gradio demo
├── requirements.txt
└── Makefile
```

## Quickstart

```bash
pip install -r requirements.txt
python data_download.py      # places MVTec AD under data/
python -m src.train --category bottle
python -m src.evaluate --category bottle
python -m src.gradcam --category bottle --sample 0
python app.py
```

## Decisions and trade-offs

| Decision | Alternative | Why |
|----------|-------------|-----|
| Classification framing | Anomaly detection / segmentation | Simpler portfolio scope; segmentation is a follow-on. |
| `timm` ResNet50 | ViT-B/16, ConvNeXt-T | ResNet50 is cheap, strong, widely used; other options documented. |
| Grad-CAM | Integrated Gradients, LIME | Grad-CAM is the standard CNN interpretability tool; fast and visually intuitive. |
| MVTec AD | Kaggle surface defects | Canonical academic benchmark; label quality good. |

## References

- `docs/theory/10_computer_vision_foundations.md`
- Bergmann et al., "MVTec AD: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection", CVPR 2019.
- Selvaraju et al., "Grad-CAM", 2017.
- `timm` library: github.com/huggingface/pytorch-image-models.
