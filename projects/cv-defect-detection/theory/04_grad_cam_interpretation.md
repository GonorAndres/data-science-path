# Grad-CAM for CV Interpretability

## The problem

A black-box CNN that predicts "defective" gives no indication of where the defect is. For quality inspection, that is a problem. Operators need to see the evidence.

## How Grad-CAM works

At a high level:

1. Forward pass the image through the CNN. Take the last convolutional feature maps $A^k$.
2. Backpropagate the predicted class's score with respect to $A^k$. The average gradient per feature map gives weights $\alpha_k$.
3. Weighted sum of feature maps: $L_{\text{Grad-CAM}} = \text{ReLU}\left(\sum_k \alpha_k A^k\right)$.
4. Upsample the map to input resolution and overlay as a heatmap.

The ReLU keeps only positive contributions (regions supporting the class).

## Why it is intuitive

It answers "where did the model look?" Without needing modifications to the model. Works for any CNN.

## Caveats

- **Coarse.** Heatmap resolution is the final feature map's (7x7 for ResNet50 at 224 input). Fine details blur.
- **Class-specific.** You pick which class to explain; different classes give different heatmaps.
- **Can miss.** Sometimes the model uses features Grad-CAM does not highlight (spurious correlations).
- **Does not imply causation.** The heatmap shows what influenced the output; it does not prove that the highlighted region is actually the defect.

## Variants

- **Grad-CAM++.** More fine-grained localization.
- **Score-CAM.** No gradients required; uses forward-pass scores only.
- **Ablation-CAM.** Slower but more principled.

For this project, vanilla Grad-CAM is enough.

## Implementation

Libraries: `pytorch-grad-cam` (github.com/jacobgil/pytorch-grad-cam) is the go-to.

```python
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

target_layer = model.layer4[-1]
cam = GradCAM(model=model, target_layers=[target_layer])
grayscale_cam = cam(input_tensor=img, targets=[ClassifierOutputTarget(class_idx)])[0]
```

## Use in the Gradio demo

Show the input image next to a heatmap overlay. Include the predicted class and confidence. Users can immediately see "the model saw a scratch here, which is why it said scratched."

## Cross-links

- `docs/theory/06_model_interpretability_shap_lime.md`
- `docs/theory/10_computer_vision_foundations.md`
- `03_data_augmentation.md`
