---
tags: [deep-learning, cnn, vision, pytorch]
status: review
created: 2026-04-23
updated: 2026-04-25
---

# Convolutional Neural Networks

## What it is

Neural networks built around convolutional layers: local, weight-shared linear maps applied across spatial positions. Originally designed for images (LeCun 1989, AlexNet 2012); adapted to audio spectrograms, 1D signals, and even text in early NLP. Even after Vision Transformers (ViT) demonstrated transformer dominance on large data, CNNs remain competitive on small to medium data and are the standard for many production vision tasks.

For deep-learning fundamentals (backprop, normalization, initialization), see `docs/theory/17_dl_fundamentals.md`. For broader vision context (object detection, segmentation, VLMs), see `docs/theory/10_computer_vision_foundations.md`.

## Why it matters

CNNs are still the workhorse for production vision systems where data is limited, labeling is expensive, or compute is constrained. Mexican retail and insurance use cases (defect detection, document scanning, satellite imagery, license-plate recognition) typically run CNN-family models on edge or near-edge hardware. Understanding the math, the training recipe, and the canonical architectures is non-negotiable for any vision-adjacent role.

## When to use it

- Image classification, object detection, segmentation.
- Audio (mel spectrograms treated as 2D images).
- Structured 1D signals where local patterns matter (ECG, accelerometer).
- Small data plus transfer learning from ImageNet pretrained backbones.
- Edge deployment where transformer self-attention is too memory-hungry.

## Convolution math

A 2D convolution (technically cross-correlation in the deep-learning convention) maps an input feature map $X \in \mathbb{R}^{H \times W \times C_{\mathrm{in}}}$ through a kernel $K \in \mathbb{R}^{k_h \times k_w \times C_{\mathrm{in}} \times C_{\mathrm{out}}}$:

$$
Y_{i, j, c_{\mathrm{out}}} = \sum_{c_{\mathrm{in}}=1}^{C_{\mathrm{in}}} \sum_{a=0}^{k_h - 1} \sum_{b=0}^{k_w - 1} X_{i + a,\, j + b,\, c_{\mathrm{in}}} \cdot K_{a, b, c_{\mathrm{in}}, c_{\mathrm{out}}} + b_{c_{\mathrm{out}}}.
$$

The kernel is the same across spatial positions (weight sharing); it is different across input and output channels.

### Output size formula

For input width $W$, kernel size $K$, padding $P$, and stride $S$:

$$
W_{\mathrm{out}} = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1.
$$

Same applies to height. Memorize this; you will compute it on every architecture design or debugging session.

### Worked example

Input $32 \times 32$, kernel $3 \times 3$, padding 1, stride 1: $W_{\mathrm{out}} = (32 - 3 + 2) / 1 + 1 = 32$. Spatial dimensions preserved. This is the canonical "same" convolution.

Input $32 \times 32$, kernel $3 \times 3$, padding 1, stride 2: $W_{\mathrm{out}} = (32 - 3 + 2)/2 + 1 = 16$. Halves spatial dimensions; replaces a max-pool layer in modern architectures.

## Receptive field arithmetic

The receptive field of a unit is the region of the input it depends on. For stacked convolutions, the receptive field grows with depth.

For a sequence of layers with kernel sizes $k_l$ and strides $s_l$:

$$
R_L = 1 + \sum_{l=1}^{L} (k_l - 1) \prod_{j=1}^{l-1} s_j.
$$

Example: three stacked $3\times3$ stride-1 convolutions give receptive field $1 + 2 + 2 + 2 = 7$, equivalent to a single $7\times7$ convolution but with fewer parameters and more non-linearities.

For dense-prediction tasks (segmentation, detection), the receptive field must cover objects of interest. Architectures like DeepLab use atrous (dilated) convolutions to expand receptive field without losing resolution.

## Pooling vs strided convolution

Classic CNNs (VGG) used max-pool layers to downsample. Modern architectures (ResNet, ConvNeXt) increasingly prefer strided convolution because:

- Strided conv learns the downsampling filter rather than fixing it to max.
- Backward pass is smoother (max-pool's gradient is sparse, only flowing to the argmax).
- Combines downsampling and feature extraction in one op, fewer parameters and faster.

Max-pooling is still occasionally used for spatial-pyramid features and in some small-mobile architectures.

## Modern architectural patterns

### Skip (residual) connections

He et al. (2015). Add the input of a block to its output: $y = x + F(x)$. Enables training of very deep networks (50 to 200+ layers). Gradient flows through the identity path; addresses vanishing gradients structurally. Standard since 2015 in nearly every CNN family.

### Depthwise separable convolution

MobileNet (Howard et al. 2017). Factor a $k \times k \times C_{\mathrm{in}} \times C_{\mathrm{out}}$ convolution into:

1. **Depthwise.** $k \times k \times 1 \times 1$ applied per input channel. $k^2 C_{\mathrm{in}}$ parameters.
2. **Pointwise.** $1 \times 1 \times C_{\mathrm{in}} \times C_{\mathrm{out}}$. $C_{\mathrm{in}} C_{\mathrm{out}}$ parameters.

Total: $k^2 C_{\mathrm{in}} + C_{\mathrm{in}} C_{\mathrm{out}}$ vs $k^2 C_{\mathrm{in}} C_{\mathrm{out}}$ for standard convolution. About 8 to 9x cheaper for $k=3$, $C_{\mathrm{out}}=64$. Standard in mobile and edge models.

### Inverted residual

MobileNetV2 (Sandler et al. 2018). The block: $1\times1$ expansion to high-dim, $3\times3$ depthwise, $1\times1$ projection back to low-dim, residual added on the low-dim path. Works well with separable convs because depthwise is cheap in high-dim space.

### Channel attention (Squeeze-and-Excitation)

Hu et al. (2018). After a convolution block, squeeze across spatial dims (global avg pool), pass through a small MLP (FC, ReLU, FC, sigmoid), and use the result to rescale channels:

$$
y = x \cdot \sigma(W_2 \mathrm{ReLU}(W_1 \mathrm{GAP}(x))).
$$

Cheap (few extra parameters) and adds 1 to 2% top-1 ImageNet accuracy on most backbones. Used in EfficientNet and many other modern architectures.

### Self-attention in CNNs

ConvNeXt (Liu et al. 2022) showed that with the right training recipe, pure CNNs match Vision Transformers. The architectural touch-ups: depthwise $7\times7$ kernels, LayerNorm instead of BatchNorm, GeLU activation, fewer activation layers per block.

## Canonical architectures (param counts)

| Family | Variant | Params | Top-1 ImageNet |
|--------|---------|--------|---------------|
| ResNet | R-50 | 25M | 76% |
| ResNet | R-152 | 60M | 78% |
| EfficientNet | B0 | 5.3M | 77% |
| EfficientNet | B7 | 66M | 84% |
| ConvNeXt | Tiny | 28M | 82% |
| ConvNeXt | Large | 198M | 86% |
| MobileNetV3 | Small | 2.5M | 67% |
| MobileNetV3 | Large | 5.4M | 75% |

For a Mexican-edge use case (in-store cameras, mobile apps), MobileNetV3 or EfficientNet-B0 are pragmatic starting points. For accuracy-first cloud-served tasks, ConvNeXt-Base or EfficientNet-B4 to B7.

## Training recipe (modern)

The 2026 canonical recipe for fine-tuning a pretrained CNN:

```python
import torch
import torchvision
from torch.optim.lr_scheduler import CosineAnnealingLR

model = torchvision.models.convnext_tiny(weights="DEFAULT")
# replace classifier head
model.classifier[2] = torch.nn.Linear(768, num_classes)

# label smoothing in the loss
loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

# AdamW with weight decay; exclude bias and norm params from decay
decay, no_decay = [], []
for name, p in model.named_parameters():
    if p.dim() == 1 or "bias" in name:
        no_decay.append(p)
    else:
        decay.append(p)
opt = torch.optim.AdamW(
    [
        {"params": decay, "weight_decay": 0.05},
        {"params": no_decay, "weight_decay": 0.0},
    ],
    lr=1e-3,
)

# linear warmup for first 5% then cosine
total_steps = epochs * len(loader)
warmup_steps = total_steps // 20
scheduler = ... # use timm.scheduler.CosineLRScheduler or write custom
```

Augmentation: `RandAugment` or `TrivialAugment` from torchvision; mixup or cutmix for additional regularization at large data scale. Test-time augmentation (TTA) at inference: average predictions over horizontal flips and a few crops.

## Transfer learning in practice

The two-stage protocol:

1. **Linear probe.** Freeze the backbone, train only a new classification head for 5 to 10 epochs at LR 1e-3.
2. **Unfreeze and fine-tune.** Unfreeze the entire backbone, train at a much lower LR (1e-4 or 1e-5) for the head and even lower for the backbone, often with layer-wise LR decay.

### Layer-wise LR decay

Earlier layers learn generic features (edges, textures) and need less adjustment; later layers are task-specific. Apply LR multipliers $\eta \cdot \gamma^{L - l}$ at layer $l$ with depth $L$ and decay factor $\gamma \in [0.5, 0.95]$. Standard in ConvNeXt and ViT fine-tuning.

### When to use linear probe alone

- Very small downstream dataset (under 1,000 images per class).
- Computational budget is tight.
- Domain is close to the pretraining domain (natural images for ImageNet pretraining).

For substantially different domains (medical imaging, satellite, document scans), full fine-tuning with low LR almost always wins.

## Gotchas & lessons learned

- **Forgetting `model.eval()` at inference.** Batch norm uses batch statistics in train mode, running stats in eval mode. Predictions on a single image in train mode are garbage.
- **Wrong normalization.** Pretrained ImageNet models expect inputs normalized with specific mean/std (typically `(0.485, 0.456, 0.406)` and `(0.229, 0.224, 0.225)`). Using raw 0-255 inputs gives unusable outputs.
- **Tiny batch size with BatchNorm.** BN with batch size 1 or 2 is unstable. Use GroupNorm or LayerNorm for very small batches.
- **Image format.** PIL gives `(H, W, C)`, PyTorch expects `(C, H, W)`. `transforms.ToTensor()` handles this; manual numpy conversions silently break.
- **Class imbalance.** Standard cross-entropy with severe imbalance underweights minority classes. Use `WeightedRandomSampler`, focal loss, or class-balanced sampling.
- **Aspect ratio.** Resize to a square crops or distorts; for tasks where shape matters (medical, scientific), use letterbox padding instead.
- **TTA disabled by default.** TTA improves accuracy 0.5 to 1% but doubles or quadruples inference cost. Enable it explicitly when accuracy is the priority.

## Resources

- `docs/theory/10_computer_vision_foundations.md`.
- `docs/theory/17_dl_fundamentals.md`.
- Krizhevsky, Sutskever, Hinton, "ImageNet Classification with Deep Convolutional Neural Networks", NeurIPS 2012.
- He, Zhang, Ren, Sun, "Deep Residual Learning for Image Recognition", CVPR 2016.
- Howard et al., "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications", 2017.
- Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks", CVPR 2018.
- Hu, Shen, Sun, "Squeeze-and-Excitation Networks", CVPR 2018.
- Tan, Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks", ICML 2019.
- Liu et al., "A ConvNet for the 2020s" (ConvNeXt), CVPR 2022.
- Cubuk et al., "RandAugment: Practical Automated Data Augmentation", NeurIPS 2020.
- torchvision and timm libraries (modern pretrained backbones with current recipes).
