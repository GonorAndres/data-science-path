# CNN Intuition

For the full theory, see `docs/theory/10_computer_vision_foundations.md`. This doc focuses on the working intuition needed to read and tune this project.

## Convolutions as local pattern detectors

A convolutional filter slides over the image. Early filters detect edges and colors; deeper filters detect textures, parts, and eventually object categories. This emerges naturally during training; we do not specify it.

For a defect detection task, the intuition is:

- Early layers see surface texture.
- Middle layers see micro-patterns like scratches, dents, discolorations.
- Late layers combine these into "defective" vs "normal" representations.

## Why ResNet50 is a strong default

- 25M parameters, proven on ImageNet.
- Pretrained weights are stable.
- `timm` loads it in one line.
- 50 ms inference on CPU, 2 ms on GPU.

Alternatives:

- **EfficientNet-B0.** Similar accuracy, faster on mobile.
- **ConvNeXt-Tiny.** More modern; often slightly better.
- **ViT-Base.** Attention-based; best with large datasets.

## Input pipeline

- Resize to 224x224 (or 256x256 with center crop).
- Normalize with ImageNet mean and std.
- Data augmentations at training only.
- ToTensor.

## Output head

For K-class classification:

```python
model = timm.create_model("resnet50", pretrained=True, num_classes=K)
```

`timm` replaces the final ImageNet 1000-class head with one of size K automatically.

## Training loop

Standard PyTorch training with:

- Cross-entropy loss.
- AdamW optimizer.
- Cosine learning rate schedule.
- Early stopping on validation macro F1.
- Save best checkpoint.

## Cross-links

- `02_transfer_learning.md`
- `03_data_augmentation.md`
- `docs/theory/10_computer_vision_foundations.md`
