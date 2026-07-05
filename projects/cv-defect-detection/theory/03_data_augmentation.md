# Data Augmentation

## Why augmentation matters here

MVTec has 5 to 100 samples per defect class. Without augmentation, a 25M-parameter CNN memorizes in a few epochs. Augmentation multiplies the effective dataset.

## Augmentations that typically help

- **Random horizontal flip.** Safe for most defects.
- **Random vertical flip.** Depends on object orientation; e.g., bottles look wrong upside-down. Apply selectively.
- **Random rotation (+/- 15 to 45 deg).** Helps if cameras have variable orientation.
- **Random crop + resize.** Simulates camera framing variation.
- **Color jitter.** Brightness, contrast, saturation +/- 20%. Helps across lighting.
- **Gaussian noise.** Simulates sensor noise.
- **CoarseDropout / CutOut.** Random rectangles zeroed; forces the model to use spatial context.

Libraries: `albumentations` is faster and richer than `torchvision.transforms`. Default to `albumentations`.

## Augmentations that can hurt

- Aggressive geometric warps on rigid objects (pills, screws): destroys shape.
- Large color shifts when color is the defect signal.
- Mixup on defect detection: blends defective and normal images. Usually not helpful.
- CutMix on objects where the defect is small: may cut out the defect.

Experiment; verify.

## Test-time augmentation (TTA)

At inference, average predictions across multiple augmented versions of the same image (horizontal flip, slight rotation). Modest accuracy gain at N times the inference cost. Useful at final evaluation; not in real-time inference.

## Implementation pattern

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_tf = A.Compose([
    A.RandomResizedCrop(256, 256, scale=(0.8, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.ColorJitter(0.2, 0.2, 0.2, 0.1, p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_tf = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
```

## Cross-links

- `02_transfer_learning.md`
- `04_grad_cam_interpretation.md`
- `docs/theory/10_computer_vision_foundations.md`
