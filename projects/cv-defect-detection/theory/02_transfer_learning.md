# Transfer Learning

## The hypothesis

A model trained on ImageNet has learned generic visual features. For a new task, we do not need to relearn edges and textures; we can adapt the existing features.

## Two strategies

### 1. Feature extraction

Freeze the backbone; train only a new head.

```python
for p in model.parameters():
    p.requires_grad = False
model.head = nn.Linear(model.num_features, num_classes)
```

Fast. Works when the downstream task is close to ImageNet (natural photos of objects).

### 2. Fine-tuning

Unfreeze all or some layers; train with a small learning rate.

- **All unfrozen.** Default modern approach. Most accurate with enough data.
- **Gradual unfreezing.** Unfreeze only the top N blocks; train; unfreeze more; repeat. Stabilizes training.

For this project, we default to full fine-tuning with weight decay and early stopping.

## Learning rate considerations

- **Head only.** ~1e-3.
- **Fine-tuning all.** ~1e-4 for the backbone, ~1e-3 for the head. Different LRs for different parameter groups is common; PyTorch supports via optimizer param groups.

## When transfer learning helps less

- Non-natural imagery (medical scans, thermal, microscopy) where ImageNet features transfer poorly. Consider self-supervised pretraining on in-domain data.
- Extremely small datasets (< 100 per class). Transfer helps but results are still limited.
- Domain shift between training and deployment (lab vs factory floor).

## Fine-tuning tips

- Preserve ImageNet normalization stats during input preprocessing.
- Lower learning rate; higher risk of disrupting pretrained weights.
- Monitor overfitting early (small dataset).
- Augment aggressively.

## Cross-links

- `01_cnns_intuition.md`
- `03_data_augmentation.md`
- `docs/theory/10_computer_vision_foundations.md`
