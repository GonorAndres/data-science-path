# Problem Framing: Industrial Defect Detection

## The business problem

Factories inspect products during and after manufacturing. Human inspectors are slow, expensive, and inconsistent. Automated vision systems:

- Inspect every unit (100 percent coverage).
- Apply consistent criteria.
- Collect data for process improvement.

Use cases:

- Cosmetic defects on consumer goods.
- Surface defects on metal, wood, plastic.
- Contamination detection on food.
- Medical device inspection.

Economic value is significant: a single false-negative defect caught in the field costs 10x to 100x more than catching it on the line.

## Problem framing

Three adjacent problems:

1. **Anomaly detection (one-class).** Train on normal-only; flag anything unusual.
2. **Multi-class classification.** Given examples of each defect type, predict class.
3. **Semantic segmentation.** Predict per-pixel whether defective.

This project tackles problem 2 (classification). It is the simplest framing that still demonstrates transfer learning, augmentation, and interpretability.

Anomaly detection is a follow-on if time permits.

## Stakeholders

- **Manufacturing operations.** Wants high recall (no missed defects) at acceptable precision (not over-flagging).
- **Quality control.** Wants interpretable outputs (why did the model flag this unit?).
- **Plant engineering.** Wants integration with existing vision infrastructure.

## Non-goals

- Not deploying to real factory cameras.
- Not handling temporal (video) inputs.
- Not doing 3D reconstruction.
- Not training from scratch (transfer learning is the point).

## Success metrics

- **Macro F1.** Per category; target > 0.85.
- **Per-class recall.** Target > 0.90 for each defect class (missed defects are costly).
- **Latency.** Under 200 ms per image on CPU.
- **Interpretability.** Grad-CAM heatmaps visibly highlight the defect region on at least 80 percent of correct predictions.

## Risks

- **Small per-class samples.** MVTec has 5 to 100 defective samples per class. Heavy augmentation required.
- **Class imbalance.** Normal class usually dominates.
- **Domain shift.** Model trained on clean lab images may not transfer to factory-floor images.
- **Anomaly types in production unseen in training.** Fundamental limitation of classification framing.

## Cross-links

- `docs/theory/10_computer_vision_foundations.md`
- `01_cnns_intuition.md`
