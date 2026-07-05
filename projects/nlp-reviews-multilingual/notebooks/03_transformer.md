# Notebook 03: Transformer Fine-tuning (outline)

Convert to `.ipynb` when running the project.

## Sections

### 1. Model choice rationale

Why multilingual DistilBERT. Link to `theory/01_transformers_and_tokenization.md`.

### 2. Tokenizer demo

Tokenize a few example reviews in ES and EN. Show the token-count asymmetry.

### 3. Training setup

Hyperparameters table. MLflow experiment name. Training time estimate.

### 4. Training run

Call `src.train.train()` or inline the HuggingFace Trainer.

### 5. Validation curves

Loss vs epoch. Macro F1 vs epoch. Confirm no overfitting.

### 6. Test-set evaluation

Run `src.evaluate.evaluate()`. Report:

- Overall macro F1.
- Per-language macro F1.
- Per-class precision and recall.
- Confusion matrix heatmap.

### 7. Comparison vs baseline

Side-by-side table: baseline vs transformer on all metrics. Quantify the gain.

### 8. Interpretability

Attention or LIME highlights on 5 example reviews. Discuss what the model looks at.

### 9. Takeaways

- Achieved metrics.
- Where the transformer beats the baseline and where it does not.
- What to improve in v2.
