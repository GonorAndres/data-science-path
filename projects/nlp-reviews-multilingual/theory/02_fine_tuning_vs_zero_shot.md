# Fine-tuning vs Zero-shot: Implementation Notes

## The tradeoff framed

For a text classification task in 2026, you have four viable paths:

1. **Train from scratch** on labeled data (classical ML or a small neural model).
2. **Fine-tune** a pretrained transformer on labeled data.
3. **Zero-shot** a large LLM (no training, just prompting).
4. **Few-shot** a large LLM (provide labeled examples in the prompt).

Accuracy, latency, cost, and control vary across these. This document explains why this project uses fine-tuning.

## Rough comparison on a similar task

Numbers vary by dataset; these are typical for short-text 3-class sentiment on Spanish + English reviews.

| Approach | Macro F1 | Training time | Inference latency | Monthly cost for 1M reviews |
|----------|----------|---------------|-------------------|-----------------------------|
| TF-IDF + LogReg | 0.74 | 5 min | 2 ms | ~$0 |
| Fine-tuned DistilBERT | 0.86 | 2 h GPU | 50 ms CPU | ~$20 compute |
| Fine-tuned XLM-R base | 0.88 | 4 h GPU | 200 ms CPU | ~$80 |
| Zero-shot GPT-4o-mini | 0.82 | 0 | 800 ms API | ~$300 |
| Few-shot GPT-4o-mini | 0.86 | 0 | 1200 ms API | ~$500 |
| Zero-shot Claude Sonnet | 0.87 | 0 | 1000 ms API | ~$1000 |
| Zero-shot Llama 3.1 8B (Ollama) | 0.80 | 0 | 2s local | ~$0 (local) |

## What fine-tuning buys you

- **Accuracy per dollar.** A fine-tuned DistilBERT matches a few-shot API model at a fraction of the inference cost.
- **Latency.** Fine-tuned models serve in tens of milliseconds; API models in hundreds.
- **Privacy and compliance.** The model and data stay on your infrastructure.
- **Determinism.** Same input gives same output (LLMs with nonzero temperature are stochastic).
- **Predictable cost.** Pay once (training) plus steady compute (inference); no per-request billing.

## What fine-tuning costs you

- **Data requirement.** Fine-tuning needs 1k to 100k labeled examples to work well. LLMs zero-shot work with none.
- **Engineering overhead.** Training loop, monitoring, retraining when distribution shifts.
- **Knowledge staleness.** The model knows what it was trained on. For factual questions, LLMs can be more current (RAG helps).
- **Less flexible.** A fine-tuned 3-class sentiment model cannot suddenly classify 5 classes without retraining.

For this project, we have 200k labeled examples per language, so fine-tuning is the right choice.

## Fine-tuning mechanics

### Hyperparameters

- **Learning rate.** 1e-5 to 5e-5 for encoder fine-tuning. 2e-5 is a safe default.
- **Batch size.** 16 to 64. Larger batches need careful learning rate tuning.
- **Epochs.** 2 to 5. Overfitting appears quickly on small datasets.
- **Warmup ratio.** 0.05 to 0.1.
- **Weight decay.** 0.01 is standard.
- **Max sequence length.** Match the 95th percentile of the input lengths, rounded up to a power of 2. For reviews, 128 is generous.

### Mixed precision

Enable `fp16` or `bf16` training. Roughly doubles throughput on modern GPUs with negligible accuracy impact.

### Gradient accumulation

If the GPU cannot hold the desired batch size, accumulate gradients across several smaller batches. `effective_batch = per_device_batch * grad_accum_steps * num_gpus`.

### Checkpointing

Save the best model by validation metric, not by last epoch. HuggingFace's `Trainer` has `load_best_model_at_end=True` for this.

### Frozen-then-unfrozen schedule

- Epoch 0: freeze the backbone, train only the classifier head.
- Epoch 1 onward: unfreeze everything.

This shortens initial training and prevents the randomly initialized head from gradient-bombing the pretrained weights. Optional for small datasets; useful for very small datasets.

### Regularization

- Dropout (default 0.1 in the model).
- Weight decay.
- Early stopping (patience 2 epochs).
- Data augmentation (back-translation or synonym replacement; optional, often overkill for this task).

## Baseline discipline

Always ship a baseline. The TF-IDF + LogReg baseline:

- Fits in 5 minutes on a laptop.
- Provides a floor the transformer must beat by a meaningful margin.
- Is interpretable out of the box (coefficients on words).
- Serves as a smoke test if the transformer training pipeline breaks.

Interviewers specifically ask for baselines. "I got 0.88 on the transformer" is a weaker answer than "The TF-IDF baseline was 0.74 and the transformer was 0.88; the 14-point gain justifies the added complexity."

## Error analysis workflow

After training:

1. Compute per-class precision and recall.
2. Inspect 20 misclassifications per class.
3. Group by error type: sarcasm, negation, typos, mixed sentiment.
4. Note which error types the baseline also makes (inherent hard examples) vs only the transformer makes (regressions to investigate).

This document discipline is its own recruiter signal.

## Inference-time tricks

For production serving:

- **Dynamic quantization** (PyTorch `torch.quantization.quantize_dynamic`). ~3x CPU speedup, ~1 percent F1 drop.
- **ONNX export.** Export the model to ONNX, serve via ONNX Runtime. ~2x faster in most cases.
- **Distillation.** Train a smaller student model to mimic the teacher's outputs. Adds work but shrinks inference cost.

The `mlops-model-service` project uses ONNX + dynamic quantization to hit sub-50 ms CPU latency.

## Monitoring in production

After deployment, monitor:

- **Latency.** P50, P95, P99.
- **Request rate.** Traffic profile.
- **Prediction distribution.** If the proportion of negatives spikes, either the world changed or the model drifted.
- **Input distribution.** Feature drift signals (review length, token distribution).
- **Error rate.** Calibration checks if ground truth is available later.

See `docs/theory/13_mlops_architecture_patterns.md` for the broader framing.

## When to revisit the choice

Consider switching to LLM zero-shot if:

- Labels become scarce (concept drift makes old labels stale).
- Task specification changes frequently (3 classes today, 7 tomorrow).
- Accuracy needs increase beyond what fine-tuning delivers (GPT-5 class models will close the gap further by 2027).

Consider training from scratch if:

- Domain is extremely specialized (rare technical jargon, non-standard language).
- Privacy constraints preclude using pretrained weights.

Neither applies for this project. Fine-tuning stays the right choice.

## Linking to the code

- `src/baseline.py`: TF-IDF + LogReg.
- `src/train.py`: transformer fine-tuning via HuggingFace Trainer.
- `src/evaluate.py`: confusion matrix, per-class metrics, error analysis dump.
- `src/predict.py`: single-input inference for the demo.
- `notebooks/02_baseline.md` and `notebooks/03_transformer.md`: storytelling notebooks that walk through both.
