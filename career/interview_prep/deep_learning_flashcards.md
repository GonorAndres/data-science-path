# Deep Learning Flashcards

Deep learning questions skew heavier in ML Engineer loops and at product companies (Mercado Libre, Nu, Salesforce, Google). Even junior DS interviews include a few conceptual DL questions.

---

## Foundations

### Q1. Why do we use activation functions?

**Crisp:** Without them, a stacked linear network collapses to a single linear transformation. Activations introduce non-linearity so deeper networks can represent richer functions.

**Common choices:** ReLU (default, cheap), GELU (used in transformers), Sigmoid / Tanh (older, for bounded outputs or gates), SiLU / Swish (smooth variants).

### Q2. Why ReLU dominated sigmoid.

**Crisp:** Sigmoid saturates at the extremes, leading to vanishing gradients. ReLU has constant gradient 1 for positive inputs, which trains faster.

**Caveat:** ReLU has dying-neuron problem (negative weights kill the unit). LeakyReLU and ELU variants address it.

### Q3. Softmax and cross-entropy.

**Crisp:** Softmax converts logits to a probability distribution. Cross-entropy penalizes the negative log probability of the true class. Their gradient simplifies to `(predicted - target)`, which is elegant and stable.

### Q4. Logits vs probabilities.

**Crisp:** Logits are unbounded real numbers (raw scores). Probabilities are [0,1]. Loss functions like `nn.CrossEntropyLoss` in PyTorch expect logits, not probabilities, because applying softmax inside the loss is numerically more stable.

---

## Optimization

### Q5. Why does gradient descent converge?

**Crisp:** In convex problems with appropriate step size, each step reduces the loss. In non-convex deep networks, it converges to a local minimum or saddle point; the empirical finding is that these are usually good enough.

### Q6. SGD vs Adam.

**Crisp:** SGD is per-parameter, same learning rate; with momentum it smooths updates. Adam adapts per-parameter via moving averages of gradient and squared gradient.

**When to use which:** Transformers and language models usually use AdamW. Image classification competitions often use SGD with momentum for best final accuracy. Use Adam as default if unsure.

### Q7. Learning rate schedulers.

**Crisp:** Constant LR is suboptimal. Common schedules: step decay (divide by 10 at milestones), cosine annealing, warmup + decay, OneCycleLR.

**Rule of thumb:** Warmup for the first few percent of training helps stabilize especially in transformers.

### Q8. Gradient clipping.

**Crisp:** Clip the gradient norm to prevent exploding updates. Use `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)`.

**When needed:** RNNs, transformers trained with high learning rates, and any model where losses spike occasionally.

---

## Regularization and training tricks

### Q9. Why batch normalization helps.

**Crisp:** Normalizes activations per mini-batch, reducing internal covariate shift, allowing higher learning rates, adding a small amount of noise that regularizes.

**Gotcha:** Behaves differently at train vs eval time (running mean/var at eval). Forgetting to call `.eval()` is a classic bug.

### Q10. Layer norm vs batch norm.

**Crisp:** Batch norm normalizes over the batch dimension; layer norm over the feature dimension. Batch norm is standard for CNNs; layer norm is standard for transformers because batch norm fails on variable-length sequences.

### Q11. Dropout mechanics.

**Crisp:** During training, randomly zero out a fraction of activations. At test time, use all activations but scale by (1 - dropout_p) to keep expectation consistent. Modern frameworks handle the scaling.

### Q12. Data augmentation for images.

**Crisp:** Random crops, flips, color jitter, rotation, mixup, cutmix. Augmentation acts as a regularizer by making the input distribution richer.

### Q13. Early stopping.

**Crisp:** Monitor validation loss; stop when it plateaus or increases. Simple, effective, and essentially free.

---

## Architectures

### Q14. Why CNNs work for images.

**Crisp:** Convolutions encode translation equivariance (the same filter applies everywhere). Pooling gives rough translation invariance. Weight sharing reduces parameters vs fully connected.

### Q15. Receptive field.

**Crisp:** The region of the input that influences a neuron's output. Grows with depth, dilations, and strides. Important when deciding whether the model can see enough context.

### Q16. Skip connections (ResNet).

**Crisp:** Allow gradients to flow directly to earlier layers, enabling very deep networks (100+ layers) to train. Intuition: the block learns a residual correction to identity, which is easier than learning the full transform.

### Q17. LSTM vs GRU.

**Crisp:** Both are gated recurrent units that address vanishing gradients in vanilla RNNs. LSTM has three gates (input, forget, output) plus a cell state; GRU has two gates (update, reset). GRU is simpler and often comparable; LSTM is better for very long sequences.

### Q18. Why transformers beat RNNs.

**Crisp:** Transformers parallelize across sequence length (RNNs are sequential). Self-attention gives direct access to any position, avoiding the information bottleneck in RNN hidden states. Scaled dot-product attention has good gradient properties.

---

## Attention and transformers

### Q19. Attention computation.

**Crisp:**

```
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
```

Q, K, V are linear projections of the input. The `sqrt(d_k)` scaling prevents softmax saturation at large `d_k`.

### Q20. Multi-head attention.

**Crisp:** Run attention multiple times in parallel with different projections, concatenate results, project back. Each head can specialize on different relations (syntactic, semantic, positional).

### Q21. Positional encoding.

**Crisp:** Self-attention is permutation-invariant, so we must inject position. Options: sinusoidal (original), learned absolute, relative (Shaw et al.), rotary (RoPE, used in modern LLMs).

### Q22. Encoder vs decoder transformers.

**Crisp:** Encoder (BERT) uses bidirectional attention, good for understanding. Decoder (GPT) uses causal attention, good for generation. Encoder-decoder (T5, BART) combines both.

### Q23. Why do LLMs hallucinate?

**Crisp:** Trained to predict the next token by likelihood, not truth. The model generates plausible-sounding continuations, which may not be factual. Mitigations: RAG, fact-check layers, fine-tuning for honesty.

---

## Training at scale

### Q24. Mixed precision training.

**Crisp:** Use fp16 (or bf16) for most ops, fp32 for a small subset (master weights, loss accumulation). Reduces memory and speeds up training on modern GPUs. Uses `torch.cuda.amp.autocast` and `GradScaler`.

### Q25. Gradient accumulation.

**Crisp:** Accumulate gradients over several mini-batches before stepping the optimizer. Simulates larger effective batch size on limited GPU memory.

### Q26. DDP vs DP.

**Crisp:** `DistributedDataParallel` (DDP) replicates the model on each GPU and synchronizes gradients; modern best practice. `DataParallel` (DP) is older, single-process, and has known bottlenecks. Use DDP.

### Q27. Model parallelism vs data parallelism.

**Crisp:** Data parallelism replicates the model and shards data. Model parallelism splits the model itself across devices (e.g., one layer per GPU). Large models use both (ZeRO, FSDP).

---

## Transfer learning and fine-tuning

### Q28. Feature extraction vs fine-tuning.

**Crisp:** Feature extraction freezes the pretrained backbone and trains only a new head. Fine-tuning unfreezes some or all layers and trains them on the downstream task. Feature extraction is faster; fine-tuning usually performs better with enough data.

### Q29. LoRA (low-rank adaptation).

**Crisp:** Inject low-rank trainable matrices into each attention layer; freeze everything else. Dramatically reduces trainable parameters (often <1 percent of the model). Standard for fine-tuning LLMs on consumer hardware.

### Q30. Catastrophic forgetting.

**Crisp:** Fine-tuning on task B can destroy performance on task A. Mitigations: elastic weight consolidation, rehearsal (mix old and new data), multi-task training, or use adapters instead of full fine-tuning.

---

## Inference and deployment

### Q31. Why quantize a model?

**Crisp:** Reduces memory and speeds up inference. Int8 quantization gives 4x smaller models and 2 to 4x faster inference on CPU, with small accuracy drop. Fp16 on GPU is nearly free.

### Q32. ONNX and TorchScript.

**Crisp:** ONNX is a cross-framework model format. TorchScript serializes a PyTorch model for deployment without the training loop. Both enable inference in production services without the full Python / training dependency chain.

### Q33. Why does a model's first inference call take longer?

**Crisp:** Model load into GPU, CUDA kernel compilation, lazy initialization. In a serving system, pre-warm the model with a dummy input at startup.

---

## Debugging deep learning

### Q34. Loss is not decreasing.

**Checklist:**
1. Verify inputs and labels are correct (print a few).
2. Verify loss function matches task (cross-entropy for classification, MSE for regression).
3. Try overfitting a single batch; if the model cannot, there is a bug.
4. Check learning rate; too high diverges, too low stalls.
5. Check gradient norms; if zero, the network is disconnected.
6. Check normalization; untrained batch norm can produce garbage on eval.

### Q35. Training and validation metrics differ wildly.

**Likely causes:** Data leakage (pattern in train not present in val), wrong augmentation at eval time, differing preprocessing between splits, incorrect label mapping.

### Q36. Model trains but generalizes poorly.

**Checklist:** Overfitting (add regularization), distribution shift (retrain on current data), small or non-representative val set (resample), metric mismatch (the offline metric does not align with the business outcome).

---

## PyTorch idioms

### Q37. What does `.to(device)` do, and when does it go wrong?

**Crisp:** Moves tensors or modules to a device (CPU, GPU). Common bug: tensor on CPU, model on GPU, crash. Always ensure inputs and model are on the same device.

### Q38. Why use `torch.no_grad()` at inference?

**Crisp:** Disables autograd tracking, saving memory and speeding up the forward pass. Essential for inference.

### Q39. `model.train()` vs `model.eval()`.

**Crisp:** Controls behavior of dropout and batch norm. `eval()` at inference is required or numbers will be wrong.

### Q40. DataLoader workers and pin_memory.

**Crisp:** `num_workers > 0` parallelizes data loading. `pin_memory=True` speeds up CPU to GPU transfer. Standard production settings.

---

## Interview-favorite scenarios

- "We have a CNN that overfits. Walk me through what you would try, in order." Expected sequence: more data, data augmentation, weight decay, dropout, smaller model, early stopping.
- "Our transformer training loss spikes randomly. What do you investigate?" Expected answer: gradient clipping, learning rate, bad batch (outliers), mixed-precision instability, weight initialization.
- "You have 10M labeled images and no GPU budget. What can you still do?" Expected: feature extraction from a pretrained model (no fine-tuning), train a simple linear head, quantize for CPU inference, sample aggressively.

Be ready to sketch architectures on a whiteboard: a basic CNN, a single transformer block, the forward pass through attention.
