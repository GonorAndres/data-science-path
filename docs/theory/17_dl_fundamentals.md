# Deep Learning Fundamentals

This essay establishes the deep-learning vocabulary that essays 09 (LLMs) and 10 (computer vision) assume. It builds on the optimization primer (00b) and is the right starting point before reading either downstream essay.

## Backpropagation

A neural network is a composition of differentiable functions. Training minimizes a loss $L$ with respect to parameters $\theta$ via gradient descent. Backpropagation is the algorithm for computing $\nabla_\theta L$ efficiently.

### Two-layer example

Consider a two-layer MLP with input $x \in \mathbb{R}^d$, hidden activation $\sigma$, and squared loss against target $y$:

$$
h = \sigma(W_1 x + b_1), \qquad \hat y = W_2 h + b_2, \qquad L = \tfrac{1}{2}(\hat y - y)^2.
$$

By the chain rule,

$$
\frac{\partial L}{\partial W_2} = (\hat y - y) h^\top, \qquad \frac{\partial L}{\partial W_1} = \left(W_2^\top (\hat y - y) \odot \sigma'(W_1 x + b_1)\right) x^\top.
$$

The pattern: gradients flow backward through the network, multiplying by Jacobians at each layer. The key efficiency win is that intermediate activations $h$ are saved during the forward pass so the backward pass reuses them.

### Computational graph

Modern frameworks (PyTorch, JAX, TensorFlow) build a computational graph during the forward pass: each operation records its inputs and the local Jacobian. Backward traversal applies the chain rule mechanically. This is **automatic differentiation**, and it is what `loss.backward()` in PyTorch invokes.

### Vanishing and exploding gradients

In a deep network with $L$ layers, the gradient at layer 1 is a product of $L$ Jacobians. If each Jacobian has spectral norm $< 1$, the product shrinks exponentially: gradients vanish, early layers stop learning. If $> 1$, the product grows: gradients explode, training diverges.

Mitigations:

- **Initialization** (next section).
- **Normalization layers** (BN, LN, RMSNorm).
- **Residual connections** that add an identity path so gradient does not have to traverse every layer.
- **Gradient clipping**: rescale gradients with norm above a threshold before stepping.

## Initialization

### Xavier (Glorot)

Glorot, Bengio (2010). For a linear layer with $n_{\mathrm{in}}$ inputs and $n_{\mathrm{out}}$ outputs feeding a tanh or sigmoid activation, draw weights from

$$
W \sim \mathcal{U}\!\left(-\sqrt{6 / (n_{\mathrm{in}} + n_{\mathrm{out}})}, \sqrt{6 / (n_{\mathrm{in}} + n_{\mathrm{out}})}\right).
$$

Equivalently, variance $2 / (n_{\mathrm{in}} + n_{\mathrm{out}})$. The motivation: preserve the variance of activations and gradients across layers under the linearity-near-zero assumption of tanh/sigmoid.

### He initialization

He, Zhang, Ren, Sun (2015). For ReLU activations, half the units output zero on average. Compensate by doubling the variance:

$$
W \sim \mathcal{N}(0, 2 / n_{\mathrm{in}}).
$$

He init is the modern default for any ReLU-family activation. PyTorch's `nn.Linear` defaults to a Kaiming uniform variant.

The bias is typically initialized to zero. Empirically, getting initialization right makes the difference between a network that trains in 50 epochs and one that stalls.

## Normalization

### Batch Normalization (BN)

Ioffe, Szegedy (2015). For each mini-batch and each feature dimension, compute the batch mean and variance, normalize, and apply learnable scale and shift:

$$
\hat x_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \qquad y_i = \gamma \hat x_i + \beta.
$$

At inference, replace batch statistics with running averages accumulated during training. **The training-vs-inference distinction is critical**: forgetting to call `model.eval()` in PyTorch leaves BN in training mode and produces unstable predictions on small batches.

BN works well in CNNs with reasonable batch sizes. It struggles with batch size 1 (sequence models, online learning) and is sensitive to batch composition (mixing classes vs not).

### Layer Normalization (LN)

Ba, Kiros, Hinton (2016). Normalize across feature dimensions per example, not across the batch:

$$
\hat x_i^{(d)} = \frac{x_i^{(d)} - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}}.
$$

Works at batch size 1, no train/inference distinction, no running statistics. Standard in transformers, RNNs, and any sequence model.

### RMSNorm

Zhang, Sennrich (2019). Drop the mean centering, keep only the RMS rescale:

$$
\hat x_i = \frac{x_i}{\sqrt{\tfrac{1}{D}\sum_d x_i^{(d) 2} + \epsilon}}.
$$

Cheaper than LN (one fewer reduction), empirically equivalent or slightly better in large language models. Used in Llama, Mistral, and most modern open-weight LLMs.

### When to use each

- **Convolutional vision models with batch size $\geq 32$**: BN.
- **Transformers, RNNs, small batches**: LN or RMSNorm.
- **LLMs in 2026**: RMSNorm is the default.

## Activations

- **ReLU.** $\max(0, x)$. Cheap, sparse, unbiased. **Dying ReLU**: a unit whose pre-activation is negative on every example produces zero output and zero gradient forever. Mitigated by Leaky ReLU ($\alpha x$ for $x < 0$) and by careful initialization.
- **GeLU.** $x \cdot \Phi(x)$ where $\Phi$ is the standard normal CDF. Smooth, non-monotonic for negative $x$. Standard in BERT and the original GPT family.
- **SiLU (Swish).** $x \cdot \sigma(x)$ with $\sigma$ the sigmoid. Smooth, similar shape to GeLU. Default in Llama 2/3 (often combined with the gated SwiGLU variant).
- **Tanh, Sigmoid.** Mostly historical. Saturate at the tails, vanishing-gradient prone. Still used in gated cells (LSTM, GRU) and in output layers when bounded outputs are required.

Modern preference: SiLU/SwiGLU for LLMs, GeLU for older transformer baselines, ReLU for vision CNNs where speed matters.

## Residual connections

He et al. (2015). Add the input of a block to its output:

$$
y = x + F(x).
$$

Two interpretations:

1. **Gradient highway.** The Jacobian of $y$ with respect to $x$ contains an identity term, so gradients flow back unchanged regardless of what $F$ does. Vanishing gradients are mitigated by construction.
2. **Ensemble of shallow networks.** Veit, Wilber, Belongie (2016) show that ResNets behave like an exponential ensemble of shallow paths: each residual decision (use the block or skip it) gives $2^L$ implicit paths.

Residual connections are now ubiquitous: ResNet, Transformer, U-Net, modern audio and video models. A network without them at depth 50+ usually fails to train.

## The modern training recipe

A typical 2026 training script for a transformer or large CNN includes:

- **Linear warmup** for the first 1 to 5 percent of steps, ramping the learning rate from 0 to its target.
- **Cosine annealing** for the remainder, decaying smoothly to a small minimum (often $10\%$ of the peak LR or zero).
- **Weight decay** at $0.01$ to $0.1$ via AdamW (essay 02). Bias and normalization parameters typically excluded from decay.
- **Label smoothing** at $\alpha = 0.1$ for classification, easing over-confident predictions.
- **EMA of weights.** Maintain an exponential moving average of model parameters in parallel to the optimizer state; evaluate using EMA weights. Improves test accuracy and stability of a few percentage points in many tasks.
- **Gradient clipping.** Clip the global gradient norm at 1.0 to prevent occasional explosions.
- **Random seeds and reproducibility.** Set torch, numpy, and CUDA seeds; document the deterministic-mode trade-off.

These knobs are not optional flourishes; their absence is a common cause of subtle quality losses.

## Mixed precision: FP16 vs BF16

Modern GPUs (Ampere, Hopper) deliver 2x to 4x throughput on lower-precision arithmetic.

### FP16

5-bit exponent, 10-bit mantissa. Range is small (about $6.5 \times 10^{4}$). Loss scaling is required: multiply the loss by a large constant before backward, divide gradients before the optimizer step, to push gradients into the representable range.

### BF16

8-bit exponent (same as FP32), 7-bit mantissa. Same range as FP32, less precision. **Loss scaling is unnecessary**: gradients fit naturally. Slightly less accurate per-operation than FP16 but stabler training.

Modern preference: BF16 on Ampere/Hopper. FP16 only on older hardware (V100). PyTorch's `torch.amp.autocast(dtype=torch.bfloat16)` is the standard incantation.

## Parameter-efficient fine-tuning (PEFT) and LoRA

Full fine-tuning of a 7B model requires updating 7 billion parameters and storing optimizer states (Adam adds 2x the model in floats). PEFT methods update a small subset of parameters and freeze the rest.

### LoRA: Low-Rank Adaptation

Hu et al. (2021). Augment a pretrained weight matrix $W \in \mathbb{R}^{m \times n}$ with a low-rank update:

$$
W' = W + \Delta W, \qquad \Delta W = B A, \qquad A \in \mathbb{R}^{r \times n},\ B \in \mathbb{R}^{m \times r},\ r \ll \min(m, n).
$$

Only $A$ and $B$ are trained; $W$ is frozen. Parameter count drops from $mn$ to $r(m+n)$. Typical ranks: $r = 8$ to $64$.

The scaling factor $\alpha / r$ multiplies $BA$; tuning $\alpha$ is roughly equivalent to scaling the learning rate of the LoRA path. Empirical practice: $\alpha = 2r$ or $\alpha = r$ are common starting points.

### Merge at inference

Once trained, $\Delta W = BA$ can be folded back into $W$: $W \gets W + BA$. The merged model has the original architecture and inference cost. No latency penalty for using LoRA in production.

### QLoRA

Dettmers et al. (2023). Train LoRA adapters on top of a 4-bit quantized base model. Reduces fine-tuning memory by another factor of 4 with negligible quality loss. Standard for fine-tuning large LLMs on a single GPU.

### Other PEFT methods

- **Prefix-tuning, prompt-tuning.** Train a small set of "soft prompt" embeddings prepended to inputs. Cheaper than LoRA but typically lower quality.
- **Adapters** (Houlsby 2019). Insert small bottleneck modules between transformer layers. The original PEFT approach; LoRA mostly replaced it.
- **IA3** (Liu et al. 2022). Multiplicative scaling of activations, even fewer parameters than LoRA.

LoRA and QLoRA are the modern defaults; they cover most fine-tuning use cases.

## Linking to downstream essays

- Essay 09 (LLMs) assumes attention, normalization, and the transformer training recipe established here.
- Essay 10 (CV) assumes convolutions, pooling, residual connections, and the modern recipe.
- Essay 17b (this essay) is the prerequisite both depend on. Read this first if you have not seen backprop, BN/LN, or LoRA before.

## References

- Goodfellow, Bengio, Courville, *Deep Learning*, MIT Press, 2016.
- Glorot, Bengio, "Understanding the difficulty of training deep feedforward neural networks", AISTATS 2010.
- He, Zhang, Ren, Sun, "Delving Deep into Rectifiers" (He init), ICCV 2015.
- He, Zhang, Ren, Sun, "Deep Residual Learning for Image Recognition" (ResNet), CVPR 2016.
- Ioffe, Szegedy, "Batch Normalization", ICML 2015.
- Ba, Kiros, Hinton, "Layer Normalization", 2016.
- Zhang, Sennrich, "Root Mean Square Layer Normalization", NeurIPS 2019.
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022.
- Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs", NeurIPS 2023.
- Veit, Wilber, Belongie, "Residual Networks Behave Like Ensembles of Relatively Shallow Networks", NeurIPS 2016.
- Touvron et al., Llama 2/3 technical reports (modern training recipe in practice).
