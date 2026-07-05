---
tags: [deep-learning, pytorch, fundamentals]
status: review
created: 2026-04-23
updated: 2026-04-23
---

# Deep Learning with PyTorch

## What it is

PyTorch is the dominant framework for deep learning research and increasingly for production. It provides:

- Tensor operations with automatic differentiation (autograd).
- Composable neural network modules (`nn.Module`).
- Data loading utilities (`Dataset`, `DataLoader`).
- GPU support via CUDA or Apple Metal.
- Mixed-precision training (`torch.cuda.amp`).
- Serialization and inference runtimes (TorchScript, ONNX export).

## Why it matters

Most modern ML models in 2026 are trained in PyTorch or a PyTorch-backed framework (HuggingFace Transformers, PyTorch Lightning). Fluency is non-negotiable for ML-Engineer roles.

## When to use it

- Any deep learning task: vision, NLP, audio, tabular deep nets, recommenders.
- Research and prototyping (fastest iteration loop).
- Production with TorchServe, Triton, or a wrapping FastAPI service.

For tabular with hundreds of features and moderate data, gradient boosting (LightGBM, XGBoost) usually beats deep learning and is 10x cheaper. Use PyTorch when the task has unstructured data or structure that deep models specifically handle.

## Core concepts

### Tensors

N-dimensional arrays on CPU or GPU with autograd.

```python
import torch
x = torch.randn(3, 4, requires_grad=True)
y = (x ** 2).sum()
y.backward()
print(x.grad)  # dy/dx = 2x
```

### Modules

Stateful layers that hold parameters.

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))
```

### Training loop

```python
model = MLP(10, 32, 3)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(10):
    for x, y in train_loader:
        logits = model(x)
        loss = loss_fn(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
```

### Mixed precision

```python
scaler = torch.cuda.amp.GradScaler()
for x, y in loader:
    opt.zero_grad()
    with torch.cuda.amp.autocast():
        logits = model(x)
        loss = loss_fn(logits, y)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
```

## Python implementation

See `example.py` for a minimal end-to-end training loop on a toy dataset.

## Gotchas

- Forgetting `model.eval()` at inference (dropout and BN behave differently).
- Forgetting `opt.zero_grad()` between steps (gradients accumulate).
- Mixing CPU and GPU tensors ("expected all tensors to be on the same device").
- Holding onto computation graphs via Python references (OOM). Use `.detach()` when you only need values.
- Using `.cuda()` instead of `.to(device)` (locks code to NVIDIA; prefer device-agnostic).
- Loading a model saved in a different version of PyTorch (use `strict=False` on `load_state_dict` cautiously).

## Autograd internals

PyTorch builds a dynamic computational graph. Every operation on a tensor with `requires_grad=True` records its inputs and the local gradient function. `loss.backward()` traverses this graph backward applying the chain rule.

### `requires_grad`, `.grad`, `.detach()`

```python
x = torch.randn(3, requires_grad=True)
y = (x ** 2).sum()
y.backward()
print(x.grad)            # tensor of dy/dx values

# stop gradients flowing through a value
z = x.detach()           # same data, requires_grad=False, not in graph
z = x.detach().clone()   # same data, independent storage
```

Use `.detach()` when you want a tensor's value but not its history. Common in evaluation, in computing metrics, and when implementing target networks in RL.

### `retain_graph=True`

By default, the computational graph is freed after `backward()`. If you call `backward()` twice on the same graph (multi-loss training, GAN updates), pass `retain_graph=True` on all but the last call:

```python
loss1.backward(retain_graph=True)
loss2.backward()
opt.step()
```

Forgetting `retain_graph=True` raises `RuntimeError: trying to backward through the graph a second time`.

### `requires_grad=False` for frozen layers

```python
for p in model.encoder.parameters():
    p.requires_grad = False
```

Freezes the encoder during fine-tuning. The optimizer should typically include only `requires_grad=True` parameters: `torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))`.

## Custom Module pattern

The PyTorch idiom: subclass `nn.Module`, register parameters in `__init__`, define forward computation in `forward`.

```python
import torch
import torch.nn as nn

class GatedMLP(nn.Module):
    def __init__(self, dim, hidden, dropout=0.1):
        super().__init__()
        self.up = nn.Linear(dim, hidden)
        self.gate = nn.Linear(dim, hidden)
        self.down = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = nn.functional.silu(self.gate(x)) * self.up(x)
        return self.down(self.dropout(h))
```

Important: use `nn.Module` for parameter-holding components, not bare tensors. `nn.Module` automatically tracks parameters for `model.parameters()`, `model.to(device)`, and serialization.

## DataLoader patterns

```python
from torch.utils.data import DataLoader, Dataset

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,         # parallel data loading processes
    pin_memory=True,       # faster CPU-to-GPU transfer
    persistent_workers=True,  # avoid worker shutdown each epoch
    collate_fn=my_collate, # custom batch assembly
    drop_last=True,        # drop incomplete final batch
)
```

- **`num_workers`.** Set to 4 to 8 on a typical machine. Too high causes memory and shared-memory issues. Too low (0) makes data loading the bottleneck.
- **`pin_memory=True`.** Memory pinned in CPU RAM transfers faster to GPU. Standard with CUDA training.
- **`collate_fn`.** Custom function for assembling a batch when default stacking fails (variable-length sequences, dictionaries of tensors, mixed-modality data).
- **`persistent_workers=True`.** Avoids the cost of restarting workers each epoch; substantial speedup on small datasets.

## Memory-saving training tricks

### Gradient checkpointing

Trade compute for memory. Recompute intermediate activations during the backward pass instead of saving them.

```python
from torch.utils.checkpoint import checkpoint

def forward(self, x):
    x = checkpoint(self.block1, x, use_reentrant=False)
    x = checkpoint(self.block2, x, use_reentrant=False)
    return x
```

Roughly 30 to 40% memory reduction at the cost of 25 to 35% slower training. Standard for large models that would otherwise OOM.

### Automatic Mixed Precision (AMP)

```python
from torch import autocast, GradScaler

scaler = GradScaler()
for x, y in loader:
    opt.zero_grad()
    with autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(x)
        loss = loss_fn(logits, y)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
```

BF16 has the same range as FP32; loss scaling is unnecessary but `GradScaler` is harmless. FP16 requires loss scaling to avoid underflow. On Ampere or newer hardware, prefer BF16.

### 8-bit optimizers

```python
import bitsandbytes as bnb
opt = bnb.optim.AdamW8bit(model.parameters(), lr=2e-5)
```

Stores Adam optimizer state in 8-bit instead of FP32, reducing optimizer memory by 4x. Negligible quality loss in practice. Standard for fine-tuning large LLMs on a single GPU. Combine with QLoRA (4-bit base model + LoRA + 8-bit optimizer) for very large fine-tunes.

## Profiling

```python
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    schedule=profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=profiler.tensorboard_trace_handler("./prof"),
    record_shapes=True,
) as prof:
    for step, batch in enumerate(loader):
        train_step(batch)
        prof.step()
        if step >= 6: break
```

View the trace in TensorBoard or chrome://tracing. Look for:

- GPU utilization gaps (data loader bottleneck).
- Long memory copies (CPU-GPU transfers; pin memory).
- Specific kernels dominating time.

## Export for production

### TorchScript

```python
model.eval()
traced = torch.jit.trace(model, example_input)
# or, if your forward has control flow:
scripted = torch.jit.script(model)
scripted.save("model.pt")
```

`trace` records ops on a sample input; fails silently on data-dependent control flow. `script` parses the Python source and supports if/while.

### ONNX

```python
torch.onnx.export(
    model, example_input, "model.onnx",
    opset_version=17,
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
)
```

Run with ONNX Runtime for cross-platform inference (CPU, CUDA, mobile, web). ORT often beats TorchScript on CPU; both are competitive on GPU. For LLMs at scale, switch to TensorRT-LLM or vLLM instead.

## Resources

- PyTorch official tutorials: pytorch.org/tutorials.
- Goodfellow, Bengio, Courville, *Deep Learning*.
- `docs/theory/17_dl_fundamentals.md` for backprop, normalization, residual connections, LoRA.
- `docs/theory/09_llm_foundations_2026.md` for transformers built on PyTorch.
- `docs/theory/10_computer_vision_foundations.md` for CNNs built on PyTorch.
- `bitsandbytes` documentation for 8-bit optimizers and 4-bit quantization.
- Microsoft, "DeepSpeed and ZeRO" for multi-GPU training at scale.
- ONNX Runtime documentation.
