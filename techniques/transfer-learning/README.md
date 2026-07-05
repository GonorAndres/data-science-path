---
tags: [transfer-learning, fine-tuning, peft]
status: complete
created: 2026-04-23
updated: 2026-04-25
---

# Transfer Learning

## What it is

Reuse a model pretrained on a large source dataset and adapt it to a new target task with a much smaller labeled dataset. Instead of training from scratch, leverage the representations the pretrained model already learned. Transfer learning is the default workflow in modern ML; training from scratch is rare and reserved for fundamentally new domains where pretrained weights do not exist.

## Why it matters

- **Sample efficiency.** A pretrained ResNet adapted to a 1,000-image medical dataset reaches accuracy that a from-scratch model needs 100,000 images to match.
- **Compute efficiency.** Fine-tuning costs a small fraction of pretraining.
- **Time-to-model.** A working transfer-learning baseline for a new computer vision task is a one-day project; a from-scratch baseline is a one-month project.
- **Regulatory acceptance.** Pretrained models from reputable labs come with documented training data and known limitations; auditors are increasingly comfortable with this provenance.

## Why pretrained features transfer

Empirical observation (Yosinski et al. 2014, Zeiler-Fergus 2013): the features learned by a deep network form a hierarchy.

- **Early layers.** Edges, oriented gradients, color blobs, simple textures. These are universal across natural images.
- **Mid layers.** Parts: corners, eye-like patches, bumper-like patches, fur textures. Less universal but still broadly transferable to natural-image tasks.
- **Late layers.** Class-discriminative features tuned to the source task: "ImageNet dog breeds" vs "ImageNet office supplies". These are the least transferable.

The practical implication: lower layers transfer well across domains; later layers must be retrained or replaced. This intuition justifies the layer-wise learning-rate schedule (smaller LR for early layers, larger for later) used by every modern fine-tuning recipe.

The same intuition holds for transformers in NLP and speech: early attention layers attend to syntactic patterns; later layers carry task-specific semantic features.

## Strategy 1: Linear probe (feature extraction)

Freeze the entire backbone; train only a new linear (or shallow MLP) head on top of the frozen features.

```python
import torch.nn as nn
import timm

backbone = timm.create_model("resnet50", pretrained=True, num_classes=0)  # 0 = no head
for p in backbone.parameters():
    p.requires_grad = False

head = nn.Linear(backbone.num_features, num_classes)
model = nn.Sequential(backbone, head)

opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
```

**When to use it.** Small target dataset (under a few thousand labels), domain very similar to pretraining (natural images for an ImageNet backbone, English text for an English-pretrained encoder), or strict compute budget. Linear probes are also the gold-standard evaluation protocol for self-supervised pretraining; if your SSL features need a non-linear head to perform well, the features themselves are weaker than they look.

**Tradeoffs.** Lowest accuracy ceiling among fine-tuning options; cheapest training; minimal overfitting risk; reuses the same backbone for many downstream heads.

## Strategy 2: Full fine-tuning

Unfreeze the entire model. Train all parameters but with a much smaller learning rate for the backbone than for the new head, and typically with weight decay to control drift from the pretrained weights.

```python
backbone = timm.create_model("resnet50", pretrained=True, num_classes=num_classes)

opt = torch.optim.AdamW([
    {"params": backbone.fc.parameters(),                      "lr": 1e-3},
    {"params": [p for n, p in backbone.named_parameters() if "fc" not in n], "lr": 1e-5},
], weight_decay=1e-4)
```

**When to use it.** Larger target dataset (tens of thousands of labels), target domain meaningfully different from pretraining (medical scans for an ImageNet backbone, biomedical text for a general-domain BERT), or you are pushing for top-of-leaderboard accuracy.

**Tradeoffs.** Higher accuracy ceiling than linear probe; more compute; risk of catastrophic forgetting and overfitting on small targets if hyperparameters are wrong.

## Strategy 3: Two-stage protocol

The recommended default for new projects. Combines the stability of feature extraction with the accuracy of full fine-tuning.

1. **Stage 1: head warmup.** Freeze the backbone. Train only the new head for $N$ epochs with a moderate learning rate ($10^{-3}$). The randomly initialized head is given time to converge before any backbone updates begin.
2. **Stage 2: full fine-tuning with layer-wise LR decay.** Unfreeze the backbone. Use a 10x smaller learning rate for the backbone than for the head, and decay the per-layer LR going from late to early layers (e.g., multiply by 0.9 per layer block).

Why two stages. If you unfreeze the backbone before the head has converged, the gradient flowing back through the head is essentially noise. The backbone weights drift in random directions, destroying the pretrained features. The warmup stage prevents this.

```python
# Stage 1
for p in backbone.parameters(): p.requires_grad = False
for p in head.parameters():     p.requires_grad = True
train(model, epochs=3, lr=1e-3)

# Stage 2
for p in backbone.parameters(): p.requires_grad = True
opt = torch.optim.AdamW(layer_wise_lr_decay(backbone, head, head_lr=1e-3,
                                            backbone_lr=1e-4, decay=0.9))
train(model, epochs=20, opt=opt)
```

The `layer_wise_lr_decay` helper assigns each parameter group a learning rate that decreases as you move from the head toward the input. ULMFiT (Howard, Ruder 2018) introduced this for NLP; it has become the universal recipe for both vision and language fine-tuning.

## Strategy 4: LoRA for LLMs

Full fine-tuning of a 7-billion-parameter LLM requires hundreds of gigabytes of GPU memory (model weights + optimizer state + gradients + activations). **LoRA** (Hu et al. 2021) reduces this to a fraction by freezing the base weights and learning small low-rank update matrices.

For a frozen weight matrix $W \in \mathbb{R}^{d \times d}$, LoRA learns $A \in \mathbb{R}^{d \times r}$ and $B \in \mathbb{R}^{r \times d}$ with rank $r \ll d$ such that the effective weight during fine-tuning is

$$
W' = W + \frac{\alpha}{r} B A
$$

Only $A$ and $B$ are trained; $W$ stays frozen. The number of trainable parameters drops from $d^2$ to $2 d r$. For $d = 4096$ and $r = 16$, that is a 128x reduction.

Hyperparameters:

- **rank $r$.** Typical values 4 to 64. Higher rank is more expressive but costs more memory and risks overfitting on small targets.
- **alpha.** Scaling factor that decouples the learning rate from the rank. Common practice is $\alpha = 2r$ or $\alpha = r$.
- **target_modules.** Which weight matrices to apply LoRA to. The original paper applied it to query and value projections only; modern practice applies LoRA to all linear layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` in Llama-style architectures), which improves accuracy at moderate extra cost.

Code with the `peft` library:

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(base, config)
model.print_trainable_parameters()
# trainable params: ~42M, total params: ~8B  -> ~0.5%
```

At inference, LoRA adapters can be **merged** into the base weights ($W \leftarrow W + \frac{\alpha}{r} B A$), eliminating the runtime overhead. Or kept separate, allowing fast switching between adapters trained for different tasks against a shared base.

## Strategy 5: QLoRA

QLoRA (Dettmers et al. 2023) combines LoRA with **4-bit quantization** of the base weights. The quantized base stays frozen; LoRA adapters are trained in higher precision (typically bfloat16). This brings fine-tuning of a 70B model within reach of a single 80GB GPU and a 7B model within reach of a 24GB consumer GPU.

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
base = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-70B", quantization_config=bnb_config
)
# then wrap with LoRA as above
```

QLoRA's `nf4` (NormalFloat-4) quantization preserves more accuracy than naive int4 because it is matched to the empirical distribution of pretrained weights. Double quantization further reduces the memory of the quantization constants themselves. The combination loses very little accuracy compared to full bfloat16 LoRA in published benchmarks.

## Domain adaptation

When source and target distributions differ significantly, plain fine-tuning is suboptimal. Two responses:

- **CORAL** (Sun, Saenko 2016) aligns the second-order statistics of source and target features. Subtract source covariance, add target covariance; cheap, often surprisingly effective for tabular and image features.
- **Adversarial domain adaptation** (Ganin et al. 2016) adds a domain-discriminator head trained to distinguish source from target features, with a gradient reversal layer feeding back into the encoder. The encoder learns features the discriminator cannot tell apart, removing domain-specific information.

For LLM applications the analogous tool is **continued pretraining**: continue masked-LM or causal-LM training on a target-domain corpus before any task-specific fine-tuning. A medical LLM fine-tuned for triage starts from a base + continued-pretrain on PubMed + LoRA on the triage dataset, rather than fine-tuning the general base directly.

## Common gotchas

1. **Mismatched preprocessing.** The single most common transfer-learning bug. The pretrained model expects a specific normalization (e.g., ImageNet mean/std for vision, the pretraining tokenizer for NLP). Use the model's own preprocessing pipeline, never your own.
2. **Unfreezing too early.** Without head warmup, gradient noise from the random head wrecks pretrained features. Always train the head first.
3. **Catastrophic forgetting on small targets.** When the target dataset is small and the LR is too high, the model overfits to the target and loses pretraining knowledge it needed. Use a smaller LR, weight decay, and early stopping based on a held-out target validation set.
4. **Tokenizer or vocabulary drift in NLP.** If you fine-tune with a different tokenizer than the base model used, the embeddings are misaligned. Always use the base model's tokenizer.
5. **Frozen batch normalization disagreement.** During fine-tuning, BatchNorm running statistics either (a) keep updating from target data, which can be unstable on small batches, or (b) are frozen. The right choice is task-dependent; for small targets, freeze BN to eval mode.
6. **LoRA target_modules too narrow.** Q-V-only LoRA is the original recipe but underperforms all-linear LoRA on most modern instruction-tuning benchmarks. Default to all linear layers.

## Resources

- `docs/theory/10_computer_vision_foundations.md` for the vision context.
- `docs/theory/09_llm_foundations_2026.md` for the LLM context.
- `projects/cv-defect-detection` for an end-to-end vision example.

## References

- Yosinski, Clune, Bengio, Lipson, "How transferable are features in deep neural networks?", NeurIPS 2014.
- Howard, Ruder, "Universal Language Model Fine-tuning for Text Classification" (ULMFiT), ACL 2018.
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022.
- Dettmers, Pagnoni, Holtzman, Zettlemoyer, "QLoRA: Efficient Finetuning of Quantized LLMs", NeurIPS 2023.
- Ganin et al., "Domain-Adversarial Training of Neural Networks", JMLR 2016.
- Sun, Saenko, "Deep CORAL: Correlation Alignment for Deep Domain Adaptation", ECCV Workshop 2016.
- HuggingFace `peft` library, github.com/huggingface/peft.
