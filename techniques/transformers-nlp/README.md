---
tags: [transformers, nlp, deep-learning, huggingface]
status: review
created: 2026-04-23
updated: 2026-04-25
---

# Transformers for NLP

## What it is

The transformer architecture (Vaswani et al. 2017) is a sequence-to-sequence model built around self-attention rather than recurrence or convolution. It is the foundation of every state-of-the-art NLP model since 2018 and the underlying engine of LLMs, encoder embeddings, and most modern multimodal systems.

This document covers the practical fine-tuning and deployment of transformer models for NLP, mostly via HuggingFace Transformers. For the theoretical background, see `docs/theory/09_llm_foundations_2026.md` and `docs/theory/17_dl_fundamentals.md`.

## Why it matters

In 2026, fine-tuning a HuggingFace transformer is a baseline DS skill. Recruiter expectations for an internship-level role: load a pretrained model, fine-tune on a labeled dataset, evaluate, and ship to a service. This README walks through that loop.

## Three architectural families

### Encoder-only (BERT family)

- **Pretraining objective.** Masked Language Modeling (MLM): mask 15% of tokens, predict them from bidirectional context.
- **Use cases.** Classification, NER, question answering with extractive answers, sentence-level embedding.
- **Examples.** BERT, RoBERTa, DistilBERT, DeBERTa-v3, XLM-R (multilingual), BETO (Spanish-specific).
- **Output.** Per-token contextual embeddings; pool the `[CLS]` token or mean-pool for sentence-level tasks.

### Decoder-only (GPT family)

- **Pretraining objective.** Causal Language Modeling (CLM): predict the next token given all previous tokens.
- **Use cases.** Generation, summarization, chat, anything autoregressive. Also used as embedding models with appropriate pooling.
- **Examples.** GPT-2, Llama 2/3, Mistral, Qwen, Phi, Gemma.

### Encoder-decoder (T5 family)

- **Pretraining objective.** Span corruption: mask contiguous spans, predict them with the decoder.
- **Use cases.** Translation, abstractive summarization, structured generation, anything seq2seq.
- **Examples.** T5, BART, mT5 (multilingual), FLAN-T5.

## MLM vs CLM in practice

| | MLM (BERT) | CLM (GPT) |
|----|-----|-----|
| Reads context | Bidirectional | Left-to-right only |
| Pretraining sample efficiency | High (15% positions per example) | Lower (1 prediction per token) |
| Suited for | Classification, NER | Generation, in-context learning |
| Typical embedding quality | High out-of-the-box | Needs special pooling |

A modern shortcut: many encoder tasks now use frozen LLM embeddings (mE5, BGE) which are decoder-trained then contrastively fine-tuned. The line between families has blurred since 2023.

## Attention mechanism (worked numerical example)

Given input embeddings $X \in \mathbb{R}^{n \times d}$, three projections produce queries, keys, and values:

$$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V.
$$

Single-head attention:

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V.
$$

### Numerical example

Take $n = 3$ tokens, $d = d_k = 2$. Suppose

$$
Q = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{pmatrix}, \quad
K = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{pmatrix}, \quad
V = \begin{pmatrix} 10 & 0 \\ 0 & 10 \\ 5 & 5 \end{pmatrix}.
$$

Then $QK^\top$:

$$
QK^\top = \begin{pmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 1 & 1 & 2 \end{pmatrix}.
$$

Divide by $\sqrt{2} \approx 1.414$ and softmax row-wise. Row 1 is approximately $(0.39, 0.22, 0.39)$. The first output token is

$$
0.39 \cdot (10, 0) + 0.22 \cdot (0, 10) + 0.39 \cdot (5, 5) = (5.85, 4.15).
$$

Each output is a soft mixture of value rows weighted by query-key compatibility.

### Multi-head attention

Run $h$ parallel attention heads with separate $W_Q^{(h)}, W_K^{(h)}, W_V^{(h)}$ projections, concatenate the outputs, and apply a final linear projection. Each head learns a different relation (subject-verb, coreference, syntactic dependence, etc.). Modern LLMs use 32 to 128 heads.

### Causal masking

For decoder-only models, set $QK^\top$ entries above the diagonal to $-\infty$ before softmax so token $t$ cannot attend to tokens $> t$. This enforces autoregressive generation.

## Fine-tuning recipes

### Full fine-tuning

Update every parameter on the downstream task. Strongest accuracy, highest compute and memory cost. Default for small encoders (BERT-base, 110M parameters) on small-data tasks.

### Linear probing

Freeze the encoder, train only a linear classification head on top. Cheap, weak baseline. Useful for diagnostic comparison and when you need fast iteration on many tasks with the same encoder.

### LoRA

Low-rank adaptation (Hu et al. 2021). Insert trainable low-rank matrices $\Delta W = BA$ into selected linear layers (typically Q and V projections in attention); freeze the base model. Train only a few million parameters; achieve close to full-FT quality on most NLP benchmarks.

Hyperparameters:

- **Rank $r$.** Typical 8 to 64. Higher $r$ for harder tasks; 16 is a reasonable default.
- **Alpha $\alpha$.** Scaling factor; effective LR multiplier. $\alpha = 2r$ or $\alpha = r$ are common.
- **Target modules.** Defaults: `q_proj` and `v_proj`. For higher quality at more cost, include `k_proj`, `o_proj`, and the MLP projections.

Implementation: HuggingFace `peft` library. Training a LoRA fine-tune on a 7B model fits on a 24 GB consumer GPU with QLoRA.

### Prompt-tuning and prefix-tuning

Train a small set of "soft prompt" embeddings prepended to the input; freeze everything else. Fewer parameters than LoRA, lower quality. Mostly superseded.

## HuggingFace Trainer: minimal sequence classification

```python
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding,
)
import evaluate
import numpy as np

ds = load_dataset("imdb")
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess(batch):
    return tok(batch["text"], truncation=True, max_length=256)

ds = ds.map(preprocess, batched=True)
collator = DataCollatorWithPadding(tokenizer=tok)

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)

f1 = evaluate.load("f1")
def metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return f1.compute(predictions=preds, references=labels, average="binary")

args = TrainingArguments(
    output_dir="./out",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    warmup_ratio=0.06,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    tokenizer=tok,
    data_collator=collator,
    compute_metrics=metrics,
)

trainer.train()
trainer.save_model("./final")
```

This is the canonical fine-tuning skeleton; substitute model name, dataset, and metric for almost any classification task.

## Evaluation per task

| Task | Metric |
|------|--------|
| Binary or multiclass classification | Accuracy, F1 (binary or macro), Matthews Correlation Coefficient (MCC) |
| Imbalanced classification | Macro-F1, AUPRC |
| Named Entity Recognition | Entity-level F1 (`seqeval`) |
| Extractive QA | Exact Match, F1 over answer span tokens |
| Translation | BLEU, chrF |
| Summarization | ROUGE-1, ROUGE-2, ROUGE-L |
| Generation quality (semantic) | BERTScore, BLEURT |
| Multilingual | Per-language metrics, then macro across languages |

For NER specifically, **token-level F1 is misleading** because most tokens are "O" (outside entity). Always use entity-level F1 from `seqeval`.

## Production gotchas

- **Tokenizer-model mismatch.** Always load the tokenizer from the same checkpoint as the model. A vocabulary mismatch silently corrupts inputs.
- **Padding side.** Encoder models (BERT) right-pad; decoder models for generation left-pad so the last token is at the end of every sequence. HuggingFace's `tokenizer.padding_side` controls this; mistakes here produce wrong logits with no error.
- **Attention masks.** Whenever inputs are padded, pass `attention_mask` so attention ignores pad tokens. Forgetting this gives subtly wrong predictions.
- **`max_length` truncation strategy.** `truncation=True` truncates to model max; for QA, `truncation="only_second"` keeps the question intact and truncates the context. Truncating the wrong field breaks the task.
- **`model.eval()` at inference.** Disables dropout. Forgetting it produces stochastic predictions.
- **CPU vs GPU device placement.** `model.to(device)` and inputs `.to(device)`; mismatches raise cryptic errors.
- **Mixed precision and overflow.** FP16 can NaN on extreme inputs. Use BF16 if your hardware supports it; it has the same range as FP32.
- **Tokenizer padding token mismatch on decoder-only models.** GPT-2 has no pad token by default; set `tokenizer.pad_token = tokenizer.eos_token` and `model.config.pad_token_id = tokenizer.eos_token_id` consistently.
- **`label` vs `labels` field.** HuggingFace Trainer expects `labels` (plural). A `label` column will be silently ignored.

## Resources

- `docs/theory/09_llm_foundations_2026.md`: theory companion with attention math, training stages, RAG.
- `docs/theory/17_dl_fundamentals.md`: backprop, normalization, residual connections, LoRA derivation.
- Vaswani et al., "Attention Is All You Need", NeurIPS 2017.
- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", NAACL 2019.
- Conneau et al., "Unsupervised Cross-lingual Representation Learning at Scale" (XLM-R), ACL 2020.
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022.
- Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (T5), JMLR 2020.
- HuggingFace Transformers docs: https://huggingface.co/docs/transformers
- HuggingFace PEFT docs: https://huggingface.co/docs/peft
