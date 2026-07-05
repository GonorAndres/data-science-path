# Transformers and Multilingual Tokenization

For the general transformer background, see `docs/theory/09_llm_foundations_2026.md`. This document focuses on the specifics that matter for this project: multilingual tokenization, subword fragmentation across languages, and the choice of backbone.

## Subword tokenization

Modern transformers tokenize at the subword level. A token is not a word; it is a chunk produced by one of:

- **Byte-Pair Encoding (BPE).** Iteratively merges frequent character pairs. Used in GPT family, RoBERTa.
- **WordPiece.** Similar to BPE but optimizes a likelihood instead of frequency. Used in BERT.
- **SentencePiece (unigram).** Learns a probabilistic segmentation. Used in T5, XLM-R.

A token can be a whole word, a prefix, a suffix, or a single character. The tokenizer has a fixed vocabulary (typically 30k to 250k tokens).

## Why multilingual tokenization is not linguistically neutral

A tokenizer trained on a corpus dominated by English produces shorter token sequences for English and longer ones for other languages. Practical effects:

- **Cost asymmetry.** A Spanish sentence may cost 30 percent more tokens than its English equivalent, meaning higher API cost and slower inference.
- **Context dilution.** With a fixed context window (say 512 tokens), Spanish inputs fit less original text.
- **Uneven representation quality.** Low-resource languages get more character-level splits, and the model has less signal per token.

For our task, Amazon reviews are short (median 30 to 80 words), so context window is not a problem. But the cost asymmetry is visible in training time per epoch.

## Multilingual model choices

### mBERT (bert-base-multilingual-cased)

- 180M parameters, 104 languages.
- Trained on Wikipedia text.
- Solid baseline; showing its age in 2026.

### XLM-R (xlm-roberta-base)

- 270M parameters, 100 languages.
- Trained on CommonCrawl, much larger.
- Standard strong multilingual encoder. Widely used.

### Multilingual DistilBERT

- 134M parameters.
- Distilled from mBERT.
- 40 percent faster, ~2 percent lower accuracy.

### XLM-RoBERTa Large

- 550M parameters.
- Notable accuracy gain over base.
- 4x slower inference.

### Specialized Spanish models

- **BETO** (bert-base-spanish-wwm-cased). Spanish-only. Strong on Spanish but not usable for English.
- **Multi-BETO.** Spanish + Portuguese.

### Decoder-only LLMs (Llama, Qwen)

- Much larger (7B+).
- Can do zero-shot sentiment well.
- Expensive for per-review inference at scale.

### The choice for this project

Default: **multilingual DistilBERT.** Reasons:

- Fits on a laptop GPU (8 GB VRAM enough for training).
- ~100 ms per-review inference on CPU.
- Good accuracy on short text.

Fallback if budget permits: **XLM-R base** for a few-point accuracy bump.

## Tokenizer behavior on review text

Consider an example Spanish review:

> "Me encanto el telefono, llego rapido y funciona perfecto. Super recomendado!"

Tokenized by mBERT:

```
['Me', 'enc', '##anto', 'el', 'tele', '##fono', ',', 'lle', '##go', 'ra', '##pido', 'y', 'fun', '##ciona', 'per', '##fecto', '.', 'Super', 'recom', '##endado', '!']
```

Word count: 11. Token count: 21. Almost 2x expansion.

The equivalent English review:

> "I loved the phone, it arrived fast and works perfect. Highly recommended!"

Tokenized:

```
['I', 'loved', 'the', 'phone', ',', 'it', 'arrived', 'fast', 'and', 'works', 'perfect', '.', 'Highly', 'recommended', '!']
```

Word count: 12. Token count: 15. Closer 1:1.

This affects:

- Training batch size (keep sequences padded to the same length).
- Attention cost (quadratic in sequence length).
- Error modes (a single rare Spanish word split into 4 tokens dilutes its signal).

## Preprocessing decisions

For robustness:

- **Lower-casing.** Use the cased model and keep cases; cases carry sentiment signal ("PERFECT" vs "perfect").
- **Emojis.** Keep them. Modern tokenizers handle them as tokens and they carry sentiment.
- **URLs.** Replace with a `[URL]` placeholder.
- **Numbers.** Keep; they are signal ("1 star").
- **Accents.** Keep. Mexican Spanish uses accents inconsistently in reviews, but the model handles both forms.
- **Max length.** Truncate at 128 tokens; reviews over that are rare and usually rambling.

## Attention mask and padding

Sequences are padded to the longest in the batch (or a fixed length). The attention mask tells the model to ignore padded positions.

Pad token: model-specific (0 for mBERT, 1 for XLM-R). HuggingFace's tokenizer handles this automatically when called with `padding=True`.

## Output layer

For classification, take the `[CLS]` token's final hidden state, pass through a dropout layer and a linear classifier to 3 output classes. Loss: cross-entropy.

Some alternatives:

- Pool mean of all token hidden states (sometimes slightly better).
- Concatenate `[CLS]`, mean, max pool.

For 3-class sentiment, the `[CLS]` pooling is strong. Experiment if headroom appears.

## Fine-tuning dynamics

Training a pretrained model on 200k examples for 3 epochs:

- Learning rate 2e-5 is the HuggingFace default for encoder fine-tuning. Higher rates cause loss spikes; lower rates underfit within the budget.
- Weight decay 0.01 prevents the adapter weights from drifting too far from pretraining.
- Warmup for 10 percent of steps stabilizes the first epoch.
- Linear learning rate schedule afterwards.

Expect 2 to 4 hours on a consumer GPU.

## Inference considerations

At serving time:

- Use `torch.no_grad()` to disable autograd.
- `model.eval()` to switch batch norm / dropout to eval mode.
- Batch size at inference: optimize for latency (1) or throughput (32+) depending on workload.
- Quantization (int8 dynamic) for CPU inference speeds up 2-3x with ~1 percent accuracy drop.

## Linking to the project

The code in `src/train.py` fine-tunes multilingual DistilBERT on the preprocessed dataset. `src/predict.py` loads the best checkpoint and serves inference on a single string. `src/evaluate.py` computes the full confusion matrix and macro F1 on the test set.

## References

- Conneau et al., "Unsupervised Cross-lingual Representation Learning at Scale" (XLM-R), 2020.
- Sanh et al., "DistilBERT, a distilled version of BERT", 2019.
- Liu et al., "Multilingual Denoising Pre-training for Neural Machine Translation" (mBART), 2020.
- Pires, Schlinger, Garrette, "How multilingual is Multilingual BERT?", 2019.
- `docs/theory/09_llm_foundations_2026.md` for transformer foundations.
