# LLM Foundations for Data Scientists (2026)

## The arc, briefly

Modern large language models descend from the Transformer (Vaswani et al. 2017), scaled up with decoder-only variants (GPT series), trained on trillions of tokens, and aligned via RLHF or DPO. By 2026, the baseline production LLM has roughly 7B to 70B parameters for open-weight models and trillions for proprietary frontier models. Inference cost has fallen 10 to 100x since 2022.

Data scientists in 2026 are expected to use LLMs (prompting, RAG, agents) far more often than to train them. This essay covers the conceptual foundation needed for that practical work.

## Tokenization

LLMs do not read characters; they read tokens. A token is a subword unit produced by byte-pair encoding (BPE), WordPiece, or SentencePiece. English text averages about 4 characters per token; Spanish tokenization is slightly less efficient because the English-dominated vocabularies are optimized for English.

Implications:

- Cost. API pricing is per token. Knowing token counts matters.
- Context limits. A 128k-token context window holds roughly 96k English words or 75k Spanish words.
- Numeric precision. Some tokenizers split numbers in odd ways; arithmetic performance varies.

## The attention mechanism

Given input embeddings $X \in \mathbb{R}^{n \times d}$, compute queries, keys, and values:

$$
Q = XW_Q, \quad K = XW_K, \quad V = XW_V
$$

Attention output:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

Intuition: each output row is a convex combination of the value rows, weighted by how similar its query is to each key.

Self-attention: Q, K, V all derive from the same sequence. Cross-attention: Q from one sequence, K and V from another (encoder-decoder).

### Multi-head attention

Run attention $h$ times in parallel with different projections, concatenate, project back. Each head learns a different relation.

### Causal masking

In decoder-only models, mask out positions to the right so the model cannot see future tokens when predicting the next. Essential for autoregressive generation.

### Positional encoding

Attention is permutation-equivariant; without positional information, it cannot distinguish sequences. Options:

- **Sinusoidal** (original Transformer). Fixed, parameter-free.
- **Learned absolute**. Parameters per position.
- **Relative** (Shaw, Transformer-XL). Encodes pairwise offsets.
- **RoPE** (Rotary Position Embedding). Rotates Q, K by a frequency dependent on position. Standard in modern LLMs (Llama, Mistral, Qwen).
- **ALiBi**. Adds a linear bias to attention scores based on position distance. Extrapolates well beyond training context.

## Training stages

### 1. Pretraining

Self-supervised on massive text. Next-token prediction loss. Produces a base model with general language capability.

### 2. Supervised fine-tuning (SFT)

Train on curated instruction-response pairs. Teaches the model to follow instructions.

### 3. Alignment

Reinforcement learning from human feedback (RLHF) or direct preference optimization (DPO). Ranks model outputs by human preference; trains the model to produce preferred outputs.

### 4. Specialty fine-tuning

Domain adaptation, tool use, safety refusals. Often done via LoRA or full fine-tuning for enterprise use cases.

## Prompting patterns

Prompting is the primary interface for non-training use. Core patterns:

### Zero-shot

Ask directly. Works for simple, common tasks.

### Few-shot

Provide 2 to 10 examples before the real query. Helps with format, style, or edge cases the model might miss.

### Chain-of-thought (CoT)

Prompt the model to reason step by step. The original Wei et al. (2022) finding was that CoT was emergent in base models above roughly 70B parameters; smaller base models produced incoherent reasoning. By 2026 this characterization is outdated: reasoning-distilled small models (DeepSeek-R1-Distill at 7B and 14B, Qwen2.5-Math, Phi-4) explicitly produce coherent CoT below 7B. The correct framing: **CoT is emergent in base models at large scale but can be distilled into small models via supervised fine-tuning on reasoning traces from a stronger model**. For new projects, prefer a reasoning-distilled small model when latency or cost constrains you.

### Self-consistency

Sample multiple CoT answers, pick the majority. Reduces variance.

### Structured output

Request JSON, XML, or specific keywords. Use JSON mode or function calling on supported APIs. For open-weight models, grammar-constrained decoding (Outlines, guidance) enforces syntactic validity.

### Tool use

The model calls named functions to access data, compute, or take actions. The tool call is a JSON object the model emits; the runtime executes and returns the result. Agent frameworks (LangGraph, AutoGen, native Claude/OpenAI tool APIs) orchestrate the loop.

## Retrieval-Augmented Generation (RAG)

The LLM answers grounded in retrieved context instead of relying on its training weights.

Architecture:

1. **Chunking.** Split documents into chunks (200 to 1,000 tokens).
2. **Embedding.** Encode each chunk with a sentence-transformer.
3. **Index.** Store embeddings in a vector database (FAISS, Chroma, pgvector, Pinecone).
4. **Retrieval.** Embed the query, find top-k nearest chunks.
5. **Prompt.** Construct a prompt with the retrieved context and the query.
6. **Generation.** LLM produces the answer.

Evaluation:

- Retrieval metrics: Recall@k, MRR.
- Generation metrics: faithfulness (stays within context), answer relevance (addresses the question), context recall (context contains the answer).

Frameworks: LangChain, LlamaIndex, Haystack. All three have matured; pick the one whose abstractions you find cleanest.

Common failure modes:

- Chunks too small to contain the answer.
- Chunks too large, retrieval precision drops.
- Retrieval returns relevant-looking but wrong documents (synonym mismatch, lexical vs semantic).
- Context ordering; LLMs give more weight to beginning and end of context (middle is weaker).

## Fine-tuning vs prompting vs RAG

| Technique | When to use |
|-----------|-------------|
| Prompting | Default. Tasks well within pretraining distribution. |
| Few-shot | Prompting is not consistent; the task has a specific format. |
| Fine-tuning | Prompt is long, inference is expensive at scale, or style must be enforced consistently. |
| RAG | The task requires current or specialized knowledge, and hallucination must be grounded. |
| Combine | Real systems use all three: RAG for knowledge, fine-tuning for style, prompting for orchestration. |

## Evaluation

Offline evaluation of LLMs is hard. Options:

- **Benchmark datasets.** MMLU, HellaSwag, TruthfulQA, HumanEval, GSM8K. Known contamination issues.
- **Task-specific curated eval.** 100 to 500 real examples with expert answers. Slow but faithful.
- **LLM-as-judge.** Ask a stronger LLM to score outputs. Cheap and scalable; be aware of biases (positional, verbosity, self-preference).
- **Human preference testing.** Gold standard for quality but expensive.

## Cost and latency

For any LLM project, budget explicitly:

- **Cost per 1M tokens.** Varies 100x across models. Claude Haiku, Gemini Flash, GPT-4o-mini are cheap; frontier models are expensive.
- **First-token latency.** Time to first token matters for chat UX. Streaming hides the total latency.
- **Throughput.** Tokens per second. Matters for batch inference.
- **Context length.** Larger context is more expensive per call.

For local deployment (Ollama, vLLM), constraints are hardware-bound. A 7B model in FP16 needs about 14 GB VRAM; Q4 quantization drops this to ~5 GB but with some quality loss.

## Agentic systems

An agent = LLM + tools + loop. The model takes an action, observes the result, takes the next action, until task is complete.

Failure modes:

- **Infinite loops.** Add max-iteration limits.
- **Hallucinated tool calls.** Validate tool names and arguments.
- **Cascading errors.** One bad tool call corrupts subsequent reasoning. Use step-level evaluation.
- **Latency.** Each agent iteration is a full LLM call. Budget accordingly.

## Safety

- **Prompt injection.** Untrusted input (from RAG or user) tries to override system instructions. Mitigate with input isolation, output classifiers, reduced tool permissions.
- **Jailbreaks.** Users try to extract prohibited outputs. Production systems layer multiple defenses.
- **Data leakage.** Model may emit PII or secrets from training or context. Mask inputs; filter outputs.
- **Hallucination.** LLMs will confidently fabricate. Ground with RAG, verify with external checks, surface uncertainty.

## Mexican market notes

- Multilingual support matters. Test on Spanish, not just English. Performance drops are real.
- Mexican Spanish has distinct vocabulary (chela, chido, neta). Models trained on Spain-centric corpora underperform.
- Local LLMs avoid data residency concerns for regulated sectors (banking, health).
- Ollama + Llama 3.1 8B Instruct with Spanish fine-tuning or prompt engineering is a reasonable starting point.

## Linking to the portfolio

- `llm-rag-qa`: applies the RAG pattern on a Spanish corpus, served via local Ollama.
- `genai-agent-tool-use`: implements the agent loop with two tools, evaluated on 10 real tasks.
- `nlp-reviews-multilingual`: uses transformers but not LLMs; benchmark against an LLM zero-shot baseline in the theory writeup.

## Production inference: KV-cache, MoE, speculative decoding

### KV-cache

Autoregressive generation produces one token at a time. Naively, each new token requires recomputing self-attention over the full prefix, costing $O(n^2)$ per token. The KV-cache stores the keys and values from prior positions; generating token $n+1$ only requires computing the new query and attending against cached K, V. Per-token cost drops to $O(n)$.

Memory cost is significant: cache size scales as $L \cdot n \cdot 2 \cdot d_{\mathrm{kv}} \cdot \mathrm{bytes\_per\_param}$ where $L$ is the number of layers and $d_{\mathrm{kv}}$ is the per-head dimension times head count. A 70B model at 8K context easily occupies tens of GB of cache memory per request. PagedAttention (vLLM) and continuous batching are the production responses.

### Mixture of Experts (MoE)

Sparse routing: instead of every token activating every parameter, a learned router sends each token to a small subset (typically 1 or 2 of $E$ experts). Total parameter count is large but per-token compute is small.

- **GShard** (Lepikhin et al. 2020) demonstrated MoE at scale.
- **Switch Transformer** (Fedus et al. 2021) simplified to top-1 routing.
- **Mixtral 8x7B** popularized open-weight MoE.

Frontier proprietary models (GPT-4, Gemini Ultra) are widely speculated to be MoE. The economic argument: MoE gives you the quality of a dense model with several-times more parameters at the inference cost of a much smaller dense model. The engineering cost: load balancing across experts, expert parallelism across GPUs, capacity factor tuning to avoid token dropping.

### Speculative decoding

Leviathan, Kalman, Matias (2023). Use a small draft model to propose $k$ tokens cheaply, then verify with the target model in a single forward pass. Accepted tokens commit; the first rejection rolls back and the target model resamples.

Latency reduction is typically 2x to 3x with no quality loss when the draft model agrees with the target frequently. Production inference stacks (vLLM, TensorRT-LLM) implement this transparently.

## Long context: RoPE scaling, sliding window, SSMs

The original RoPE was trained on 2K to 8K tokens; naive extrapolation to 32K or 128K degrades quality. Three families of fixes:

- **YaRN** (Peng et al. 2024). Frequency-dependent interpolation of RoPE; preserves high-frequency rotations, scales low-frequency. Standard for context extension via continued pretraining.
- **Dynamic NTK scaling.** Adaptive interpolation factor based on actual sequence length. Cheap, no retraining required.
- **Sliding-window attention** (Mistral, Longformer). Each token attends only to the last $W$ tokens. Drops global context but linear in sequence length.
- **State-space models (Mamba, Mamba-2).** Replace attention with linear-time recurrent state updates. $O(n)$ inference vs attention's $O(n^2)$. Competitive on long-context tasks; integration with attention (Jamba) is the current pragmatic choice.

## Embedding model selection for RAG

Most RAG quality wins come from the embedding model, not the LLM. Considerations:

- **Bi-encoder vs cross-encoder.** Bi-encoder embeds query and document independently; fast retrieval but lower precision. Cross-encoder concatenates and scores; slow but high-precision. Production: bi-encoder for retrieval, cross-encoder for reranking.
- **MTEB benchmark.** Massive Text Embedding Benchmark (Muennighoff et al. 2022). Standard for embedding model selection; track the leaderboard.
- **Multilingual coverage.** For Spanish content, prefer multilingual embedding models trained explicitly for cross-lingual retrieval: mE5 (Microsoft), BGE-M3 (BAAI), Cohere Embed v3 multilingual. Mexican Spanish lands well on these because they are trained on diverse Spanish-speaking corpora.
- **Dimension and cost.** Larger embeddings (3072 in OpenAI text-embedding-3-large) marginally improve retrieval but multiply storage and compute. Matryoshka embeddings (Kusupati et al. 2022) let you truncate embeddings post-hoc to any prefix length.

## Rerankers

After top-$k$ bi-encoder retrieval (typically $k = 50$ to 200), apply a cross-encoder reranker to score query-document pairs and keep the top 5 to 10 for the LLM context.

- **Cross-encoder rerankers.** BGE-Reranker, Cohere Rerank, Jina Reranker. 100x slower per pair than bi-encoder but typically 5 to 15 percentage points NDCG improvement.
- **ColBERT** (Khattab, Zaharia 2020). Late-interaction retrieval: encode tokens separately and compute MaxSim at query time. Quality between bi-encoder and full cross-encoder, latency closer to bi-encoder. ColBERTv2 with PLAID indexing scales to large corpora.

The latency bill is real: a reranker step adds 100 to 500 ms per query. Worth it for high-stakes retrieval (legal, medical, regulatory); often not worth it for casual chatbots.

## Quantization

Inference-time quantization reduces parameter precision from FP16 (2 bytes) to lower formats:

- **GPTQ** (Frantar et al. 2022). Layer-wise quantization with second-order error correction. Common for 4-bit weights.
- **AWQ** (Lin et al. 2023). Activation-aware quantization preserving important weights at higher precision.
- **GGUF** (llama.cpp format). Container format supporting Q4_K, Q5_K, Q6_K, Q8 quantization variants. Standard for CPU and consumer-GPU local inference.

Empirically, Q8 weights match FP16 perplexity within noise. Q4 typically shows a 1 to 5 percent perplexity increase, sometimes more on reasoning-heavy tasks. The 14 GB FP16 footprint of a 7B model drops to about 4 to 5 GB at Q4, enabling consumer-laptop inference.

## Hallucination measurement: RAGAS and FActScore

Beyond LLM-as-judge, two structured frameworks:

### RAGAS metrics

Es et al. (2023). For RAG systems:

- **Faithfulness.** Are claims in the answer supported by the retrieved context? Computed by extracting claims, then verifying each against context with an LLM.
- **Answer relevance.** Does the answer address the question? Reverse-engineer the question from the answer; compare to the original.
- **Context recall.** Does the retrieved context contain the information needed to answer? Compares context against a ground-truth answer.
- **Context precision.** Are retrieved chunks ranked correctly?

Implementation: `ragas` Python library.

### FActScore

Min et al. (2023). For long-form generation. Decompose the response into atomic facts, verify each against a knowledge source, report the fraction supported. More granular than RAGAS for fact-heavy tasks.

## References

- Vaswani et al., "Attention Is All You Need", 2017.
- Brown et al., "Language Models are Few-Shot Learners" (GPT-3), 2020.
- Ouyang et al., "Training language models to follow instructions with human feedback" (InstructGPT), 2022.
- Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", 2020.
- Touvron et al., "Llama 2: Open Foundation and Fine-Tuned Chat Models", 2023.
- Rafailov et al., "Direct Preference Optimization" (DPO), 2023.
- Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding", 2021.
- Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models", NeurIPS 2022.
- Leviathan, Kalman, Matias, "Fast Inference from Transformers via Speculative Decoding", ICML 2023.
- Fedus, Zoph, Shazeer, "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity", JMLR 2022.
- Lepikhin et al., "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding", ICLR 2021.
- Peng et al., "YaRN: Efficient Context Window Extension of Large Language Models", ICLR 2024.
- Gu, Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2023.
- Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers", ICLR 2023.
- Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration", 2023.
- Khattab, Zaharia, "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT", SIGIR 2020.
- Muennighoff et al., "MTEB: Massive Text Embedding Benchmark", 2022.
- Es et al., "RAGAS: Automated Evaluation of Retrieval Augmented Generation", 2023.
- Min et al., "FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation", EMNLP 2023.
- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (vLLM), SOSP 2023.

## Exercises

1. Explain why a KV-cache is necessary for efficient autoregressive generation. What is the memory cost per token?
2. A company claims chain-of-thought improves their 1B-parameter model. Is this plausible in 2026? Under what conditions?
3. Compare bi-encoder and cross-encoder retrieval for RAG. When does the latency cost of a cross-encoder reranker justify itself?
4. What is the difference between GPTQ and GGUF quantization? Which would you use for local inference on a laptop?
5. Your RAG system has a faithfulness score of $0.62$ on RAGAS. Name two interventions to improve it.
