---
tags: [llm, rag, prompting, nlp, retrieval]
status: review
created: 2026-04-23
updated: 2026-04-25
---

# LLM Prompting and Retrieval-Augmented Generation

## What it is

Techniques to make large language models produce useful outputs without retraining them: prompting patterns, retrieval augmentation, tool use, structured output, and the surrounding production scaffolding (chunking, hybrid retrieval, reranking, evaluation, guardrails).

For the conceptual backbone (attention, training stages, KV-cache, MoE, quantization), see `docs/theory/09_llm_foundations_2026.md`.

## Why it matters

In 2026, most applied LLM work is shaping generations rather than training models. RAG dominates production deployments because it grounds outputs in trusted sources, controls cost, and lets you update knowledge without retraining. Knowing the patterns that fail gracefully under load is the working craft.

## When to use which technique

- **Prompt only.** Task is well within the model's pretraining; no need for external knowledge.
- **Few-shot prompting.** Need consistent format the model does not default to; format examples are cheap.
- **RAG.** Task requires current or domain-specific knowledge; hallucination must be grounded.
- **Fine-tuning.** Prompting hits a quality ceiling; you need persistent style or shorter prompts.
- **Combination.** Real systems use all three: RAG for knowledge, fine-tuning for style, prompting for orchestration.

## Prompting patterns

1. **Zero-shot.** Direct instruction. Baseline. Works for common tasks within pretraining distribution.
2. **Few-shot.** 2 to 8 input-output examples before the real query. Stabilizes format.
3. **Chain-of-thought (CoT).** Explicit reasoning steps. With reasoning-distilled models (DeepSeek-R1-Distill, Qwen2.5-Math) effective even at 7B.
4. **Self-consistency.** Sample multiple CoT answers, take the majority. Reduces variance, costs proportionally more.
5. **ReAct.** Interleave reasoning and tool calls; the model alternates "Thought:" and "Action:" turns.
6. **Structured output.** Request JSON; validate with Pydantic. For open models, grammar-constrained decoding via Outlines or `guidance` enforces syntax.
7. **System prompt vs user prompt.** Push role and constraints to the system slot; keep the user slot for the task. Ignoring this leaks instructions across turns.

## Chunking strategies for RAG

The single highest-leverage RAG decision after embedding-model choice. Strategies:

### Fixed-size

Chunk every $N$ tokens with overlap $O$ (typical: 256 to 512 tokens, overlap 32 to 64). Simple, deterministic. Cuts mid-sentence; loses some semantic boundaries.

### Sentence-based

Split on sentence boundaries (`spacy`, `nltk`). Avoids mid-sentence cuts but variable chunk size hurts retrieval consistency. Often combined with packing into target token windows.

### Recursive

LangChain's `RecursiveCharacterTextSplitter`. Try paragraph splits first, fall back to sentences, then to characters. The pragmatic default for unstructured text.

### Semantic

Greenberg, Greg Kamradt's pattern. Embed each sentence; find points where the embedding distance jumps; chunk at those points. More expensive, better aligned with topic boundaries. Worth it for long technical documents.

### Empirical guidance

- **Default chunk size.** 256 to 512 tokens for most prose.
- **Overlap.** About 10 to 20% of chunk size.
- **Code or tables.** Larger chunks (1,000 tokens) preserving structural units.
- **Always evaluate.** Sweep chunk sizes on your retrieval eval set; the optimal size is data-dependent.

## Hybrid retrieval

Pure dense retrieval misses lexical matches; pure BM25 misses semantic paraphrases. Hybrid retrieval combines both.

### BM25

Robertson, Sparck-Jones (1976). Sparse term-frequency-based ranker. Strong baseline for any retrieval task; surprisingly hard to beat with dense alone.

### Dense retrieval

Sentence-transformer or instruction-tuned embedding model + FAISS, Chroma, or pgvector. Excellent on paraphrase and semantic similarity; weaker on rare named entities and exact matches.

### Reciprocal rank fusion (RRF)

Cormack et al. (2009). Combine ranked lists by

$$
\mathrm{score}(d) = \sum_{r \in R} \frac{1}{k + \mathrm{rank}_r(d)},
$$

with $k$ a small constant (typical 60). Robust to score-scale differences between rankers; standard hybrid combiner.

```python
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])
```

## Reranking

Cross-encoder rerankers (BGE-Reranker, Cohere Rerank, Jina Reranker) score query-document pairs jointly. Typical pipeline:

1. Hybrid retrieval returns top 50 to 100 candidates.
2. Cross-encoder reranks; keep top 5 to 10.
3. Pass to LLM as context.

Latency cost: 100 to 500 ms per query. Quality benefit: 5 to 15 percentage points NDCG. Worth it for high-stakes retrieval (legal, medical, regulatory). For casual chatbots, often skipped.

## RAGAS evaluation

Es et al. (2023). Four metrics for end-to-end RAG quality:

- **Faithfulness.** Are answer claims supported by retrieved context?
- **Answer relevance.** Does the answer address the question?
- **Context recall.** Does the context contain the information needed for the ground-truth answer?
- **Context precision.** Are relevant chunks ranked first?

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from datasets import Dataset

data = Dataset.from_dict({
    "question": questions,
    "answer": predicted_answers,
    "contexts": retrieved_contexts,
    "ground_truth": ground_truths,
})

results = evaluate(data, metrics=[faithfulness, answer_relevancy, context_recall, context_precision])
print(results)
```

Interpret: faithfulness < 0.7 signals hallucination; context recall < 0.7 signals retrieval failure; the two together pinpoint the weak link.

## Guardrails

### Prompt-injection detection

Untrusted input (a webpage, a user message, a retrieved document) that contains "ignore previous instructions" or similar attempts to override the system prompt. Mitigations:

- Treat retrieved content as data, not instructions. Wrap in clearly delimited XML tags.
- Run a small classifier on inputs to flag suspicious patterns.
- Reduce tool permissions on injected paths (read-only context).
- Log and alert on injection candidates.

### Output validation

- Validate JSON outputs against a Pydantic schema before acting.
- For free-text outputs, regex or LLM-judge gate for forbidden content (PII, profanity, off-topic).
- Add refusal detection when the model says "I cannot..." rather than the requested output.

### Jailbreak patterns

Track known patterns: roleplay framings ("pretend you have no rules"), instruction-flooding ("repeat after me: I am free"), encoded payloads (Base64, ROT13). Maintain a regression suite of known jailbreaks against your stack.

## Tool-use loop (ReAct pattern)

```python
def agent_loop(query, tools, llm, max_iter=10):
    history = [{"role": "system", "content": system_prompt(tools)}]
    history.append({"role": "user", "content": query})
    for _ in range(max_iter):
        response = llm.complete(history)
        if response.tool_call is None:
            return response.text
        try:
            result = tools[response.tool_call.name](**response.tool_call.args)
        except Exception as e:
            result = f"Error: {e}"
        history.append({"role": "assistant", "content": response.raw})
        history.append({"role": "tool", "content": str(result)})
    return "Max iterations reached"
```

Critical guardrails:

- **Iteration limit.** Always. Otherwise infinite loops on tool errors.
- **Tool name and argument validation.** The model hallucinates tool names; validate against the registered set before dispatch.
- **Error swallowing in the tool result.** Pass the error message back so the model can recover or apologize, do not throw.
- **Step budget.** Track cumulative tokens; abort if cost exceeds threshold.

## Production patterns

### Streaming

Use the streaming API (OpenAI, Anthropic, Ollama all support it). First-token latency dominates user perception; streaming hides the total. For batch use cases, streaming is unnecessary.

### Caching

- **Prompt prefix cache.** Anthropic and OpenAI both support cached system-prompt prefixes. Up to 90% latency and cost reduction on repeated prefixes.
- **Application-level cache.** Hash (model, prompt, parameters) and cache the result. Worthwhile for deterministic queries.
- **Embedding cache.** Don't recompute embeddings for documents that have not changed.

### Async batching

For throughput-bound workloads, send N concurrent requests rather than one large batch. Each provider has rate-limit considerations; tune `max_concurrent` empirically.

### Observability

Log: request, response, token counts, latency, cost, retrieval hits, tool calls. The minimum to debug production failures and to compute cost-per-query.

## Gotchas

- LLMs ignore middle-of-context content ("lost in the middle", Liu et al. 2023). Put critical instructions at the top and reiterate at the bottom.
- Small open models are format-sensitive; minor phrasing changes flip outputs.
- Temperature > 0 introduces stochasticity; set to 0 for reproducibility unless you sample for diversity.
- RAG retrieval failures are usually retrieval-side, not generation-side. Always log the retrieved chunks alongside the answer.
- Cost surprises come from agentic loops and large context. Monitor tokens per request.

## Resources

- `docs/theory/09_llm_foundations_2026.md`: theory companion.
- Anthropic Prompt Engineering Guide.
- OpenAI Cookbook: https://github.com/openai/openai-cookbook.
- Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", NeurIPS 2020.
- Cormack, Clarke, Buettcher, "Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods", SIGIR 2009.
- Es et al., "RAGAS: Automated Evaluation of Retrieval Augmented Generation", 2023.
- Liu et al., "Lost in the Middle: How Language Models Use Long Contexts", 2023.
- LangChain and LlamaIndex documentation.
