# Retrieval Evaluation

## Two layers of evaluation

1. **Retrieval.** Did we find the right chunks?
2. **Generation.** Given the chunks, did the answer stay grounded and address the question?

Both matter. A good retriever plus a bad generator gives wrong-feeling answers; a bad retriever plus a strong generator produces confident hallucinations.

## Retrieval metrics

### Recall@K

Of the gold chunks (chunks labeled relevant for a query), how many are in the top-K retrieved?

$$
\text{Recall@K} = \frac{|\text{gold} \cap \text{top-K}|}{|\text{gold}|}
$$

### Precision@K

Of the top-K retrieved, how many are gold?

$$
\text{Precision@K} = \frac{|\text{gold} \cap \text{top-K}|}{K}
$$

### MRR (Mean Reciprocal Rank)

$1/\text{rank}$ of the first gold chunk, averaged across queries.

### NDCG@K

Discounted cumulative gain; higher weight to higher ranks.

## Generation metrics

### Faithfulness

Share of claims in the generated answer that are supported by the retrieved context.

Measured by:

- **Manual review.** Gold standard, slow.
- **LLM-as-judge.** A stronger LLM scores whether each claim is supported.
- **NLI (Natural Language Inference).** An entailment model checks if context entails each claim.

### Answer relevance

Does the answer actually address the question?

- Similar to faithfulness but anchored to the question, not the context.
- LLM-as-judge or a simple similarity score between question and answer.

### Context recall

Of the gold context (human-curated ideal context), how much is in the retrieved chunks?

Requires labeling ideal context per question. Manual work; smaller eval set.

## Building an eval set

For this project:

1. 50 questions written by hand, covering:
   - Easy factual (date, name, definition).
   - Multi-hop (requires combining two chunks).
   - Negative ("not in corpus") tests.
   - Edge cases (ambiguous wording, pronouns, dates).
2. For each, a gold answer and expected source chunks.
3. Store in `eval/questions.jsonl`.

## RAGAS framework

`ragas` is a Python library for automated RAG eval. It provides:

- `context_precision`
- `context_recall`
- `faithfulness`
- `answer_relevance`
- `answer_similarity`

All use an LLM-as-judge under the hood. Good starting point; cross-check with manual review for the first few dozen.

## Monitoring production

When the system is live:

- Log retrieved chunks and answers.
- Sample for manual review weekly.
- Track user-provided feedback (thumbs up/down).
- Alert on faithfulness or context-recall drops.

## Common mistakes

1. Reporting only retrieval metrics, ignoring generation.
2. Using LLM-as-judge without checking its calibration against human labels.
3. Overfitting to the eval set. Keep a held-out set for final reports.
4. Ignoring negative tests (questions the system should refuse). Hallucination is often worst on these.

## Cross-links

- `01_rag_architecture.md`
- `02_embeddings_and_vector_stores.md`
- `04_prompt_patterns.md`
