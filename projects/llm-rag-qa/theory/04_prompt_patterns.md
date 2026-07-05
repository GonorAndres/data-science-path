# Prompt Patterns for RAG

## The baseline prompt

```
You are a helpful assistant answering questions based ONLY on the provided context.
If the answer is not in the context, say you do not know.

Context:
{context}

Question:
{question}

Answer:
```

Works surprisingly well for the 8B-and-up open models.

## Pattern: cite chunk IDs

Ask the model to attribute each claim:

```
Answer the question using the provided context. For each claim, cite the chunk ID
in square brackets like [3]. If the answer is not in the context, say so.

Context:
[1] {chunk_1}
[2] {chunk_2}
[3] {chunk_3}

Question:
{question}

Answer:
```

Forces the model to keep grounding. Downstream you can parse the citations.

## Pattern: structured output

JSON for downstream use:

```
Respond in JSON with two fields: "answer" (string) and "citations" (list of chunk IDs).
Do not output anything outside the JSON.

...
```

For 8B models, JSON output is hit-or-miss; use a grammar-constrained decoder (Outlines, guidance) for reliability.

## Pattern: chain-of-thought

For complex questions:

```
Before answering, think step by step. Then give the final answer.
```

Often improves accuracy; sometimes verbose. Post-process to extract only the answer for display.

## Pattern: self-check

Ask the model to verify:

```
After writing your answer, review it: does every claim appear in the context?
If not, revise.
```

Mixed results; can help or hurt.

## Pattern: negative-example guardrail

Teach the model when to refuse:

```
Example 1:
Context: "Paris is the capital of France."
Question: "What is the population of Paris?"
Answer: "The population is not in the context."

Example 2:
Context: "Paris is the capital of France."
Question: "What is the capital of France?"
Answer: "Paris. [1]"
```

Reduces hallucination on out-of-corpus questions.

## Templating

Keep prompts in a separate file (`prompts.py` or similar). Makes A/B testing easier.

```python
PROMPT_V1 = """..."""
PROMPT_V2 = """..."""
```

Log which version was used per response for evaluation tracking.

## Guardrails

- **Input sanitization.** Strip or escape user queries that try to override the system prompt.
- **Output validation.** Check JSON parses; check citations reference valid chunks.
- **Length limits.** Cap context tokens and generation tokens.
- **Refusal training.** Explicit examples teach the model when to decline.

## Local-model quirks

- **Format sensitivity.** Small open models are more sensitive to prompt format than frontier models. Minor changes in phrasing can change outputs.
- **Chat template.** Ollama applies the model's default chat template; match it (Llama 3.1 has specific role tokens).
- **Determinism.** Set temperature=0 for reproducible outputs; relax to 0.2 for more natural phrasing.

## Cross-links

- `01_rag_architecture.md`
- `03_retrieval_evaluation.md`
- `docs/theory/09_llm_foundations_2026.md`
