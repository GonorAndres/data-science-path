# Problem Framing: Local RAG QA

## The problem

A user asks a factual question in Spanish. The answer should be grounded in a specific corpus (not the LLM's general training knowledge). If the corpus does not contain the answer, the system should say so.

This mirrors real use cases at companies:

- Internal knowledge base QA (policies, procedures, product docs).
- Legal document QA.
- Customer support over product docs.
- Research assistants over scientific literature.

## Why local LLMs

This project uses Ollama with a 7-8B open-weight model instead of Claude or GPT. Tradeoffs:

- **Cost.** Zero marginal cost per query. API models are fractions of a cent per query but compound at scale.
- **Privacy.** Queries and context stay on the user's machine or your infrastructure.
- **Latency.** 1 to 5 seconds per generation on consumer hardware; API is usually 2 to 10 seconds plus network.
- **Quality.** API models are stronger at reasoning and long-form generation. Local 8B models are strong enough for retrieval-grounded short-form QA.

For many enterprise use cases, local is the right choice. For a portfolio project, it signals comfort with both worlds.

## Non-goals

- Not building a general-purpose chatbot (RAG-only, factual).
- Not fine-tuning the LLM itself (prompting suffices for this task).
- Not handling multi-hop reasoning (single-hop retrieval is enough).
- Not building agent tool-use (that is `genai-agent-tool-use`).

## Success metrics

- **Retrieval precision @5.** Of the top-5 retrieved chunks, how many are relevant.
- **Retrieval recall @5.** Of the relevant chunks in the corpus, how many are in top-5.
- **Faithfulness.** Of the claims in the generated answer, what share is supported by the retrieved context.
- **Answer relevance.** Does the answer address the question as asked.

## Stakeholders

- **End user.** Wants accurate, grounded answers in Spanish, fast.
- **Content team.** Wants to curate the corpus.
- **Compliance.** Wants auditable provenance (every answer traces back to specific chunks).

## Cross-links

- `docs/theory/09_llm_foundations_2026.md`: broader context.
- `01_rag_architecture.md`: architecture deep dive.
