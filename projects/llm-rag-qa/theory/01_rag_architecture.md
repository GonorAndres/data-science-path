# RAG Architecture

## The canonical pipeline

```
User Query
   |
   v
Embed query  -----+
                  |
                  v
        Vector store  ->  top-k chunks
                                |
                                v
                 Prompt template: context + question
                                |
                                v
                          LLM generate
                                |
                                v
                           Answer + citations
```

## Stages in detail

### 1. Chunking

Documents are split into passages (chunks). Why:

- LLMs have context limits.
- Retrieval precision is higher when chunks are focused.

Strategies:

- **Fixed-size.** 200 to 500 tokens per chunk. Simple.
- **Semantic.** Split on paragraph or sentence boundaries.
- **Sliding window.** Overlap chunks so boundary content is not lost.
- **Hierarchical.** Multiple chunk sizes; retrieve small, supply surrounding large.

Guideline: start with 300-500 tokens, 10-20 percent overlap. Tune if retrieval struggles.

### 2. Embedding

Each chunk becomes a dense vector. Model options:

- **OpenAI text-embedding-3-small / large.** Strong but paid.
- **sentence-transformers.** Local, free, many multilingual variants.
- **E5, BGE.** Open-weight, state-of-the-art in 2025 benchmarks.
- **Cohere embed v3.** Paid, multilingual.

This project uses `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` for Spanish+English support. ~420 MB, ~80 ms per encode on CPU.

### 3. Vector store

Stores embeddings, supports approximate nearest neighbor (ANN) search.

- **FAISS.** Library, not a service. Fast, in-memory.
- **Chroma.** Lightweight database, SQLite-backed. Default for small projects.
- **pgvector.** PostgreSQL extension. Good for transactional integration.
- **Pinecone / Weaviate / Qdrant.** Managed services.
- **Milvus.** Large-scale self-hosted.

This project uses Chroma for simplicity.

### 4. Retrieval

Given the query embedding, find top-k chunks by cosine similarity.

Extensions:

- **Hybrid search.** Combine dense (embedding) and sparse (BM25) retrieval. Often improves precision.
- **Reranking.** Retrieve top-50, then re-rank with a cross-encoder for the top-5 final. More compute, better quality.
- **Query expansion.** Rewrite the query (e.g., with an LLM) to improve coverage.
- **Metadata filtering.** Restrict by document type, date, author.

### 5. Prompting

A classic RAG prompt:

```
You are a helpful assistant. Answer the question using ONLY the provided context.
If the answer is not in the context, say you don't know.

Context:
{retrieved_chunks}

Question:
{query}

Answer:
```

Variations:

- Few-shot with example QA pairs.
- Chain-of-thought for multi-step reasoning.
- Structured output (JSON with answer + cited chunks).

### 6. Generation

The LLM produces the answer. For short factual QA, generation is the straightforward decoder pass.

For longer answers, streaming improves UX.

### 7. Response

Return the answer plus citations. Let the user click a citation to see the exact chunk it came from. Essential for trust.

## Common failure modes

- **Retrieval missed the relevant chunk.** The right answer exists in the corpus but never makes top-5.
- **LLM ignored the context.** Recites general knowledge instead of grounding.
- **LLM hallucinated.** Synthesizes claims not in the context.
- **Context too long, LLM focuses on irrelevant parts.** Known "lost in the middle" phenomenon.

Mitigations:

- Rerank retrieved chunks.
- Instruct the LLM explicitly to cite chunk IDs.
- Use stronger models for generation.
- Keep context focused (prefer 3 to 5 chunks over 10+).

## Cross-links

- `00_problem_framing.md`
- `02_embeddings_and_vector_stores.md`
- `03_retrieval_evaluation.md`
- `04_prompt_patterns.md`
- `docs/theory/09_llm_foundations_2026.md`
