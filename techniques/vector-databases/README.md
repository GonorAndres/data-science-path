---
tags: [vector-db, ann, retrieval]
status: draft
created: 2026-04-23
updated: 2026-04-23
---

# Vector Databases

## What it is

Databases optimized for approximate nearest neighbor (ANN) search over dense vectors.

## Why it matters

Embeddings power RAG, recommenders, visual search, fraud detection via similarity. Scanning all vectors linearly is infeasible above millions; ANN indexes give sub-linear query time at high recall.

## Options

- **FAISS.** Library, not a service. Fast, flexible. In-memory.
- **Chroma.** Lightweight, SQLite-backed; Python-friendly.
- **Qdrant.** Rust-based, hybrid filter + vector search.
- **Milvus.** Designed for billion-scale.
- **Weaviate.** Multi-modal, knowledge-graph integration.
- **pgvector.** Postgres extension. Good for transactional + vector workloads.
- **Pinecone.** Managed SaaS.
- **ScaNN.** Google's library.

Choice depends on scale, hosting preference, and whether you need structured filters alongside vector queries.

## Index types

- **HNSW.** Graph-based; high recall, moderate memory.
- **IVF-PQ.** Inverted file + product quantization; lower memory, lower recall.
- **DiskANN.** SSD-backed for massive scale.

## Metrics

Inner product, cosine (dot after L2-normalize), or L2. Match the embedding model's training metric.

## Gotchas

- Index build time grows with scale; plan for offline rebuilds.
- Metadata filters can dominate query cost if not indexed well.
- Updating a vector often requires re-indexing.

## Resources

- FAISS docs: github.com/facebookresearch/faiss/wiki.
- `docs/theory/09_llm_foundations_2026.md`.
- Benchmarks: ann-benchmarks.com.
