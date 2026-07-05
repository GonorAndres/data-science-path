# Embeddings and Vector Stores

## What a semantic embedding is

A function that maps text to a dense vector in $\mathbb{R}^d$ (typically 384 to 1536 dimensions) such that semantically similar text lands near each other in vector space.

Unlike TF-IDF, embeddings capture meaning beyond keywords: "dog" and "perro" are close in a multilingual model; "bank" the institution and "bank" the river side are usually disambiguated by surrounding context.

## Training objectives

Modern embeddings come from:

- **Contrastive learning.** Pair similar texts (from NLI, paraphrase datasets), push dissimilar ones apart.
- **Instruction-tuning.** E5 family uses task prefixes ("query: ..." vs "passage: ...") to specialize the embedding per role.
- **Bi-encoder training from larger teacher models.** Distill a strong cross-encoder into a faster bi-encoder.

## Choosing an embedding model

Dimensions to compare:

- **Language coverage.** Multilingual models are necessary for Spanish. `paraphrase-multilingual-mpnet-base-v2` is a strong default.
- **Dimension.** Smaller dimension is faster; less memory.
- **Context length.** Some models cap at 512 tokens; newer ones extend to 8k or more.
- **Benchmark performance.** MTEB leaderboard is the standard (huggingface.co/spaces/mteb/leaderboard).

For this project, `paraphrase-multilingual-mpnet-base-v2` (768 dim, 512 tokens context).

## Vector stores: what the DB does

- **Store.** Embeddings + metadata + original text.
- **Index.** ANN structure (HNSW, IVF, PQ) for sub-linear-time nearest neighbor search.
- **Filter.** Metadata predicates (e.g., only documents from 2026).
- **Update.** Add, update, delete without rebuilding the index.

## Chroma basics

```python
import chromadb
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="corpus")
collection.add(
    embeddings=[[...]],
    documents=["chunk text"],
    metadatas=[{"source": "wiki", "date": "2026-01-01"}],
    ids=["chunk_001"],
)
results = collection.query(query_embeddings=[[...]], n_results=5)
```

For larger corpora, Chroma's SQLite-based persistence becomes slow; switch to FAISS or a managed service.

## Similarity metrics

- **Cosine similarity.** Standard. Normalize vectors, then dot product equals cosine.
- **Dot product.** Matches some models (e.g., sentence-transformers normalizes internally).
- **L2 (Euclidean).** Sometimes used; less common for sentence embeddings.

Match the metric the model was trained for. `paraphrase-multilingual-mpnet` uses cosine.

## Re-ranking with cross-encoders

A cross-encoder takes (query, chunk) as a single input and produces a relevance score. More expensive than bi-encoders (which embed query and chunk independently) but much more accurate.

Pattern:

1. Bi-encoder retrieves top-50.
2. Cross-encoder re-scores top-50.
3. Keep top-5.

Cross-encoder models: `cross-encoder/ms-marco-MiniLM-L-12-v2` (English-centric; multilingual variants exist).

## Index performance

For 10k to 100k chunks, any index works. For millions, choose carefully:

- **HNSW** (hierarchical navigable small world). High recall, moderate memory.
- **IVF_PQ**. Lower memory, lower recall. Good for billions.
- **ScaNN.** Google's index; fast and accurate.

Chroma defaults to HNSW. Fine here.

## Monitoring retrieval quality

Production RAG systems log:

- Retrieved chunk IDs per query.
- Whether the user clicked the citation.
- Whether the user marked the answer as helpful.

Offline eval runs regularly against a labeled set.

## Cross-links

- `01_rag_architecture.md`
- `03_retrieval_evaluation.md`
- `docs/theory/09_llm_foundations_2026.md`
