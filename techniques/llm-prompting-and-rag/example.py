"""Minimal RAG demo with sentence-transformers + Chroma + Ollama.

Requires Ollama running locally with a model pulled.
Falls back to printing the retrieved context if Ollama is unavailable.
"""

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


DB_DIR = Path(__file__).parent / "chroma_example"

DOCUMENTS = [
    "Mexico City is the capital of Mexico.",
    "The Constitution of Mexico was promulgated in 1917.",
    "UNAM stands for Universidad Nacional Autonoma de Mexico.",
    "LightGBM is a gradient boosting framework.",
    "PyTorch is an open-source machine learning library.",
]


def build_store():
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_or_create_collection(name="demo")
    embeddings = model.encode(DOCUMENTS, normalize_embeddings=True).tolist()
    collection.upsert(
        ids=[f"doc_{i}" for i in range(len(DOCUMENTS))],
        embeddings=embeddings,
        documents=DOCUMENTS,
    )
    return model, collection


def search(model, collection, query: str, k: int = 3):
    q_emb = model.encode([query], normalize_embeddings=True)[0].tolist()
    res = collection.query(query_embeddings=[q_emb], n_results=k)
    return res["documents"][0]


if __name__ == "__main__":
    model, collection = build_store()
    query = "Which university issued the Mexican constitution?"
    hits = search(model, collection, query)
    print(f"Query: {query}")
    for i, h in enumerate(hits, start=1):
        print(f"  [{i}] {h}")
