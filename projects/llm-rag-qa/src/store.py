"""Thin Chroma wrapper used by the pipeline."""

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_DIR = Path(__file__).resolve().parent.parent
DB_DIR = PROJECT_DIR / "chroma_db"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(DB_DIR))
        self.collection = self.client.get_collection(name="corpus")
        self.model = SentenceTransformer(MODEL_NAME)

    def search(self, query: str, k: int = 5) -> list[dict]:
        embedding = self.model.encode([query], normalize_embeddings=True)[0].tolist()
        res = self.collection.query(query_embeddings=[embedding], n_results=k)
        return [
            {
                "id": res["ids"][0][i],
                "text": res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
                "distance": res["distances"][0][i] if "distances" in res else None,
            }
            for i in range(len(res["ids"][0]))
        ]
