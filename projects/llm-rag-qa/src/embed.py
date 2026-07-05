"""Embed chunks with sentence-transformers; populate Chroma."""

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
DB_DIR = PROJECT_DIR / "chroma_db"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def run() -> None:
    chunks_path = DATA_DIR / "chunks.jsonl"
    if not chunks_path.exists():
        raise FileNotFoundError(f"{chunks_path} not found. Run src.chunk first.")

    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_or_create_collection(name="corpus")

    ids, texts, metadatas = [], [], []
    with chunks_path.open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            ids.append(obj["id"])
            texts.append(obj["text"])
            metadatas.append({"source": obj["source"]})

    print(f"embedding {len(texts)} chunks ...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    collection.upsert(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas,
    )
    print(f"upserted {len(ids)} into Chroma at {DB_DIR}")


if __name__ == "__main__":
    run()
