"""Chunk the corpus into overlapping passages."""

import json
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CHUNK_SIZE = 400   # words
OVERLAP = 80


def split_words(text: str) -> list[str]:
    return text.split()


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    words = split_words(text)
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks


def run() -> None:
    in_path = DATA_DIR / "corpus.jsonl"
    out_path = DATA_DIR / "chunks.jsonl"
    if not in_path.exists():
        raise FileNotFoundError(f"{in_path} not found. Run data_download.py first.")

    n_chunks = 0
    with in_path.open(encoding="utf-8") as f_in, out_path.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            doc = json.loads(line)
            chunks = chunk_text(doc["text"])
            for i, c in enumerate(chunks):
                f_out.write(json.dumps({
                    "id": f"{doc['title']}::{i}",
                    "source": doc["title"],
                    "text": c,
                }, ensure_ascii=False) + "\n")
                n_chunks += 1
    print(f"wrote {n_chunks} chunks to {out_path}")


if __name__ == "__main__":
    run()
