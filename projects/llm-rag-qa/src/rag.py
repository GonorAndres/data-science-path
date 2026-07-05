"""End-to-end RAG: retrieve + generate."""

import sys

from .generate import generate
from .store import VectorStore


PROMPT_TEMPLATE = """Responde la pregunta usando SOLO el contexto proporcionado.
Si la respuesta no esta en el contexto, di "No encuentro la respuesta en el material disponible."
Cita los chunks que usaste entre corchetes: [1], [2], ...

Contexto:
{context}

Pregunta: {question}

Respuesta:"""


def answer(question: str, k: int = 5) -> dict:
    store = VectorStore()
    results = store.search(question, k=k)
    context = "\n\n".join(
        f"[{i + 1}] ({r['metadata']['source']}) {r['text']}"
        for i, r in enumerate(results)
    )
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    output = generate(prompt)
    return {
        "question": question,
        "answer": output,
        "chunks": results,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m src.rag \"your question here\"")
        sys.exit(1)
    q = sys.argv[1]
    out = answer(q)
    print("Q:", q)
    print("A:", out["answer"])
    print("Chunks:")
    for i, c in enumerate(out["chunks"], start=1):
        print(f"  [{i}] {c['metadata']['source']} :: {c['text'][:120]}...")
