"""Lightweight RAG evaluation.

Reads eval/questions.jsonl and computes:
- retrieval hit@5 (fraction of questions where the gold source is among top-5)
- simple LLM-as-judge faithfulness on a sample

Produces a summary CSV.
"""

import json
from pathlib import Path

from .generate import generate
from .rag import answer


PROJECT_DIR = Path(__file__).resolve().parent.parent
EVAL_PATH = PROJECT_DIR / "eval" / "questions.jsonl"
RESULTS_PATH = PROJECT_DIR / "eval" / "results.jsonl"


FAITHFULNESS_PROMPT = """Dada la siguiente respuesta y el contexto,
evalua si CADA afirmacion de la respuesta esta respaldada por el contexto.
Contesta solo 'YES' o 'NO'.

Contexto:
{context}

Respuesta:
{answer_text}

Resultado:"""


def load_questions() -> list[dict]:
    if not EVAL_PATH.exists():
        return []
    with EVAL_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def run() -> None:
    questions = load_questions()
    if not questions:
        print(f"No questions at {EVAL_PATH}. Create the file with questions.")
        return
    hits = 0
    faith = 0
    total = len(questions)

    with RESULTS_PATH.open("w", encoding="utf-8") as out:
        for q in questions:
            result = answer(q["question"])
            sources = {c["metadata"]["source"] for c in result["chunks"]}
            hit = q["gold_source"] in sources
            if hit:
                hits += 1

            ctx = "\n".join(c["text"] for c in result["chunks"])
            fj = generate(FAITHFULNESS_PROMPT.format(context=ctx, answer_text=result["answer"]))
            if "YES" in fj.upper():
                faith += 1

            out.write(json.dumps({
                **q,
                "answer": result["answer"],
                "sources": list(sources),
                "hit_at_5": hit,
                "faithfulness": fj,
            }, ensure_ascii=False) + "\n")

    print(f"hit@5: {hits}/{total} = {hits/total:.3f}")
    print(f"faithfulness (YES rate): {faith}/{total} = {faith/total:.3f}")


if __name__ == "__main__":
    run()
