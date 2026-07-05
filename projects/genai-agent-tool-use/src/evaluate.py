"""Run eval tasks and print a summary."""

import json
from pathlib import Path

from .agent import run


EVAL_PATH = Path(__file__).resolve().parent.parent / "eval" / "tasks.jsonl"


def load_tasks() -> list[dict]:
    if not EVAL_PATH.exists():
        return []
    return [json.loads(line) for line in EVAL_PATH.open(encoding="utf-8")]


def check(task: dict, answer: str) -> bool:
    target = task.get("expected_substring")
    if target and target.lower() in answer.lower():
        return True
    refusal = task.get("should_decline", False)
    if refusal:
        markers = ["no encuentro", "no puedo", "fuera de", "cannot", "don't know"]
        return any(m in answer.lower() for m in markers)
    return False


def main() -> None:
    tasks = load_tasks()
    if not tasks:
        print(f"no tasks at {EVAL_PATH}")
        return

    passed = 0
    for task in tasks:
        out = run(task["prompt"])
        ok = check(task, out["answer"])
        passed += int(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] ({task['id']}) -> {out['answer'][:120]}")

    print(f"\npassed {passed}/{len(tasks)}")


if __name__ == "__main__":
    main()
