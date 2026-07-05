"""Agent tool-use loop."""

import json
import sys

from .llm import chat
from .prompts import SYSTEM_PROMPT
from .tools.sql import sql_query
from .tools.web import web_fetch


MAX_STEPS = 8


def try_parse_json(text: str) -> dict | None:
    try:
        return json.loads(text)
    except Exception:
        pass
    # Salvage: find the first balanced JSON object.
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                snippet = text[start : i + 1]
                try:
                    return json.loads(snippet)
                except Exception:
                    return None
    return None


def call_tool(name: str, arguments: dict) -> str:
    if name == "sql_query":
        rows = sql_query(arguments["query"])
        return json.dumps({"rows": rows}, default=str)[:4000]
    if name == "web_fetch":
        text = web_fetch(arguments["url"])
        return json.dumps({"content": text})
    raise ValueError(f"Unknown tool: {name}")


def run(task: str) -> dict:
    history = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    trajectory = []

    for step in range(MAX_STEPS):
        response = chat(history)
        parsed = try_parse_json(response)
        trajectory.append({"step": step, "response": response, "parsed": parsed})

        if parsed is None:
            history.append({"role": "assistant", "content": response})
            history.append({
                "role": "user",
                "content": "Your response was not valid JSON. Respond with ONE valid JSON object.",
            })
            continue

        if "final_answer" in parsed:
            return {
                "task": task,
                "answer": parsed["final_answer"],
                "trajectory": trajectory,
                "steps_used": step + 1,
            }

        tool = parsed.get("tool")
        args = parsed.get("arguments", {})
        try:
            result = call_tool(tool, args)
        except Exception as e:
            result = json.dumps({"error": str(e)})
        history.append({"role": "assistant", "content": response})
        history.append({"role": "user", "content": f"TOOL RESULT: {result}"})

    return {
        "task": task,
        "answer": "(step limit reached without final answer)",
        "trajectory": trajectory,
        "steps_used": MAX_STEPS,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m src.agent \"your task here\"")
        sys.exit(1)
    out = run(sys.argv[1])
    print("Answer:", out["answer"])
    print(f"Steps used: {out['steps_used']}")
