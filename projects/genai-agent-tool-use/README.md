# genai-agent-tool-use

## Resumen (ES)

**Problema:** construir un agente LLM local con dos herramientas (consulta SQL sobre la base de `sql-analytics-ecommerce` y busqueda web limitada) que resuelva tareas de analisis.

**Metodo:** Ollama + Llama 3.1 8B Instruct, loop tool-use desde cero con salidas JSON estructuradas, evaluador de 10 tareas, guardrails de seguridad.

**Resultado objetivo:** agente pasa 7/10 tareas de prueba, latencia bajo 10 segundos por turno, documentacion del costo-cero y tradeoffs vs API.

## Summary (EN)

**Problem:** build a local LLM agent with two tools (SQL query over the `sql-analytics-ecommerce` DB, and bounded web fetch) that solves analysis tasks.

**Method:** Ollama + Llama 3.1 8B Instruct, a from-scratch tool-use loop with structured JSON outputs, a 10-task evaluator, and safety guardrails.

**Target result:** agent passes 7/10 evaluation tasks, per-turn latency under 10 seconds, documented zero-cost and tradeoffs vs API.

---

## Tools

1. **sql_query**. Runs a read-only SQL query against the Olist DuckDB warehouse (built by `sql-analytics-ecommerce`). Input: SQL string. Output: JSON rows (capped at 50).
2. **web_fetch**. Fetches a URL from an allowlist (e.g., wikipedia.org). Input: URL. Output: cleaned plain text snippet.

## Structure

```
genai-agent-tool-use/
├── README.md
├── theory/
│   ├── 00_problem_framing.md
│   ├── 01_llm_tool_use.md
│   ├── 02_local_vs_api_tradeoffs.md
│   ├── 03_agent_evaluation.md
│   └── 04_safety_guardrails.md
├── src/
│   ├── __init__.py
│   ├── agent.py              # tool-use loop
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── sql.py
│   │   └── web.py
│   ├── llm.py                # Ollama client
│   ├── prompts.py
│   └── evaluate.py
├── eval/
│   └── tasks.jsonl           # 10 tasks
├── tests/
├── requirements.txt
└── Makefile
```

## Quickstart

Prereqs: Ollama installed and `llama3.1:8b-instruct-q4_K_M` pulled. The SQL project's DuckDB file available at the default path.

```bash
pip install -r requirements.txt
python -m src.agent "Cual fue el mes con mayor revenue?"   # single task
python -m src.evaluate                                     # run all 10 eval tasks
```

## Resource requirements

- **CPU.** Works (~20 tokens/s on an 8-core CPU).
- **GPU.** Dramatically faster (~100 tokens/s on consumer GPU).
- **RAM.** ~6 GB for the 8B quantized model plus ~1 GB overhead.

## References

- `docs/theory/09_llm_foundations_2026.md`
- Ollama docs.
