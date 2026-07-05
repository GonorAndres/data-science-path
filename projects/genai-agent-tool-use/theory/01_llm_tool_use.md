# LLM Tool Use Mechanics

## The core loop

```
System prompt (with tool schemas)
  +
User task
  |
  v
LLM proposes: { "tool": "sql_query", "arguments": {...} }
  |
  v
Runtime executes tool, returns result
  |
  v
LLM sees result, proposes next action
  |
  v
... repeat until LLM emits { "final_answer": "..." }
```

## System prompt

The system prompt describes:

- The task domain.
- Available tools and their schemas.
- Output format (JSON).
- Constraints and safety rules.

Example (abbreviated):

```
You are a data analyst agent. You have access to these tools:

1. sql_query(query: str) -> list of rows
2. web_fetch(url: str) -> text content

Respond with JSON only. One of:
  { "tool": "sql_query", "arguments": { "query": "..." } }
  { "tool": "web_fetch", "arguments": { "url": "..." } }
  { "final_answer": "..." }

Constraints:
  - SQL queries must be read-only (SELECT only).
  - URLs must start with "https://" and match the allowlist.
  - Maximum 8 tool calls per task.
```

## Parsing the output

Two common approaches:

1. **Raw JSON.** Parse with `json.loads`. Simple. Breaks on extra prose.
2. **Grammar-constrained decoding.** Use `outlines` or `guidance` to force valid JSON. Reliable but adds dependencies.

For this project, raw JSON plus a small retry-on-parse-error pattern.

## Error handling

- **Tool fails.** Return the error as a tool result; LLM can retry or adjust.
- **LLM produces invalid JSON.** Retry with an explicit reminder.
- **Step limit reached.** Halt, emit partial answer with warning.

## Format sensitivity

Small models (7-8B) are more format-sensitive. Tips:

- Put the schema inline in the system prompt, not only in documentation.
- Show 1-2 few-shot examples of correct output.
- Use a stop sequence to prevent rambling.

## Cross-links

- `00_problem_framing.md`
- `03_agent_evaluation.md`
- `04_safety_guardrails.md`
- `docs/theory/09_llm_foundations_2026.md`
