# Problem Framing: Local Agent with Tools

## What the agent does

Given a natural-language task, the agent:

1. Thinks about which tool to call.
2. Emits a structured tool call (JSON with tool name and arguments).
3. Receives the tool's output.
4. Decides: call another tool, or produce a final answer.
5. Repeats until done or until a step limit is reached.

## Why local

API agents (Claude, GPT) are easier to build but have cost, privacy, and latency constraints. Local agents with smaller models require more prompt engineering but:

- Cost zero per query.
- Keep data on-premises.
- Produce deterministic, auditable runs.
- Showcase engineering discipline: "I can make this work without paying OpenAI."

For this portfolio, the local-agent angle is a differentiator.

## Tasks in scope

Short analytical tasks combining SQL queries over the Olist warehouse with occasional web fetches. Examples:

- "Which month had the highest revenue?" (SQL only)
- "What is the Spanish Wikipedia definition of cohort analysis?" (web only)
- "Compare our average order value to industry benchmarks." (SQL + web)

Tasks deliberately avoid multi-step reasoning that 70B-class models handle better.

## Non-goals

- Not building a general-purpose agent framework.
- Not implementing vector memory or long-term state.
- Not doing agentic RAG (that is `llm-rag-qa`).
- Not connecting to live APIs beyond the allowlist.

## Success metrics

- **Task success rate.** Of 10 eval tasks, how many complete correctly.
- **Turn count.** How many tool calls per task on average.
- **Latency.** Wall-clock per task.
- **Hallucination rate.** On tasks with no answer in data, does the agent correctly say so.

## Cross-links

- `docs/theory/09_llm_foundations_2026.md`
- `01_llm_tool_use.md`
