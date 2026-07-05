# Agent Evaluation

## Why evaluation is harder for agents

For a classifier, you compare prediction to label. For an agent, there are many valid paths to the correct answer, partial credit matters, and the same run is sometimes non-deterministic.

## Levels of evaluation

### 1. End-to-end success

Did the final answer match the gold?

- Strict: exact match on answer.
- Soft: answer contains the key number / entity.
- Semantic: LLM-as-judge compares answer to gold.

### 2. Tool-use trajectory

Did the agent call the right tools in a reasonable order?

- Check whether `sql_query` was called when the answer required SQL.
- Count unnecessary tool calls.
- Track hallucinated tool names.

### 3. Cost and latency

- Total tokens generated (input + output).
- Total wall-clock time.
- Tool call count.

Even a correct answer is bad if it took 50 tool calls and 5 minutes.

## The eval set

Build 10 to 50 tasks covering:

- **Easy single-tool.** "How many orders in April 2018?" (SQL only).
- **Easy web-only.** "Summarize the Wikipedia article on cohort analysis." (web only).
- **Multi-step.** "Find the month with lowest revenue and look up what holidays happened that month."
- **Negative.** "What is Mars's GDP?" (should decline).
- **Ambiguous.** "Tell me about revenue." (should ask for clarification or pick a reasonable interpretation).

For each task, record:

- task_id
- natural language prompt
- expected answer (value or criterion)
- ideal tool sequence
- difficulty tag

## LLM-as-judge

For free-form answers, have a stronger LLM score each response on a rubric:

- Correctness (0-5)
- Completeness (0-5)
- Faithfulness to tool outputs (0-5)

Validate the judge against human ratings on a subset to confirm the judge is not systematically biased.

## Reporting

Per run:

- Overall success rate.
- Breakdown by task difficulty and tool category.
- Average tool calls per task.
- Average latency.
- Failure modes (parse errors, hallucinations, wrong tool choice, infinite loops).

Compare iterations of the agent prompt against a baseline prompt. Trend success over time.

## Common pitfalls

- Tiny eval set (5 tasks). One task swings the metric wildly.
- Overfitting prompts to the eval set. Keep a held-out subset.
- Ignoring latency. A 70 percent accurate agent that takes 5 minutes is not deployable.
- Not logging intermediate steps. When a task fails, you cannot diagnose without the trajectory.

## Cross-links

- `00_problem_framing.md`
- `02_local_vs_api_tradeoffs.md`
- `04_safety_guardrails.md`
