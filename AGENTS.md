# AGENTS.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Personal data science learning repository and portfolio. Owner is an actuarial science bachelor doing a data science internship. This repo serves as:
- **Technique reference**: Theory + working code for each method (techniques/)
- **Portfolio projects**: End-to-end analyses with business framing (projects/)
- **Sandbox**: Quick experiments and explorations (notebooks/exploratory/)
- **Reusable scripts**: Utility functions that get reused across projects (scripts/utils/)

## Repository layout

```
techniques/          -- One folder per technique (regression, clustering, etc.)
                        Each has: README.md (theory), .py scripts, .ipynb notebooks
projects/            -- Self-contained portfolio projects
                        Each has: README.md (business context + retrospective), code, outputs
scripts/utils/       -- Shared Python utilities (plotting helpers, data loaders, evaluation)
notebooks/exploratory/ -- Throwaway exploration notebooks
data/                -- raw/, processed/, external/ (not tracked in git)
docs/                -- theory/, presentations/, lessons-learned/
```

## Tech stack

- **Python** (primary): pandas, scikit-learn, xgboost, lightgbm, statsmodels, lifelines, plotly, seaborn
- **R** (secondary): Used for statistical modeling and actuarial-specific work
- **Jupyter notebooks** for exploration and presentation; `.py` scripts for reusable logic

## Commands

```bash
# Environment setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run a specific notebook
jupyter notebook notebooks/exploratory/

# Run a script
python scripts/utils/<script>.py

# R scripts (when applicable)
Rscript <path-to-script>.R
```

## Focus areas

1. **ML & Predictive Modeling** -- classification, regression, ensembles, model evaluation
2. **Customer Analytics** -- segmentation, churn, CLV, cohort analysis
3. **Actuarial / Risk Modeling** -- survival analysis, GLMs, reserving, pricing

## Secrets & Environment Variables -- MANDATORY

**NEVER commit secrets, API keys, passwords, tokens, or credentials to this repository.**

Before every commit, verify that no private variables are being staged. If a hardcoded secret is found anywhere in the codebase:
1. **Remove** the hardcoded value immediately
2. **Replace** it with an environment variable reference (e.g., `os.environ["DB_PASSWORD"]`)
3. **Add** the variable name and a placeholder to `.env.example` so collaborators know what to set
4. **Confirm** `.env` is listed in `.gitignore` (it already is -- `.env`, `*.env`, `.env.*`, `credentials.json`, `*.pem`, `*.key` are all gitignored)

Common patterns:
```python
# WRONG -- never do this
API_KEY = "sk-abc123..."
conn = psycopg2.connect(password="my_password")

# RIGHT -- always do this
import os
API_KEY = os.environ["API_KEY"]
conn = psycopg2.connect(password=os.environ["DB_PASSWORD"])
```

If you need a new secret variable, add it to `.env.example` with an empty or placeholder value before using it in code.

## Conventions

- Every technique folder must have a `README.md` following `techniques/TEMPLATE.md`
- Every project folder must have a `README.md` following `projects/TEMPLATE.md`
- Project READMEs include a **Retrospective** section (what went well, what to improve, skills developed)
- Project READMEs include **Presentation notes** for non-technical audience framing
- Notebooks are for storytelling and exploration; move reusable logic into `scripts/utils/`
- When both Python and R implementations exist, note the tradeoffs in the technique README
- Data files stay in `data/` and are never committed to git

## Non-technical presentation design

When creating outputs for non-technical audiences:
- Lead with the business question and answer, not the methodology
- Use plotly or seaborn for clean, labeled visuals -- no raw matplotlib defaults
- Include a "So what?" interpretation after every chart or table
- Keep technical jargon in footnotes or appendices, not the main narrative

## Lessons-learned workflow

After completing any project or technique deep-dive, update `docs/lessons-learned/` with:
- What worked and what didn't
- Debugging insights or data gotchas encountered
- Techniques or patterns worth reusing
