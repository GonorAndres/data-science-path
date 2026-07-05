# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Personal data science learning repository and portfolio. Owner is an actuarial science bachelor doing a data science internship. This repo serves as:
- **Technique reference**: Theory + working code for each method (techniques/)
- **Portfolio projects**: End-to-end analyses with business framing (projects/)
- **Sandbox**: Quick experiments and explorations (notebooks/exploratory/)
- **Reusable scripts**: Utility functions that get reused across projects (scripts/utils/)

## Repository layout

```
techniques/          -- One folder per technique (~65 folders after 2026-04-26 expansion)
                        Each has: README.md (theory + code), YAML frontmatter required
projects/            -- Self-contained portfolio projects (18 total: 9 complete, 3 in progress, 6 scaffolds)
                        Each has: README.md (business context + retrospective), code, outputs
scripts/utils/       -- Shared Python utilities (plotting helpers, data loaders, evaluation)
notebooks/exploratory/ -- Throwaway exploration notebooks
data/                -- raw/, processed/, external/ (not tracked in git)
docs/theory/         -- 22 theory essays organized in 6 canonical parts (see Theory Library below)
docs/references/     -- Academic PDFs + broader curriculum reference list
docs/lessons-learned/ -- Post-project retrospectives
docs/presentations/  -- Portfolio one-pagers (EN + ES)
subagents_outputs/   -- Subagent research and audit files (never delete; feed future agents)
next_steps/          -- Roadmap docs for 6 future technique areas
career/              -- CV templates, LinkedIn copy, recruiter outreach, target companies MX
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

1. **ML & Predictive Modeling**: classification, regression, ensembles, model evaluation
2. **Customer Analytics**: segmentation, churn, CLV, cohort analysis
3. **Actuarial / Risk Modeling**: survival analysis, GLMs, reserving, pricing
4. **Modern ML**: LLMs, RAG, computer vision, time series, recommender systems
5. **Production**: MLOps, cloud deployment (GCP Cloud Run), SQL analytics

## Theory Library (docs/theory/)

22 essays organized into 6 canonical parts. The canonical reading order is defined in `docs/theory/00_index.md`. Do not reorder or rename files without updating that index.

**Part I: Mathematical Foundations**
`00_probability_estimation_primer.md`, `00b_optimization_and_loss_primer.md`, `00_glossary.md`

**Part II: Generalization, Validation, and Feature Engineering**
`01_ml_fundamentals_bias_variance.md`, `02_regularization_and_generalization.md`,
`03_cross_validation_strategies.md`, `04_data_leakage_patterns.md`, `05_feature_engineering_cookbook.md`

**Part III: Inference, Causality, and Uncertainty**
`08a_experimentation.md`, `08b_observational_causal.md`, `18_calibration_and_conformal.md`

**Part IV: Interpretability, Fairness, and Governance**
`06_model_interpretability_shap_lime.md`, `07_fairness_and_responsible_ai.md`

**Part V: Modern Technique Families**
`17_dl_fundamentals.md`, `09_llm_foundations_2026.md`, `10_computer_vision_foundations.md`,
`11_time_series_foundations.md`, `12_recommender_system_foundations.md`

**Part VI: Production and Infrastructure**
`13_mlops_architecture_patterns.md`, `14_cloud_deployment_primer.md`, `15_sql_for_data_scientists.md`

**Do NOT touch (audited as best-in-class):**
`04_data_leakage_patterns.md` and techniques: `glm`, `regression`, `survival-analysis`,
`ratemaking-basics`, `bayesian-credibility`, `tweedie-compound-models`, `cross-validation`,
`model-calibration`, `shap-interpretability`.

## PDF output

`docs/ds-theory_intuitive.pdf` is the compiled intuitive-flavor PDF of the full theory library.
- 22 pages, 174 KB, generated 2026-04-26 via `/doc intuitive docs/theory`
- Mirrored to Google Drive at `vm-claude-dev/projects_1.0/data-engineering/data-science-path/docs/ds-theory_intuitive.pdf`
- To regenerate after edits: run `/doc intuitive docs/theory` and then `/gdrive upload docs/ds-theory_intuitive.pdf`

## Syllabus audit workflow

When auditing theory or technique content, spawn an **opus** subagent with the role of "senior DS curriculum designer." The audit output goes to `subagents_outputs/syllabus_audit_<date>.md` and feeds the rewrite agents. Key files from the 2026-04-26 audit:
- `subagents_outputs/syllabus_audit_2026_04_25.md` — 40-item improvement manifest (HIGH/MED/LOW)
- `subagents_outputs/project_state_2026_04_25.md` — project completion status table
- `subagents_outputs/doc_gather/ds-theory_raw.md` — gather file for PDF generation

## Subagent conventions

- Subagent outputs always go to `subagents_outputs/` as `.md` files
- Use **opus** for curriculum audits, rewrites, and content creation (quality matters)
- Use **haiku** for readability checks and lightweight validation (token efficiency)
- For PDF readability checks: use haiku with vision (Read tool on the PDF), not bash/pdftotext
- Split large page-range checks across 2 parallel haiku agents (e.g., pages 1-11 and 12-22)

## Secrets & Environment Variables (MANDATORY)

**NEVER commit secrets, API keys, passwords, tokens, or credentials to this repository.**

Before every commit, verify that no private variables are being staged. If a hardcoded secret is found anywhere in the codebase:
1. **Remove** the hardcoded value immediately
2. **Replace** it with an environment variable reference (e.g., `os.environ["DB_PASSWORD"]`)
3. **Add** the variable name and a placeholder to `.env.example` so collaborators know what to set
4. **Confirm** `.env` is listed in `.gitignore` (it already is: `.env`, `*.env`, `.env.*`, `credentials.json`, `*.pem`, `*.key` are all gitignored)

Common patterns:
```python
# WRONG: never do this
API_KEY = "sk-abc123..."
conn = psycopg2.connect(password="my_password")

# RIGHT: always do this
import os
API_KEY = os.environ["API_KEY"]
conn = psycopg2.connect(password=os.environ["DB_PASSWORD"])
```

If you need a new secret variable, add it to `.env.example` with an empty or placeholder value before using it in code.

## Writing standards

- **No double-dash em-dashes**: Never use `--` as punctuation in any user-facing text (READMEs, dashboard strings, notebook markdown, blog posts). Use commas, semicolons, colons, or restructure the sentence instead. This applies to `.md`, `.tsx`, `.ts`, and `.ipynb` files.

## Production & Deployment Standards

- Every project README must include a **Deployment** section with production evidence (live URL, GCP console screenshot, cost estimate)
- Every project README must include a **Decisions & Trade-offs** table and a **Retrospective** section
- CI must pass (lint + test) before deployment. Green CI badge required on root README.
- Technique READMEs must have YAML frontmatter: `tags`, `status` (draft/review/complete), `created`, `updated`

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
- Use plotly or seaborn for clean, labeled visuals; no raw matplotlib defaults
- Include a "So what?" interpretation after every chart or table
- Keep technical jargon in footnotes or appendices, not the main narrative

## Lessons-learned workflow

After completing any project or technique deep-dive, update `docs/lessons-learned/` with:
- What worked and what didn't
- Debugging insights or data gotchas encountered
- Techniques or patterns worth reusing

## Technique folder standards (updated 2026-04-26)

All technique READMEs must have YAML frontmatter and reach at least 150 lines to be considered non-stub.
Folders under 80 lines are stubs. As of 2026-04-26 the following new folders were created:
`conformal-prediction`, `gaussian-processes`, `multi-armed-bandits`, `reinforcement-learning`,
`state-space-and-kalman`, `garch-volatility`.

The two causal-inference folders are intentionally split:
- `techniques/causal-inference/README.md` — canonical, full treatment including DML and causal forests
- `techniques/causal-inference-intro/README.md` — redirect note pointing to the canonical file

## Project status snapshot (2026-04-26)

| Status | Count | Notable |
|---|---|---|
| Complete | 9 | insurance-pricing, causal-uplift, cv-defect-detection, llm-rag-qa, mlops-model-service, nlp-reviews-multilingual, recsys-movielens, sql-analytics-ecommerce, timeseries-retail-forecast |
| In progress | 3 | genai-agent-tool-use (eval tasks missing), streamlit-exec-dashboard (no live URL), kaggle-competition-log (awaiting competition) |
| Scaffold only | 6 | claims-reserving-dl, claims-text-classifier, claims-triage-system, customer-analytics, risk-modeling, vehicle-damage-cv |
