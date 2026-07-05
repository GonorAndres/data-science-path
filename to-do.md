# To-Do

Last updated: 2026-03-15

## Status Snapshot

| Project | Code | Notebooks | Backend | Dashboard | Tests | Deployed |
|---------|------|-----------|---------|-----------|-------|----------|
| insurance-pricing | done | 4 notebooks | FastAPI | Next.js | yes (2 files) | not started |
| claims-reserving-dl | scaffold only | -- | -- | -- | -- | -- |
| claims-text-classifier | scaffold only | -- | -- | -- | -- | -- |
| claims-triage-system | scaffold only | -- | -- | -- | -- | -- |
| customer-analytics | scaffold only | -- | -- | -- | -- | -- |
| risk-modeling | scaffold only | -- | -- | -- | -- | -- |
| vehicle-damage-cv | scaffold only | -- | -- | -- | -- | -- |

Techniques: 44 READMEs written (no frontmatter), 5 empty scaffolds.
Docs: 4 academic PDFs in `references/`; `lessons-learned/`, `theory/`, `presentations/` all empty.
CI/CD: no workflows (`.github/workflows/` is empty).
Root README.md: does not exist.
MLflow: tracking DB exists (`mlflow.db`, 1.67 MB).

---

## 1. Projects -- Remaining Build

### Unbuilt projects (6 scaffolds with only .gitkeep files)
- [ ] [L] `projects/claims-reserving-dl/` -- deep learning for loss reserving
- [ ] [L] `projects/claims-text-classifier/` -- NLP classification of claim descriptions
- [ ] [L] `projects/claims-triage-system/` -- automated claim routing/prioritization
- [ ] [L] `projects/customer-analytics/` -- segmentation, churn, CLV modeling
- [ ] [L] `projects/risk-modeling/` -- GLM/GBM risk scoring
- [ ] [L] `projects/vehicle-damage-cv/` -- computer vision for damage assessment

### Insurance-pricing polish
- [ ] [M] Add screenshots/output evidence to project README
- [ ] [S] Add "Decisions & Trade-offs" section to project README
- [ ] [S] Add "Retrospective" section to project README (per CLAUDE.md convention)
- [ ] [S] Add nbviewer badges to README for the 4 notebooks
- [ ] [M] Add integration tests (currently only unit tests for utils, not for model training pipeline)

## 2. Deployment & Infrastructure

- [ ] [L] Deploy insurance-pricing backend to Cloud Run (Dockerfile already exists)
- [ ] [M] Create GitHub Actions deploy workflow for insurance-pricing
- [ ] [S] Upload processed data to GCS bucket (3 Parquet files currently local)
- [ ] [M] Set up Workload Identity Federation for CI/CD (follow data-analyst-path pattern)
- [ ] [S] Document deployment steps and Cloud Run service URLs

## 3. CI/CD & Quality

- [ ] [M] Create lint + test CI workflow (`.github/workflows/ci.yml`) -- ruff lint, pytest
- [ ] [S] Add ruff lint step for `projects/insurance-pricing/src/`, `scripts/utils/`, `tests/`
- [ ] [S] Add pre-commit hooks (ruff, trailing whitespace, YAML lint)
- [ ] [S] Add CI status badge to README once workflow exists
- [ ] [M] Expand test coverage -- add tests for `projects/insurance-pricing/src/` modules (train.py, glm_models.py, gbm_models.py, shap_analysis.py, fairness.py, feature_engineering.py)

## 4. Techniques Library

### Content quality
- [ ] [M] Add YAML frontmatter (`tags`, `status`, `created`, `updated`) to all 44 technique READMEs
- [ ] [M] Review and promote strongest technique READMEs to `status: complete` (start with: glm, regression, classification, clustering, survival-analysis, shap-interpretability)
- [ ] [M] Add working code examples (`.py` or `.ipynb`) to technique folders that only have READMEs
- [ ] [S] Verify all 44 READMEs follow `techniques/TEMPLATE.md` structure

### Empty scaffolds (5 techniques)
- [ ] [M] `convolutional-neural-networks/` -- write README + code example
- [ ] [M] `entity-embeddings/` -- write README + code example
- [ ] [M] `temporal-fusion-transformer/` -- write README + code example
- [ ] [M] `transfer-learning/` -- write README + code example
- [ ] [M] `transformers-nlp/` -- write README + code example

## 5. Documentation

### Root level
- [x] [M] Create `README.md` at repo root (2026-03-15)

### Populate empty doc directories
- [ ] [M] Write at least one entry in `docs/theory/` (e.g., bias-variance tradeoff, model selection framework)
- [ ] [M] Write at least one entry in `docs/lessons-learned/` (retrospective from insurance-pricing project)
- [ ] [M] Create at least one presentation in `docs/presentations/` (e.g., insurance-pricing non-technical summary)

### References
- [ ] [S] Verify `docs/references/README.md` links to all 4 PDFs with brief descriptions

### Next steps roadmaps
- [ ] [S] Review 6 roadmaps in `next_steps/` and prioritize which to tackle first (causal-ml, generative-ai-llms, graph-neural-networks, mlops-platform-engineering, real-time-ml-systems, reinforcement-learning)

## 6. Notebooks & Exploration

- [ ] [S] Add at least one exploratory notebook in `notebooks/exploratory/` (currently empty)
- [ ] [S] Add index or README in `notebooks/` explaining what lives where
- [ ] [S] Ensure project notebooks (insurance-pricing) render cleanly on GitHub

## 7. Scripts & Utilities

- [ ] [S] Add docstrings to `scripts/utils/data_loader.py` and `scripts/utils/evaluation.py` if missing
- [ ] [M] Add more shared utilities (e.g., plotting helpers, model persistence, feature importance extraction)
- [ ] [S] Add a `scripts/setup.sh` or bootstrap script for environment setup

## 8. MLflow & Experiment Tracking

- [ ] [S] Document MLflow setup and usage in README or a dedicated doc
- [ ] [S] Add `mlflow.db` to `.gitignore` (binary DB file, 1.67 MB, should not be tracked)
- [ ] [M] Set up MLflow UI instructions (local `mlflow ui` command, what experiments exist)

## 9. Cleanup

- [ ] [S] Remove or `.gitignore` `temp/` directory (contains a stray screenshot)
- [ ] [S] Remove `__pycache__/` directories from git tracking if committed
- [ ] [S] Verify `.env.example` has all required variables documented

---

## Pure Data Science Track (added 2026-04-23)

Parallel track on top of the actuarial / insurance work, built around the 2026 Mexican DS job market. Each project pairs a hands-on implementation with a `theory/` companion folder of 3-4 documents. Repo-wide `docs/theory/` holds 15 cross-cutting essays.

### Scaffolded (README + theory + src skeleton ready)
- [x] `projects/nlp-reviews-multilingual/` -- multilingual sentiment (ES+EN)
- [x] `projects/mlops-model-service/` -- FastAPI + Docker + Cloud Run + CI/CD
- [x] `projects/timeseries-retail-forecast/` -- hierarchical M5 forecasting
- [x] `projects/recsys-movielens/` -- two-tower + ALS recommender
- [x] `projects/sql-analytics-ecommerce/` -- DuckDB warehouse + dbt-lite
- [x] `projects/streamlit-exec-dashboard/` -- bilingual executive BI
- [x] `projects/llm-rag-qa/` -- local RAG (Ollama + Chroma)
- [x] `projects/genai-agent-tool-use/` -- local LLM agent with SQL + web tools
- [x] `projects/cv-defect-detection/` -- transfer learning on MVTec AD
- [x] `projects/causal-uplift-marketing/` -- uplift modeling on Criteo
- [x] `projects/kaggle-competition-log/` -- live competition writeup (template)

### Theory library (docs/theory/) -- all 15 essays
- [x] 01 ML fundamentals and bias-variance
- [x] 02 Regularization and generalization
- [x] 03 Cross-validation strategies
- [x] 04 Data leakage patterns
- [x] 05 Feature engineering cookbook
- [x] 06 Model interpretability (SHAP, LIME)
- [x] 07 Fairness and responsible AI
- [x] 08 Experimentation and causal primer
- [x] 09 LLM foundations 2026
- [x] 10 Computer vision foundations
- [x] 11 Time series foundations
- [x] 12 Recommender system foundations
- [x] 13 MLOps architecture patterns
- [x] 14 Cloud deployment primer
- [x] 15 SQL for data scientists

### Technique library additions / backfills
- [x] `techniques/deep-learning-pytorch/`
- [x] `techniques/llm-prompting-and-rag/`
- [x] `techniques/recommender-systems/`
- [x] `techniques/causal-inference/`
- [x] `techniques/mlops-fundamentals/`
- [x] `techniques/feature-stores/`
- [x] `techniques/model-monitoring-drift/`
- [x] `techniques/vector-databases/`
- [x] `techniques/graph-neural-networks/`
- [x] `techniques/hyperparameter-optimization-optuna/`
- [x] `techniques/experiment-tracking-mlflow/`
- [x] `techniques/data-contracts-and-validation/`
- [x] `techniques/convolutional-neural-networks/` (populated)
- [x] `techniques/transformers-nlp/` (populated)
- [x] `techniques/transfer-learning/` (populated)
- [x] `techniques/entity-embeddings/` (populated)
- [x] `techniques/temporal-fusion-transformer/` (populated)

### Career toolkit (career/)
- [x] CV templates (ES, EN)
- [x] LinkedIn headline and About (bilingual)
- [x] Mexican job boards guide (OCC, Bumeran, Indeed MX, LinkedIn MX)
- [x] Target companies shortlist
- [x] Recruiter outreach templates (ES, EN)
- [x] SQL drills, pandas drills, ML flashcards, DL flashcards
- [x] Case studies, ML system design, behavioral (ES+EN)

### Pending implementation for pure DS track
- [ ] [L] Download datasets and actually train each project's models
- [ ] [L] Deploy mlops-model-service to Cloud Run; put live URL in root README
- [ ] [M] Convert notebook outlines (.md) into real .ipynb files once datasets are fetched
- [ ] [M] Add real retrospective content to each project after first training run
- [ ] [M] Author the LinkedIn post announcing the track once 2-3 projects are live

---

## Done

_Completed items move here with date._
