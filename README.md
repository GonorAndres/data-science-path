# Data Science Portfolio & Knowledge Base

**Andres Gonzalez Ortega** | Actuarial Science, UNAM (Mexico) | Based in CDMX

A data science portfolio and knowledge base. It has two parallel tracks:

- **Pure Data Science Track** (below): general-purpose ML, NLP, CV, time series, recommenders, causal ML, MLOps on public non-insurance datasets. Built for data-science and ML-engineer recruiters.
- **Actuarial / Insurance Track** (further down): domain-specific work on insurance pricing, claims, reserving, and risk. Built around UNAM actuarial training.

Every project ships with a `theory/` companion folder (3-4 documents per project). The repo-wide [`docs/theory/`](docs/theory/00_index.md) library holds 15 cross-cutting essays covering the theoretical backbone behind every project. A bilingual [career toolkit](career/README.md) targets the 2026 Mexican data science job market.

---

## Pure Data Science Track

Eleven hands-on projects on public datasets, each with its own theory layer.

| Project | What it demonstrates | Stack | Status |
|---------|----------------------|-------|--------|
| [nlp-reviews-multilingual](projects/nlp-reviews-multilingual/) | Multilingual sentiment + aspect classification (ES+EN) | transformers, sklearn, SHAP | Scaffolded |
| [llm-rag-qa](projects/llm-rag-qa/) | Retrieval-augmented QA on a public corpus | sentence-transformers, Chroma, Ollama (Llama 3.1) | Scaffolded |
| [cv-defect-detection](projects/cv-defect-detection/) | Transfer learning for visual defect detection | PyTorch, timm, Grad-CAM | Scaffolded |
| [timeseries-retail-forecast](projects/timeseries-retail-forecast/) | Hierarchical demand forecasting | statsmodels, Prophet, LightGBM | Scaffolded |
| [recsys-movielens](projects/recsys-movielens/) | Two-tower recommender on MovieLens 25M | implicit, LightFM, PyTorch | Scaffolded |
| [causal-uplift-marketing](projects/causal-uplift-marketing/) | Uplift modeling and A/B readout | causalml, econml | Scaffolded |
| [mlops-model-service](projects/mlops-model-service/) | FastAPI + Docker + Cloud Run + CI/CD | FastAPI, Docker, GCP, GitHub Actions | Scaffolded |
| [sql-analytics-ecommerce](projects/sql-analytics-ecommerce/) | Warehouse-grade SQL case study | DuckDB, BigQuery, dbt | Scaffolded |
| [genai-agent-tool-use](projects/genai-agent-tool-use/) | Local LLM agent with tool use | Ollama, LangGraph | Scaffolded |
| [streamlit-exec-dashboard](projects/streamlit-exec-dashboard/) | Executive BI dashboard | Streamlit, plotly | Scaffolded |
| [kaggle-competition-log](projects/kaggle-competition-log/) | Live competition log and reflection | Kaggle, notebooks | Planned |

See the full index at [`projects/00_pure_ds_index.md`](projects/00_pure_ds_index.md).

---

## Actuarial / Insurance Track

Insurance-domain projects built on UNAM actuarial coursework.

## Projects

| Project | What It Demonstrates | Stack | Status |
|---------|---------------------|-------|--------|
| [Insurance Pricing](projects/insurance-pricing/) | GLM/GBM pricing, SHAP interpretability, fairness analysis | Python, scikit-learn, XGBoost, SHAP | Complete |
| [Claims Reserving DL](projects/claims-reserving-dl/) | Deep learning for loss reserving | Python, PyTorch | Planned |
| [Claims Text Classifier](projects/claims-text-classifier/) | NLP classification of claim descriptions | Python, NLP | Planned |
| [Claims Triage System](projects/claims-triage-system/) | Automated claim routing/prioritization | Python, ML | Planned |
| [Customer Analytics](projects/customer-analytics/) | Segmentation, churn, CLV modeling | Python, scikit-learn | Planned |
| [Risk Modeling](projects/risk-modeling/) | GLM/GBM risk scoring | Python, statsmodels | Planned |
| [Vehicle Damage CV](projects/vehicle-damage-cv/) | Computer vision for damage assessment | Python, PyTorch | Planned |

## Techniques Library

44 technique reference guides covering:
- **ML & Predictive Modeling**: classification, regression, ensembles, model evaluation, cross-validation
- **Customer Analytics**: segmentation, churn, CLV, cohort analysis, RFM
- **Actuarial / Risk Modeling**: survival analysis, GLMs, credibility theory, experience studies, loss reserving
- **Deep Learning**: neural networks, CNNs, transformers, transfer learning
- **Interpretability**: SHAP, feature importance, partial dependence

Each technique folder contains a README with theory, when-to-use guidance, and trade-offs. See [techniques/TEMPLATE.md](techniques/TEMPLATE.md) for the standard structure.

## Tech Stack

- **Python**: pandas, scikit-learn, XGBoost, LightGBM, statsmodels, lifelines, plotly, seaborn
- **R**: tidyverse, ggplot2, broom (secondary)
- **Experiment Tracking**: MLflow
- **Notebooks**: Jupyter for exploration and presentation

## Quick Start

```bash
git clone <repo-url>
cd data-science
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run insurance-pricing project
cd projects/insurance-pricing
python src/main.py
```

## Repository Structure

```
data-science/
├── techniques/          # Technique reference guides (theory + code)
├── projects/            # Portfolio projects (each self-contained)
├── scripts/utils/       # Shared Python utilities
├── career/              # Bilingual CV, LinkedIn, interview prep (MX market)
├── notebooks/           # Exploratory notebooks
├── docs/
│   ├── theory/          # 15-essay cross-cutting theory library
│   ├── lessons-learned/ # Retrospectives per project
│   └── presentations/   # Non-technical briefs, bilingual one-pagers
├── next_steps/          # Roadmaps for advanced topics
├── tests/               # Test files
└── data/                # raw/, processed/, external/ (gitignored)
```

## Career Toolkit (Mexican DS market)

- [CV templates](career/cv_template_es.md) (Spanish and English, ATS-optimized)
- [LinkedIn copy](career/linkedin_headline_and_about.md) (headline and About variants)
- [Mexican job boards guide](career/mexican_ds_job_boards.md) (OCC, Bumeran, Indeed MX, LinkedIn MX)
- [Target companies shortlist](career/target_companies_mexico.md)
- [Interview prep](career/interview_prep/) (SQL, pandas, ML theory, ML system design, behavioral)

## Related Repositories

- [Data Analyst Portfolio](https://github.com/GonorAndres/data-analyst-path) -- Business analytics, dashboards, storytelling
- [Data Engineering Portfolio](https://github.com/GonorAndres/data-engineer-path) -- Pipelines, infrastructure, orchestration
- [Portfolio Website](https://github.com/GonorAndres/GonorAndres.github.io) -- Astro-based professional portfolio
