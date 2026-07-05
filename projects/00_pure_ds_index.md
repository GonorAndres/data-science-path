# Pure Data Science Track: Project Index

## Resumen (ES)

Once proyectos de ciencia de datos puros sobre datasets públicos. Cada proyecto incluye una carpeta `theory/` con 3 a 4 documentos técnicos que explican el problema, el método, la implementación y la evaluación. El stack está alineado con lo que piden los recruiters de ciencia de datos en México (OCC, Bumeran, Indeed MX, LinkedIn MX) en 2026: Python, SQL, scikit-learn, XGBoost, PyTorch, transformers, MLflow, Docker, GCP.

## Summary (EN)

Eleven pure data science projects on public datasets. Each project ships with a `theory/` folder of 3 to 4 companion documents covering problem framing, method deep-dive, implementation notes, and evaluation. Stack aligned with what Mexican data science recruiters screen for on OCC Mundial, Bumeran, Indeed MX, and LinkedIn MX in 2026: Python, SQL, scikit-learn, XGBoost, PyTorch, HuggingFace transformers, MLflow, Docker, GCP.

---

## Matrix

| # | Project | Domain | Core technique | Dataset | Deploy | Status |
|---|---------|--------|----------------|---------|--------|--------|
| 1 | [nlp-reviews-multilingual](nlp-reviews-multilingual/) | NLP | Fine-tuned multilingual transformer | Amazon Reviews Multi | Streamlit Cloud | Scaffolded |
| 2 | [llm-rag-qa](llm-rag-qa/) | LLM / RAG | Retrieval + local Llama 3.1 | Public corpus (Wiki ES slice) | Gradio local | Scaffolded |
| 3 | [cv-defect-detection](cv-defect-detection/) | Computer Vision | Transfer learning CNN | MVTec AD | Gradio local | Scaffolded |
| 4 | [timeseries-retail-forecast](timeseries-retail-forecast/) | Forecasting | Hierarchical ensemble | M5 (Walmart) | Streamlit | Scaffolded |
| 5 | [recsys-movielens](recsys-movielens/) | Recommenders | Two-tower + matrix factorization | MovieLens 25M | Streamlit | Scaffolded |
| 6 | [causal-uplift-marketing](causal-uplift-marketing/) | Causal ML | Uplift trees + meta-learners | Criteo Uplift | Notebook | Scaffolded |
| 7 | [mlops-model-service](mlops-model-service/) | MLOps | FastAPI + Docker + CI/CD | Wraps project #1 | Cloud Run | Scaffolded |
| 8 | [sql-analytics-ecommerce](sql-analytics-ecommerce/) | SQL / Analytics | Warehouse SQL, dbt | Olist Brazilian E-commerce | Streamlit | Scaffolded |
| 9 | [genai-agent-tool-use](genai-agent-tool-use/) | GenAI / Agents | Local LLM with tool loop | Works on project #8 data | Local | Scaffolded |
| 10 | [streamlit-exec-dashboard](streamlit-exec-dashboard/) | BI / Viz | Exec-level storytelling | Olist or M5 | Streamlit Cloud | Scaffolded |
| 11 | [kaggle-competition-log](kaggle-competition-log/) | Competition | Live Kaggle submission | Active competition | Kaggle | Planned |

---

## Skill coverage (recruiter keyword alignment)

| Skill | Projects where it shows |
|-------|------------------------|
| Python, pandas, NumPy | All |
| SQL (window, CTEs, cohorts) | 8, 9 |
| scikit-learn, XGBoost, LightGBM | 1, 4, 5, 6 |
| PyTorch | 1, 3, 5, 9 |
| HuggingFace transformers | 1, 2 |
| NLP (tokenization, embeddings) | 1, 2, 9 |
| Computer Vision (CNN, transfer) | 3 |
| Time series forecasting | 4 |
| Recommender systems | 5 |
| Causal inference, A/B testing | 6 |
| MLflow, experiment tracking | 1, 4, 5 |
| FastAPI, Docker | 7 |
| Cloud Run / GCP | 7 |
| GitHub Actions CI/CD | 7 |
| LLM prompting, RAG, agents | 2, 9 |
| Streamlit / dashboards | 1, 4, 8, 10 |
| Responsible AI, fairness, drift | 1, 6, 7 |
| Bilingual documentation | All |

---

## How each project is organized

```
<project>/
├── README.md                 business framing, bilingual Resumen/Summary, results, demo link
├── theory/
│   ├── 00_problem_framing.md
│   ├── 01_<topic>.md         method deep dive (math and mechanics)
│   ├── 02_<topic>.md         implementation notes (libraries, tradeoffs)
│   └── 03_<topic>.md         evaluation and interpretation
├── data_download.py          reproducible dataset fetch (no data committed)
├── notebooks/                storytelling notebooks
├── src/                      reusable code modules
├── tests/                    pytest smoke tests
├── requirements.txt          pinned dependencies
└── Makefile or justfile      setup, train, evaluate, serve targets
```

## How to navigate the theory layer

Each project's `theory/` folder answers these four questions in order:

1. **What is the business problem and why this method?** (`00_problem_framing.md`)
2. **How does the method work mathematically?** (`01_*.md`)
3. **What are the practical tradeoffs in this implementation?** (`02_*.md`)
4. **How do we measure success and read the results?** (`03_*.md`)

Cross-cutting fundamentals live in [`docs/theory/`](../docs/theory/00_index.md). Project-specific theory links out to those essays whenever a broader concept is referenced.

## Related

- Actuarial / insurance track: see root [README.md](../README.md).
- Technique library: [`techniques/`](../techniques/).
- Career toolkit for Mexican DS market: [`career/README.md`](../career/README.md).
