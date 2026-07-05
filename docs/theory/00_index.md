# Theory Library Index

A canonical, beginner-to-advanced reading sequence for the data science theory library. Each essay is standalone (1,500 to 3,000 words), cites primary sources, and assumes only the prerequisites listed in earlier essays. Project-level theory writeups link back here rather than duplicate content.

## Purpose

- Give a fresh reader a single ordered path from first principles to production.
- Give an interviewer a way to browse the conceptual backbone in one place.
- Give the author a consolidated reference for interview prep and on-the-job problem framing.
- Surface the connections between methods so the portfolio reads as one body of work.

## Canonical reading order

The numbering reflects the intended sequence. Earlier essays are prerequisites for later ones.

### Block A: foundations (probability, optimization, SQL, generalization)

| # | Essay | Status |
|---|-------|--------|
| 00 | [Probability and statistical estimation primer](00_probability_estimation_primer.md) | draft |
| 00b | [Optimization and loss-function primer](00b_optimization_and_loss_primer.md) | draft |
| 15 | [SQL for data scientists: beyond SELECT](15_sql_for_data_scientists.md) | review |
| 01 | [ML fundamentals: bias, variance, and generalization](01_ml_fundamentals_bias_variance.md) | review |
| 02 | [Regularization and the art of generalization](02_regularization_and_generalization.md) | review |
| 03 | [Cross-validation strategies](03_cross_validation_strategies.md) | review |
| 04 | [Data leakage patterns](04_data_leakage_patterns.md) | complete |
| 05 | [Feature engineering cookbook](05_feature_engineering_cookbook.md) | review |

### Block B: inference, decisions, uncertainty

| # | Essay | Status |
|---|-------|--------|
| 08a | [Experimentation and A/B testing](08a_experimentation.md) | draft |
| 08b | [Observational causal inference](08b_observational_causal.md) | draft |
| 18 | [Calibration and conformal prediction](18_calibration_and_conformal.md) | draft |

### Block C: interpretability and ethics

| # | Essay | Status |
|---|-------|--------|
| 06 | [Model interpretability with SHAP, LIME, and friends](06_model_interpretability_shap_lime.md) | review |
| 07 | [Fairness and responsible AI](07_fairness_and_responsible_ai.md) | review |

### Block D: modern technique families

| # | Essay | Status |
|---|-------|--------|
| 17 | [Deep learning fundamentals](17_dl_fundamentals.md) | draft |
| 10 | [Computer vision foundations](10_computer_vision_foundations.md) | review |
| 09 | [LLM foundations for data scientists (2026)](09_llm_foundations_2026.md) | review |
| 11 | [Time series foundations](11_time_series_foundations.md) | review |
| 12 | [Recommender system foundations](12_recommender_system_foundations.md) | review |

### Block E: production

| # | Essay | Status |
|---|-------|--------|
| 13 | [MLOps architecture patterns](13_mlops_architecture_patterns.md) | review |
| 14 | [Cloud deployment primer for ML services](14_cloud_deployment_primer.md) | review |

## Glossary anchor

A consolidated acronym list. Each item is defined in the essay listed in parentheses; click through for full treatment.

- **ALE** Accumulated Local Effects (06).
- **ATE / ATT / CATE** Average / Treatment-on-treated / Conditional treatment effect (08b).
- **BH** Benjamini-Hochberg false discovery rate (00).
- **BPR** Bayesian Personalized Ranking, pairwise loss (12, 00b).
- **CRPS** Continuous Ranked Probability Score (11).
- **CUPED** Controlled-experiment Using Pre-Experiment Data (08a).
- **DiD** Difference-in-differences (08b).
- **DML** Double Machine Learning (08b).
- **ECE** Expected Calibration Error (18).
- **ERM** Empirical Risk Minimization (01).
- **FDR** False Discovery Rate (00).
- **GARCH** Generalized AutoRegressive Conditional Heteroskedasticity (11).
- **GLM** Generalized Linear Model (techniques/glm).
- **IPW / IPSW** Inverse Propensity Weighting (08b).
- **IV** Information Value (05) and Instrumental Variable (08b); context-dependent.
- **KV-cache** Key-Value cache for transformer inference (09).
- **LoRA** Low-Rank Adaptation (17, 09).
- **MAP** Maximum A Posteriori (00, 02).
- **MDE** Minimum Detectable Effect (08a).
- **MoE** Mixture of Experts (09).
- **MLE** Maximum Likelihood Estimation (00).
- **NDCG** Normalized Discounted Cumulative Gain (12).
- **PDP / ICE** Partial Dependence Plot / Individual Conditional Expectation (06).
- **PSI** Population Stability Index (13).
- **RAG** Retrieval-Augmented Generation (09).
- **RDD** Regression Discontinuity Design (08b).
- **RoPE** Rotary Position Embedding (09).
- **SHAP** SHapley Additive exPlanations (06).
- **SRM** Sample Ratio Mismatch (08a).
- **SUTVA** Stable Unit Treatment Value Assumption (08a).
- **WoE** Weight of Evidence (05).

## Reading-order suggestions for shorter goals

- **Interview prep:** 15, 00, 01, 02, 03, 04, 06, 08a, 13.
- **Modern ML catch-up:** 17, 09, 10, 12, 18.
- **Responsible AI:** 06, 07, 08b, 18.
- **Production engineering:** 13, 14, 15.
- **Actuarial bridge:** 00, 02 (Bayesian-prior interpretation), 05 (WoE), 11 (GARCH), 18 (conformal pricing intervals).

## How projects reference these essays

Each project's `theory/` folder contains a short companion essay applying the concept to that specific business problem, then links back here. Example:

```markdown
For the general framing of attention, see
[docs/theory/09_llm_foundations_2026.md](../../../docs/theory/09_llm_foundations_2026.md).
In this project we apply it to multilingual product review classification.
```

This keeps project writeups focused on the business problem while the theory library remains the single source of truth.

## Writing conventions

- Math in LaTeX, rendered through GitHub markdown (`$...$` inline, `$$...$$` display).
- Every essay ends with a **References** section: 3 to 8 canonical papers, books, or documentation pages.
- No em-dashes in prose; no double-dash (`--`) as punctuation. Use commas, semicolons, colons, or restructure.
- No emojis.
- Bilingual titles only on project READMEs (Resumen / Summary block); theory essays remain English-only.

## Changelog

- 2026-04-25: Restructure index from flat catalog to canonical reading order; add status column, glossary anchor, and changelog. Introduce essays 00, 00b, 17, 18, and split 08 into 08a (experimentation) and 08b (observational causal).
- 2026-04-23: Initial 15-essay flat catalog.
