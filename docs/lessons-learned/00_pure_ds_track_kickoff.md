# Pure Data Science Track: Kickoff Retrospective

**Date started:** 2026-04-23
**Author:** Andres Gonzalez Ortega
**Status:** Active

## Why this track exists

The repository originally housed a single-vertical actuarial / insurance portfolio. One complete project (`insurance-pricing`), six insurance-domain scaffolds, 44 technique READMEs with no working code, and empty docs folders. Strong as a domain specialist signal, weak as a generalist data science signal.

The 2026 Mexican data science job market, as observed across OCC Mundial, Bumeran, Indeed MX, and LinkedIn MX, screens for:

- Python, SQL, scikit-learn, XGBoost, PyTorch, HuggingFace transformers, MLflow, Docker, cloud ML (usually GCP or Azure).
- Deployed artifacts over notebooks (live URL, CI badge, Docker image).
- At least one NLP, one CV, one time-series, and one MLOps project on the portfolio GitHub.
- Bilingual documentation (Spanish and English).
- Evidence of experimentation (A/B testing, causal reasoning) for product roles.

The pure data science track fills this gap without touching the actuarial work. Both tracks coexist on the repository landing page.

## Success metrics

| Metric | Target | How we measure it |
|--------|--------|-------------------|
| Projects shipped with theory, code, and README | 10 of 11 | Visual audit of `projects/` tree |
| Live demo links per project (where applicable) | 8 of 11 | Click-through on README |
| Theory essays in `docs/theory/` | 15 | File count |
| Career toolkit bilingual coverage | 100 percent | Grep for `(ES)` and `(EN)` headers |
| Recruiter replies on cold DMs | 10 percent+ | Manual log |
| LinkedIn profile views uplift | 3x baseline | LinkedIn analytics (monthly) |
| Interview invites per month during active search | 3+ | Manual log |

## Guardrails

- No edits to actuarial projects or techniques (explicit list in the plan file).
- No em-dashes in prose (repository-wide rule, CLAUDE.md).
- No hardcoded secrets (CLAUDE.md); new env vars go to `.env.example`.
- Every new project uses public non-insurance data.
- Every new project ships a `theory/` folder before being considered complete.

## What worked (running log, updated at each wave close)

_Wave 0 kickoff:_
- Split the landing page into two clearly labeled tracks. A recruiter can tell in ten seconds whether they are looking at a generalist or a specialist. The matrix format in the README gave fast signal density.

_Later waves:_ to be filled.

## What to watch for (running log)

_Wave 0 kickoff:_
- Temptation to over-polish indexes before shipping real code. Indexes are low-cost; the flagship pair (Wave 3) is what recruiters actually scroll through.
- Risk of diverging narratives between English README and Spanish Resumen. Must audit with `/write audit` at each wave close.
- Local LLM stack (Ollama) will need hardware notes so reviewers do not assume cloud inference.

## Linked artifacts

- Plan file: `/home/andtega349/.claude/plans/hi-this-is-project-hashed-truffle.md`
- Portfolio inventory (pre-track): [`subagents_outputs/portfolio_inventory.md`](../../subagents_outputs/portfolio_inventory.md)
- Project index: [`projects/00_pure_ds_index.md`](../../projects/00_pure_ds_index.md)
- Theory library: [`docs/theory/00_index.md`](../theory/00_index.md)
- Career toolkit: [`career/README.md`](../../career/README.md)

## Wave 1 Audit Outcomes (2026-04-25)

The first full curriculum audit (`subagents_outputs/syllabus_audit_2026_04_25.md`) reviewed all 15 theory documents, 60 technique folders, and the references library. Headline counts: 17 HIGH-priority items, 13 MED, 10 LOW. The audit confirmed the track has shipped enough surface area to read as a generalist DS portfolio, but exposed structural weaknesses that the catalog view of `00_index.md` was hiding.

### Three biggest structural problems

- **Missing foundations.** The 15-doc sequence opens with bias-variance and regularization without ever establishing probability, MLE, optimization, or gradient descent. An actuarial reader has these from school but a generalist DS reader does not, and the docs that follow read as ungrounded. Fix: insert a probability + estimation primer and an optimization + loss primer before doc 01.
- **Outdated LLM coverage.** Doc 09 carries the 2022 claim that chain-of-thought "works best with models above roughly 70B parameters" while reasoning-distilled 7B and 14B models in 2026 explicitly do CoT. Mixture of experts, speculative decoding, long-context (RoPE scaling, YaRN), reranker selection for RAG, and quantization formats (GPTQ, AWQ, GGUF) are absent despite the doc carrying "2026" in its filename. Fix: rewrite doc 09 with 2024 to 2026 updates.
- **SQL as an afterthought.** Doc 15 sits at the end of the sequence, but in the Mexican market SQL is the first interview screen and a prerequisite for everything from feature engineering (05) to MLOps (13). It also lacks MERGE/UPSERT, lateral joins, window-frame semantics, and a worked EXPLAIN ANALYZE. Fix: extend doc 15 and reorder it into the foundations block.

### Actuarial gaps

- **GARCH absent from the time series doc.** Unforgivable for an actuarial / finance reader. ARCH, GARCH(p,q), GJR, and EGARCH belong in doc 11 with one paragraph each plus a code pointer.
- **Mack chain ladder missing from `loss-reserving`.** The folder covers the mean estimate but not the closed-form variance, ODP bootstrap reserving, or Cape Cod. CAS Exam 5 and MAS-II expect Mack.
- **No IFRS 17 mention anywhere.** The standard has been in force since 2023; for any reserving or capital-modeling role the absence reads as out-of-date training.

### What was preserved as-is

The audit explicitly flagged ten files as strong enough to leave untouched (cosmetic edits only): doc 04 (data leakage), `techniques/glm/README.md`, `techniques/regression/README.md`, `techniques/survival-analysis/README.md`, `techniques/ratemaking-basics/README.md`, `techniques/bayesian-credibility/README.md`, `techniques/tweedie-compound-models/README.md`, `techniques/cross-validation/README.md`, `techniques/model-calibration/README.md`, and `techniques/shap-interpretability/README.md`. These set the pedagogical bar (lemonade-stand analogy, weather-forecaster framing, Alice/Bob/Carol Shapley walkthrough) for everything else.

### Next action

Three parallel agents execute the rewrite in HIGH, MED, and LOW priority lanes against the manifest in Section 7 of the audit. Each agent works directly on the live files, not copies. Once the wave closes, regenerate the theory PDF bundle via `/doc technical docs/theory/` so the offline reading copy matches the repository.
