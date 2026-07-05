# Problem Framing: Multilingual Review Classification

## The business problem

Product reviews in Spanish and English arrive at retailers, marketplaces, and service providers in volumes no human team can triage. A Mexican electronics retailer operating nationally plus selling cross-border generates tens of thousands of reviews per month across Amazon Mexico, Mercado Libre, and their own channels.

Without automation, three pain points compound:

1. **Negative reviews go unanswered.** Support teams handle tickets first; reviews are downstream. Response times of days or weeks are common.
2. **Trends in complaints go unnoticed.** Product managers cannot read every review. If 200 customers complain about a specific SKU in the past week, it may take another month before it surfaces in a meeting.
3. **Bilingual customer bases double the work.** Teams either staff bilingual reviewers or rely on imperfect machine translation that loses tone.

The classification problem: given a review text, predict its sentiment (positive, neutral, negative). Route negatives to support, feed aggregates to product teams.

## Why not just use machine translation plus a single-language model?

Translating Spanish to English and running an English model loses signal:

- Tone and idiom translate poorly. Mexican Spanish markers (neta, caro, me quede con ganas) lose precision when translated.
- Translation latency adds to inference time.
- Pipeline complexity doubles.

A native multilingual model keeps semantics in the original language.

## Alternative methods we considered

1. **Rule-based (keyword lists).** Simple, fast, interpretable. Fails on negation ("no esta mal" is positive), sarcasm, and typos. Accuracy plateau around 0.65.
2. **Sentiment lexicons (VADER, SentiWordNet).** Slightly better than keyword lists. Language-specific; English VADER does not transfer to Spanish.
3. **Classical ML with TF-IDF.** Logistic regression on TF-IDF vectors. Strong baseline, ~0.72-0.75 macro F1 on this dataset. Interpretable. No transfer learning benefit.
4. **Pretrained transformer, fine-tuned.** Current best practice. ~0.85-0.88 macro F1. Requires GPU for training but not for inference.
5. **Zero-shot large LLM (GPT-4, Claude).** Near-SOTA without training; cost and latency constraints for high-volume inference.

We chose option 4 (fine-tuned transformer) with option 3 (TF-IDF baseline) for comparison because:

- Budget is zero (local GPU or free Colab suffices for training).
- Inference latency must be under 100 ms per review; an API-backed LLM is borderline.
- The data has clear class signal; fine-tuning extracts it cleanly.
- TF-IDF baseline sets a floor and quantifies the value added by the transformer (recruiter signal: "I know how to measure the value of complexity").

## Success metrics

### Model-level

- **Macro F1 per language.** Target above 0.85; baseline above 0.75 (value-add above 10 percentage points).
- **Confusion matrix.** Even errors across classes; avoid a model that predicts "neutral" for everything.
- **Calibration.** Predicted confidence correlated with correctness; useful if downstream triage uses thresholds.

### Business-level (if deployed)

- **Negative-review response time.** Median from review creation to first support response.
- **Triage accuracy.** Of items flagged negative, what fraction were correctly negative (precision).
- **Coverage.** Of all negative reviews, what fraction were flagged (recall).
- **Time saved.** Reviewer-hours per week replaced by automated triage.

### Non-goals

- Not building a translation system.
- Not detecting fraud or fake reviews (a different problem).
- Not identifying specific product defects (NER / IE, different project).
- Not predicting star ratings with regression (business wants classes).

## Stakeholders

- **Customer Experience lead.** Cares about response time and triage accuracy.
- **Merchandising.** Cares about aggregates per SKU.
- **Product management.** Cares about emerging complaint patterns.
- **Legal / compliance.** Cares about bias (model should not over-flag one language).

Each gets a different view of the output: CX sees real-time alerts; merchandising sees dashboards; PM sees weekly reports; compliance gets a per-language fairness audit.

## Risks and ethical considerations

- **Language bias.** If the model is consistently weaker in Spanish, Spanish-speaking customers get slower responses. Explicit per-language evaluation and balanced training data mitigate but do not eliminate.
- **Review tone vs validity.** A politely worded complaint is still a complaint; a rude positive review still expresses satisfaction. Model should latch on to sentiment content, not politeness. Audit for this in error analysis.
- **Drift.** Review styles shift (new slang, new product categories). Plan for quarterly retraining and drift monitoring.
- **Misuse.** The model could be used to suppress or hide negative reviews rather than address them. The tool should surface, not filter. Reviewers documented this in the project's responsibility note.

## Project scope

In scope:

- Binary/ternary sentiment classification on `es` and `en` portions of Amazon Reviews Multi.
- Baseline TF-IDF + transformer fine-tuning.
- Local deployment plus Streamlit demo.
- Error analysis and interpretability.

Out of scope:

- Other languages (French, German, etc. in the full dataset).
- Topic or aspect extraction beyond a simple optional multi-label head.
- Live data collection.
- Production-grade MLOps (handled in the `mlops-model-service` project).

## Links to cross-cutting theory

- `docs/theory/09_llm_foundations_2026.md` covers the transformer architecture.
- `docs/theory/06_model_interpretability_shap_lime.md` covers SHAP and attention-based interpretability.
- `docs/theory/07_fairness_and_responsible_ai.md` covers per-language fairness audit framing.
- `docs/theory/03_cross_validation_strategies.md` covers stratified splits used here.
