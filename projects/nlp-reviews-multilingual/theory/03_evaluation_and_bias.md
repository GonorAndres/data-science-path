# Evaluation and Bias Analysis

## Primary metric: macro F1

Macro F1 averages per-class F1 with equal weight, which penalizes models that do well on majority classes and poorly on minority classes. On balanced data macro F1 is close to accuracy; on imbalanced data it tells a much fairer story.

For Amazon Reviews Multi, star-rating-derived classes are moderately imbalanced: ~50 percent positive, ~30 percent neutral, ~20 percent negative. Macro F1 gives the negative class equal weight, which aligns with the business priority (we care about negatives most).

## Secondary metrics

- **Per-class precision and recall.** Where is the model strong or weak?
- **Confusion matrix.** What does the model confuse?
- **ROC-AUC and PR-AUC** (one-vs-rest). For threshold tuning if the downstream system uses confidence-based triage.
- **Calibration.** Reliability diagrams; Brier score. Matters if thresholds are used.
- **Macro F1 per language.** Our fairness concern.

## Cross-language bias audit

Because we train one model for two languages, we must confirm performance is comparable across languages. Protocol:

1. Compute macro F1 on the test set, overall.
2. Compute macro F1 on the Spanish subset.
3. Compute macro F1 on the English subset.
4. The gap `|F1_ES - F1_EN|` should be under 3 percentage points. If larger, investigate.

Common causes of language bias:

- **Training-data imbalance.** If English has more training examples, the model tilts toward English.
- **Tokenizer bias.** Rare Spanish words split into many pieces; short Spanish expressions fragment.
- **Label noise asymmetry.** Star-to-class mapping might mean different things across regions.

Mitigations (in order of cost):

1. Check training data balance per language. Rebalance if needed.
2. Fine-tune longer on the underperforming language.
3. Use a larger multilingual model (XLM-R base, then large).
4. Train separate per-language heads on a shared backbone (multi-task).
5. Augment the underperforming language (back-translation, paraphrase).

## Sub-group analysis within languages

Beyond language, stratify by:

- **Review length.** Short reviews (under 20 words) are harder. Quantify.
- **Star rating.** The boundary between 3-star (neutral) and 4-star (positive) is noisy. Quantify.
- **Product category.** If the dataset has category metadata, check per-category F1.

Sub-group reports surface failure modes that aggregate metrics hide.

## Error analysis protocol

For each language:

1. Sample 10 to 20 misclassified reviews per class.
2. Categorize each error:
   - **Sarcasm** ("Qué maravilla que se rompio en un día").
   - **Negation** ("no esta mal" -> positive; model may miss).
   - **Mixed sentiment** (good product, bad shipping).
   - **Typos and slang.**
   - **Ambiguous ratings** (3-star neutral reviews that read positive or negative).
   - **Code-switching** (ES reviews with English words and vice versa).
3. Count error categories; identify the top two or three to target in the next iteration.

## SHAP and attention-based interpretability

### TF-IDF baseline

For the linear model, coefficients give a direct interpretation: what words push the prediction toward each class. SHAP on LogReg is equivalent to the coefficient times the feature value, but SHAP normalizes across examples and supports local explanations.

### Transformer

SHAP for transformers is approximate (KernelSHAP is slow; specialized methods like Captum's Layer Integrated Gradients are faster). Alternatives:

- **Attention rollout / attention map.** Sum attention weights from the `[CLS]` token to each input token. Caveat: attention is not explanation (Jain and Wallace, 2019), but it is a useful debug view.
- **Integrated Gradients.** Sensitivity of the output to each input token. More principled than raw gradients.
- **LIME.** Perturb the input, learn a local linear model. Model-agnostic.

For the demo, LIME highlights on the user's input work reliably and are easy to explain.

## Reporting template

At end of project:

- **Headline:** "Macro F1 on test: 0.86 (Spanish 0.85, English 0.87). Baseline: 0.74. Gain from transformer: 12 pp."
- **Plot:** Confusion matrix per language.
- **Plot:** Per-class precision/recall bars.
- **Table:** Per-error-category counts from error analysis.
- **Plot:** Calibration reliability diagram.
- **Narrative:** What worked, where the model struggles, what to do in v2.

This report is what the `docs/lessons-learned/` retrospective draws from.

## Stress tests

Before deploying, run these sanity checks:

1. **Empty string.** Model should return neutral / abstain, not crash.
2. **Very long string.** Truncation should kick in cleanly.
3. **Pure emoji review.** Test whether the tokenizer handles.
4. **Code-switched review.** "Me gusto a lot" should not break.
5. **Obvious negation.** "No esta mal" -> should be positive or neutral, not negative.
6. **Obvious sarcasm.** Likely misclassified; document rather than pretend.
7. **Adversarial input.** Random typos, leetspeak, copy-pasted ads.

Include these in `tests/test_predict.py`.

## Fairness beyond languages

In some deployments, additional protected attributes matter. Possible extensions:

- **Region within language.** Mexican Spanish vs Castilian Spanish have lexical differences. If the dataset tags origin, report per-variant F1.
- **Author demographics.** Usually not available in review data. Skip.
- **Product price tier.** A systematic bias against low-price reviews (more negative) could conflate price and sentiment.

Document which analyses were run and which were not possible due to data limitations.

## Responsible deployment

Beyond accuracy and bias:

- **Human-in-the-loop for low-confidence predictions.** Set a confidence threshold; uncertain reviews go to a human reviewer.
- **Escalation path.** Negative reviews with high confidence get support tickets; any prediction the user disputes goes through review.
- **Periodic re-audit.** Monthly re-check of per-language F1 on new data.
- **Sunset criteria.** If drift erodes performance beyond 3 pp, retrain; if a better open model arrives, migrate.

## Linking to the theory library

- `docs/theory/07_fairness_and_responsible_ai.md` for fairness framing.
- `docs/theory/06_model_interpretability_shap_lime.md` for SHAP / LIME.
- `docs/theory/03_cross_validation_strategies.md` for stratified evaluation.

## Linking to the code

- `src/evaluate.py` implements macro F1, confusion matrix, per-language breakdown.
- `tests/test_predict.py` covers the stress tests listed here.
- `notebooks/04_error_analysis.md` outlines the error-analysis notebook.
