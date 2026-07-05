# nlp-reviews-multilingual

## Resumen (ES)

**Problema:** clasificar resenas de productos en espanol e ingles por sentimiento (positivo, neutral, negativo) y, opcionalmente, por aspecto (calidad, precio, entrega, servicio).

**Datos:** Amazon Reviews Multi (Keung et al. 2020), 1.2M resenas en 6 idiomas. Se usan las particiones en espanol e ingles.

**Metodo:** Fine-tuning de un transformer multilingue (DistilBERT multilingue o XLM-RoBERTa base) con baseline TF-IDF + regresion logistica para comparacion. Analisis de errores por idioma, SHAP para explicabilidad lexica.

**Resultado objetivo:** F1 macro mayor a 0.85 en cada idioma. Baseline TF-IDF alrededor de 0.75. Demo desplegada en Streamlit Cloud.

## Summary (EN)

**Problem:** classify product reviews in Spanish and English by sentiment (positive, neutral, negative) and optionally by aspect (quality, price, delivery, service).

**Data:** Amazon Reviews Multi (Keung et al. 2020), 1.2M reviews in 6 languages. Spanish and English splits used here.

**Method:** fine-tune a multilingual transformer (multilingual DistilBERT or XLM-RoBERTa base) with a TF-IDF + logistic regression baseline for comparison. Per-language error analysis. SHAP for lexical explainability.

**Target result:** macro F1 above 0.85 in each language. TF-IDF baseline around 0.75. Streamlit Cloud deployed demo.

---

## Business framing

A retailer or marketplace operating in both Spanish and English markets needs to monitor product sentiment at scale. Manual triage of reviews does not scale past a few hundred a day. A multilingual classifier:

- Flags negative reviews in real time for support to follow up.
- Aggregates sentiment per SKU, per category, per channel for merchandising.
- Lights up aspect breakdowns so product teams know whether complaints are about price, quality, or service.

The stakeholder is a Customer Experience or Merchandising team. The business metric is time-to-response for negative reviews plus overall CSAT.

## Why this problem for the pure-DS track

- Demonstrates **multilingual NLP**, a concrete recruiter signal (Mexican market prizes bilingual capability).
- Demonstrates **fine-tuning HuggingFace transformers**, currently standard practice.
- Demonstrates **baseline discipline** (TF-IDF first), which separates methodical candidates from ones who jump to the fanciest tool.
- Demonstrates **error analysis and interpretability**, not just headline metrics.
- Pairs directly with `mlops-model-service`, which wraps this model in a production-style FastAPI service.

## Dataset

Amazon Reviews Multi by Keung et al. (2020). Public via HuggingFace Datasets (`amazon_reviews_multi`).

- Languages used: `es`, `en`.
- Labels derived from star ratings: 1-2 negative, 3 neutral, 4-5 positive.
- Train: 200k Spanish + 200k English. Validation: 5k each. Test: 5k each.

Run `python data_download.py` to fetch and cache locally in `./data/`.

## Approach

### Stage 1: EDA

See `notebooks/01_eda.ipynb` (outlined in `notebooks/01_eda.md`).

- Class balance per language.
- Review length distribution.
- Common words per class per language.
- Sanity check for label noise.

### Stage 2: Baseline (TF-IDF + logistic regression)

- Spanish and English pipelines separately.
- 1 and 2-grams, min_df=5, max_features=50k.
- Logistic regression with L2.
- Metric: macro F1.

Expected baseline: ~0.72 ES, ~0.75 EN (varies slightly by seed).

### Stage 3: Transformer fine-tuning

- Model: `distilbert-base-multilingual-cased` or `xlm-roberta-base`.
- Training: 3 epochs, learning rate 2e-5, batch 32, weight decay 0.01.
- Frozen for first epoch (head only), then unfrozen.
- Tracked with MLflow.
- Checkpoint at best validation macro F1.

Expected: ~0.85-0.88 per language.

### Stage 4: Aspect-based classification (optional)

If time permits: lightweight multi-label head on aspect categories (quality, price, delivery, service). Use the existing backbone; train a second linear head.

### Stage 5: Error analysis

- Confusion matrix per language.
- Per-class precision and recall.
- Representative misclassifications (10 per class).
- SHAP explanations for the TF-IDF baseline; attention highlighting for the transformer.

### Stage 6: Deployment

- Streamlit Cloud demo (text-in, label-out, SHAP highlight).
- Project pairs with `mlops-model-service` for a full FastAPI + Cloud Run deployment.

## Repository structure

```
nlp-reviews-multilingual/
├── README.md                      # this file
├── theory/
│   ├── 00_problem_framing.md
│   ├── 01_transformers_and_tokenization.md
│   ├── 02_fine_tuning_vs_zero_shot.md
│   └── 03_evaluation_and_bias.md
├── data_download.py               # fetch Amazon Reviews Multi
├── src/
│   ├── __init__.py
│   ├── data.py                    # load, preprocess, split
│   ├── baseline.py                # TF-IDF + logistic regression
│   ├── train.py                   # fine-tune transformer
│   ├── evaluate.py                # metrics, confusion matrix
│   └── predict.py                 # single-input inference
├── tests/
│   ├── test_data.py
│   └── test_predict.py
├── notebooks/
│   ├── 01_eda.md                  # outline; converts to .ipynb
│   ├── 02_baseline.md
│   └── 03_transformer.md
├── requirements.txt
├── Makefile
└── .streamlit/
    └── app.py                     # Streamlit demo
```

## Quickstart

```bash
cd projects/nlp-reviews-multilingual
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python data_download.py
python src/baseline.py            # fit TF-IDF baseline, log to MLflow
python src/train.py               # fine-tune transformer, log to MLflow
python src/evaluate.py            # full evaluation on test
streamlit run .streamlit/app.py   # local demo
```

## Testing

```bash
pytest tests/
```

## Decisions and trade-offs

| Decision | Alternative | Why we chose this |
|----------|-------------|-------------------|
| Multilingual DistilBERT | Separate model per language | Single model is simpler to deploy; ES and EN share substantial vocabulary. |
| 3-class sentiment | 5-star regression | Downstream teams act on classes, not fractional stars. Regression is noisier. |
| Fine-tuning | Frozen features + head | Fine-tuning is ~5 pp better on this dataset; cost is still modest. |
| SHAP on baseline only | SHAP on transformer | TreeSHAP does not apply to transformers. Transformer uses attention highlights and integrated gradients instead. |
| Amazon Reviews Multi | TripAdvisor MX dataset | ARM is public and standardized; TripAdvisor would require scraping. |

## Retrospective (to fill after first training run)

- **What went well:**
- **What to improve:**
- **Skills developed:**

## Presentation notes (non-technical)

- Lead with: "We built a tool that reads product reviews in Spanish and English and flags the negative ones for the support team."
- Show the Streamlit demo live: type a review, watch the model label it, hover over highlighted words.
- Business metric: "Before, ~30 percent of negative reviews got a response in 24 hours. With this, we could get that to 80 percent because triage is automated."
- Caveat: "The model is good, not perfect. About 15 percent of reviews require human judgment, especially edge cases."

## References

- Keung et al., "The Multilingual Amazon Reviews Corpus", EMNLP 2020.
- Conneau et al., "Unsupervised Cross-lingual Representation Learning at Scale" (XLM-R), 2020.
- HuggingFace Transformers documentation.
- `docs/theory/09_llm_foundations_2026.md` for transformer background.
- `docs/theory/06_model_interpretability_shap_lime.md` for SHAP.
