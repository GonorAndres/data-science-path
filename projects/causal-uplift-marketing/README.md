# causal-uplift-marketing

## Resumen (ES)

**Problema:** a quien enviar una campana de marketing? No a todos: algunos iban a comprar de cualquier forma, otros no van a comprar sin importar que hagamos. Nos interesan los "persuadibles".

**Metodo:** modelado de uplift usando Criteo Uplift Prediction. Tres modelos: T-learner, S-learner, R-learner. Metricas Qini y uplift@K. Simulacion de A/B test con poder estadistico.

**Resultado objetivo:** uplift@10 mayor a 2x el baseline aleatorio; recomendacion de negocio con intervalo de confianza.

## Summary (EN)

**Problem:** who to target in a marketing campaign? Not everyone: some would buy regardless, some would never buy no matter what we do. We want "persuadables."

**Method:** uplift modeling on the Criteo Uplift Prediction dataset. Three models: T-learner, S-learner, R-learner. Qini and uplift@K metrics. A/B test simulation with power analysis.

**Target result:** uplift@10 at least 2x over random baseline; business recommendation with confidence interval.

---

## Dataset

Criteo Uplift Prediction (https://ailab.criteo.com/criteo-uplift-prediction-dataset/). 13M rows with treatment flag, outcome, 12 anonymized features. Public.

## Structure

```
causal-uplift-marketing/
├── README.md
├── theory/
│   ├── 00_problem_framing.md
│   ├── 01_potential_outcomes.md
│   ├── 02_ab_testing_math.md
│   ├── 03_uplift_modeling.md
│   └── 04_sensitivity_and_assumptions.md
├── data_download.py
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── t_learner.py
│   ├── s_learner.py
│   ├── r_learner.py
│   ├── evaluate.py
│   └── ab_simulation.py
├── tests/
├── notebooks/
├── requirements.txt
└── Makefile
```

## Quickstart

```bash
pip install -r requirements.txt
python data_download.py
python -m src.t_learner
python -m src.s_learner
python -m src.r_learner
python -m src.evaluate
python -m src.ab_simulation
```

## Decisions and trade-offs

| Decision | Alternative | Why |
|----------|-------------|-----|
| Uplift modeling (not classification) | Propensity model | We care about the treatment effect, not the outcome probability. |
| Three-meta-learner comparison | Single model | Different learners fail differently; comparison is the lesson. |
| Qini curve | ROC-like metric | Qini is the standard uplift metric; captures treatment effect monotonically. |
| Criteo | Lenta, X5 | Criteo is larger and anonymized, easier to share. |

## References

- `docs/theory/08_experimentation_causal_primer.md`
- Athey, Imbens, "Machine Learning Methods for Estimating Heterogeneous Causal Effects", 2016.
- Kunzel et al., "Meta-learners for Estimating Heterogeneous Treatment Effects", 2019.
- `causalml` and `scikit-uplift` libraries.
