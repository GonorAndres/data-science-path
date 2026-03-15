# Insurance Pricing: Lessons from Europe for Mexico's Auto Insurance Market

**Business question**: What can Mexico's auto insurance market, where only 30% of vehicles are insured and most pricing relies on traditional methods, learn from Europe's actuarial data science revolution?

## Why This Matters

- Mexico is the only OECD country without federal mandatory auto liability insurance
- Only 15-20% of Mexican insurers use AI/ML in pricing (vs 60%+ in Europe)
- Smarter pricing means fairer premiums, which could expand insurance adoption from 30% to higher levels
- Mexico adopted Solvency II (LISF 2015) but has no ML-specific pricing regulation yet

## Approach

**Frequency-severity decomposition** (actuarial standard):
- Pure premium = E[claim frequency] x E[claim severity]
- Frequency: Poisson GLM vs XGBoost vs LightGBM
- Severity: Gamma GLM vs XGBoost

**Dataset**: freMTPL2 (French Motor Third-Party Liability): 677,991 policies, the gold-standard benchmark in actuarial data science.

**Key innovations**:
1. Cross-border framing: European techniques analyzed for Mexican market applicability
2. Hybrid GLM+ML approach: GBM audits GLM residuals to find where traditional models fail
3. SHAP explainability: satisfies emerging regulatory requirements
4. Fairness audit: checks if Area/Density acts as proxy for socioeconomic discrimination

## Results

| Model | Gini | D-squared | Poisson Deviance |
|-------|------|-----------|-----------------|
| Poisson GLM | ~0.23 | ~0.043 | ~0.326 |
| XGBoost | ~0.27 | ~0.055 | ~0.318 |
| LightGBM | ~0.28 | ~0.058 | ~0.316 |

*Results match benchmarks from Noll et al. (2020) and scikit-learn's official tutorial.*

## Running the Project

```bash
# 1. Train models and generate dashboard data
cd projects/insurance-pricing
python src/train.py --trials 100

# 2. Start the API backend (port 2060)
cd backend
uvicorn pricing_backend.main:app --port 2060

# 3. Start the dashboard (port 3060)
cd dashboard
npm install && npm run dev

# 4. Run tests
pytest tests/
```

## Tech Stack

- **Modeling**: statsmodels (GLM), XGBoost, LightGBM, Optuna, MLflow
- **Explainability**: SHAP (TreeExplainer)
- **Backend**: FastAPI + Uvicorn (pre-computed JSON, no live inference)
- **Frontend**: Next.js 14, Recharts, SWR, Tailwind CSS, Framer Motion

## Dashboard Tabs

1. **Contexto**: Mexico's market problem, GLM-vs-ML debate, regulatory comparison, methodology
2. **EDA Explorer**: Feature distributions, claim frequency heatmap, exposure analysis
3. **Comparacion de Modelos**: KPIs, lift, calibration, GLM coefficients, SHAP, fairness audit
4. **Proceso Tecnico**: Architecture, MLflow experiments, implications for Mexico, retrospective

## Academic References

- Noll, Salzmann & Wuthrich (2020): GBM outperforms GLM on freMTPL2
- Colella & Jones, CAS E-Forum (2023): No universal optimal model
- MDPI Risks (2024): Hybrid GLM+ANN beats all individual models
- Kuo & Lupton, Variance (2023): SHAP needed for regulatory review

## Retrospective

**What worked well**:
- Frequency-severity decomposition (standard actuarial practice, differentiates from typical ML projects)
- Optuna + MLflow for systematic experiment tracking
- Cross-border framing makes this more than a model-fitting exercise

**What to improve**:
- Add CatBoost and Explainable Boosting Machine for more complete comparison
- Implement Tweedie GLM (models pure premium directly)
- Bootstrap confidence intervals for metrics
- Mexican data (AMIS/CNSF) when available
