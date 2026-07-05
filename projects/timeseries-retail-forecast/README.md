# timeseries-retail-forecast

## Resumen (ES)

**Problema:** forecast jerarquico de demanda diaria sobre el dataset M5 (Walmart, Kaggle). Predecir ventas por SKU-tienda para el siguiente horizonte de 28 dias.

**Metodo:** tres capas: (1) naive estacional como piso; (2) Prophet y ETS por serie; (3) LightGBM global con features de rezago, calendario y promociones. Reconciliacion jerarquica MinT. Intervalos conformales para incertidumbre.

**Resultado objetivo:** WAPE alrededor de 9-11 por ciento en top-down, bias cerca de cero, cobertura empirica de intervalo 80 por ciento entre 78 y 82 por ciento.

## Summary (EN)

**Problem:** hierarchical daily-demand forecasting on the M5 dataset (Walmart, Kaggle). Predict sales per SKU-store for the next 28 days.

**Method:** three layers: (1) seasonal-naive floor; (2) Prophet and ETS per series; (3) global LightGBM with lag, calendar, and promotional features. MinT hierarchical reconciliation. Conformal intervals for uncertainty.

**Target result:** WAPE around 9 to 11 percent top-down, bias near zero, 80 percent interval coverage between 78 and 82 percent.

---

## Business framing

Retail demand forecasting is the backbone of inventory, staffing, and pricing decisions. Under-forecasting causes stockouts and lost revenue; over-forecasting causes waste and carrying cost. The business metric is usually WAPE (weighted by revenue or volume) because it naturally weights the items that matter financially.

M5 is a well-known benchmark because it mirrors the real problem: thousands of SKU-store combinations, strong seasonality (weekly, annual, holiday), intermittent demand for long-tail items, and known covariates (price, SNAP calendar, events).

## Repository structure

```
timeseries-retail-forecast/
├── README.md
├── theory/
│   ├── 00_problem_framing.md
│   ├── 01_decomposition_and_stationarity.md
│   ├── 02_classical_vs_ml_forecasts.md
│   ├── 03_temporal_cv.md
│   └── 04_hierarchical_forecasting.md
├── data_download.py              # fetch M5 from Kaggle
├── src/
│   ├── __init__.py
│   ├── data.py                   # loading and long-form reshaping
│   ├── features.py               # lag, rolling, calendar features
│   ├── baselines.py              # seasonal naive
│   ├── classical.py              # Prophet / ETS per-series
│   ├── gbm.py                    # global LightGBM with lag features
│   ├── reconcile.py              # hierarchical reconciliation
│   ├── evaluate.py               # WAPE, bias, interval coverage
│   └── conformal.py              # conformal intervals
├── tests/
├── notebooks/
├── requirements.txt
└── Makefile
```

## Quickstart

```bash
cd projects/timeseries-retail-forecast
pip install -r requirements.txt

# Download M5 (requires Kaggle API credentials in ~/.kaggle/kaggle.json)
python data_download.py

python -m src.baselines       # compute seasonal-naive baseline
python -m src.classical       # Prophet / ETS
python -m src.gbm             # LightGBM global
python -m src.reconcile       # hierarchical reconciliation
python -m src.evaluate        # final WAPE table
```

## Expected results

| Model | Top-down WAPE | Bottom-up WAPE | Avg WAPE | Runtime |
|-------|---------------|----------------|----------|---------|
| Seasonal naive | 15-18% | 25-30% | 22% | 1 min |
| Prophet (per series) | 13-15% | 20-25% | 18% | 4-8 h |
| ETS (per series) | 12-14% | 19-22% | 17% | 2-4 h |
| LightGBM (global) | 9-11% | 14-17% | 12% | 1-2 h |
| LightGBM + MinT | 9-10% | 13-15% | 11% | +10 min |

Numbers are approximate and vary with hyperparameters.

## Decisions and trade-offs

| Decision | Alternative | Why we chose this |
|----------|-------------|-------------------|
| LightGBM global | Per-series model | Global models exploit cross-series patterns; cheaper to retrain; M-competitions confirm this. |
| MinT reconciliation | Bottom-up only | Reconciliation improves aggregates without sacrificing leaves. |
| Conformal intervals | Quantile regression | Conformal gives calibrated coverage without distributional assumptions. |
| M5 dataset | Favorita or Rossmann | M5 is the most recent competition benchmark; strong baselines to compare. |

## Retrospective (to fill after first run)

- **What went well:**
- **What to improve:**
- **Skills developed:**

## References

- `docs/theory/11_time_series_foundations.md`
- `docs/theory/03_cross_validation_strategies.md`
- Makridakis et al., "The M5 accuracy competition: Results, findings, and conclusions", 2022.
- Hyndman, Athanasopoulos, *Forecasting: Principles and Practice*, 3rd ed.
