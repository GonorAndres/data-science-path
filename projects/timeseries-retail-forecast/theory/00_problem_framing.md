# Problem Framing: Hierarchical Demand Forecasting

## The decision the forecast enables

Retail operates on a weekly cycle of replenishment. The chain of decisions:

1. Forecast demand per SKU per store for the next 1 to 4 weeks.
2. Compare to on-hand inventory plus incoming shipments.
3. Generate replenishment orders for suppliers.
4. Staff stores for the expected volume.
5. Price-promote overstocked items.

Each step is sensitive to forecast quality. An under-forecast produces stockouts (lost sales, customer dissatisfaction). An over-forecast produces carrying cost, markdowns, and waste on perishable items.

## Stakeholders

- **Supply chain planners.** Own the replenishment decisions. Care about bias and lead-time horizons.
- **Merchandising.** Plan promotions and pricing. Care about short-horizon accuracy around promoted SKUs.
- **Store operations.** Staff to demand. Care about weekly aggregates.
- **Finance.** Quarterly revenue plan. Care about the top-level totals, not per-SKU detail.

Each stakeholder consumes a different level of the hierarchy, which is why hierarchical reconciliation matters: the forecasts must be internally consistent.

## Non-goals

- Not predicting promotion uplift separately (that is its own causal problem).
- Not optimizing inventory directly (that is a separate optimization step that consumes this forecast).
- Not forecasting margin (price x quantity x cost; we forecast quantity).

## Success metrics

| Metric | Target | Why |
|--------|--------|-----|
| Top-level WAPE | under 10 percent | Finance uses this number. |
| Bottom-level WAPE | under 15 percent | Replenishment tolerates noise at SKU-store. |
| Bias at each level | within +/- 1 percent | No systematic over- or under-forecast. |
| 80 percent interval coverage | 78 to 82 percent | Calibrated intervals enable safety-stock tuning. |
| Horizons 1, 7, 14, 28 days | All evaluated | Different stakeholders use different horizons. |

## Why M5

The M5 competition dataset matches the real problem:

- 42,840 SKU-store series from Walmart.
- Three product categories, seven departments, ten stores across three states.
- Five years of daily sales.
- Calendar, prices, promotional events available.
- Natural hierarchy (state -> store -> category -> department -> item).

Winning solutions were ensembles of LightGBM models. We follow that structure.

## Risks and caveats

- **Intermittent demand.** Many SKUs have zero sales on most days. Traditional continuous-forecast methods produce biased estimates. We address with LightGBM's natural ability to predict exact zeros and with Tweedie or Poisson objectives.
- **Promotions.** Sales spikes during promotions dwarf baseline demand. If the promotional calendar is not known accurately, the model mispredicts. M5 provides it; in production, this is often the main pipeline challenge.
- **Stockouts.** Historical zero sales may mean "no demand" or "out of stock". We cannot distinguish in raw M5 data; treat as noise.
- **Holidays and events.** SNAP days, Super Bowl, Christmas, Easter. Explicit features help; deep learning models learn them implicitly.

## Linking to the theory library

- `docs/theory/11_time_series_foundations.md`: general forecasting theory.
- `docs/theory/05_feature_engineering_cookbook.md`: lag and calendar features.
- `docs/theory/03_cross_validation_strategies.md`: temporal CV.
