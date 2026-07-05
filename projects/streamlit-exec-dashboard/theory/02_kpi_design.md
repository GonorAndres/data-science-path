# KPI Design

## What makes a good KPI

1. **Specific.** Tied to a clear business question.
2. **Measurable.** Computable from data you have.
3. **Actionable.** Stakeholders can change it through decisions.
4. **Time-bound.** Over a defined period.
5. **Comparable.** Has a baseline (target, last period, peer).

## KPI tile anatomy

Each tile on the overview page includes:

- **Label.** What the metric is (ES + EN).
- **Value.** The current number.
- **Baseline.** Previous period's number.
- **Delta.** Absolute or percent change.
- **Direction indicator.** Arrow or color.
- **Sparkline.** Trend over last N periods.

## Which KPIs for this dashboard

- Monthly revenue (current vs last month, vs same month last year).
- Monthly active customers.
- Average order value.
- 28-day retention (cohort-normalized).
- Average delivery time (days from purchase to delivered).
- NPS proxy (average review score).

Six KPIs fit on one tile row without crowding.

## Avoiding vanity metrics

- Total users (ever signed up, without filter). Growing forever but meaningless.
- Pageviews. Without conversion context, not actionable.
- Social media followers. Often not tied to business outcomes.

Swap for: active users (last N days), conversion rate, revenue per active user.

## Normalization

Raw numbers hide per-unit effects. Always prefer:

- Revenue per active customer (not total revenue).
- Retention rate (not raw returning count).
- Error rate per 1K transactions (not total errors).

## Target lines

If the business has a target, show it. A KPI with no target is a number; a KPI with a target is a status.

## Cross-links

- `01_storytelling_for_execs.md`
- `03_anomaly_visual_cues.md`
