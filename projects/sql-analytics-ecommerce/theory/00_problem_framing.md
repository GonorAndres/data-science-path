# Problem Framing: E-commerce Analytics

## What questions we answer

1. How has revenue grown month over month?
2. How does signup-cohort retention look at 1, 3, 6, 12 months?
3. Which product categories drive revenue? Which grow fastest?
4. What is the lifetime value of customers acquired through different channels?
5. Which products have the worst review scores, and why?
6. How does the funnel work: from visit to checkout to purchase to repeat?

These are the standard questions a product, finance, or operations stakeholder asks. SQL is the universal tool for answering them at scale.

## Stakeholders

- **Finance.** Cares about revenue trends, cohort LTV.
- **Marketing.** Cares about channel performance, acquisition cost vs LTV.
- **Product.** Cares about conversion funnel, feature adoption.
- **Merchandising.** Cares about category performance, seasonality.
- **CX / Support.** Cares about review trends, complaint categories.

Each gets a different dashboard built on the same underlying marts.

## Why Olist

Olist's dataset has:

- Realistic schema (orders, customers, products, payments, reviews, geolocation).
- Multiple years (2016-2018) for trend analysis.
- Non-trivial size (100k orders) that exercises performance patterns.
- Public under CC license.

It is missing:

- Traffic data (no session or pageview records). We cannot build a full visit-to-purchase funnel; we can build purchase-stage funnel.
- Acquisition channel per customer. We proxy with payment method or seller as a rough channel signal.

These limitations are documented per mart.

## Cross-links

- `docs/theory/15_sql_for_data_scientists.md`: broader SQL context.
- `01_dimensional_modeling.md`: how the staging/marts split maps to a dimensional model.
