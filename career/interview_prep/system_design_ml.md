# ML System Design: 5 Classic Problems

ML system design questions appear in mid and senior interviews and at top product companies (Mercado Libre, Nu, Salesforce, Microsoft, Google). The question is usually open-ended: "Design a [feed ranker, fraud system, recommender, forecaster, chatbot]." You have 45 to 60 minutes.

The interviewer grades on:

1. Do you ask the right scoping questions?
2. Do you cover the full lifecycle (data to serving to monitoring)?
3. Do you make tradeoffs explicit?
4. Do you think about what can break in production?

---

## The framework

Use this structure for every problem. Write it on the whiteboard.

1. **Clarify requirements and constraints.** Traffic (QPS), latency, data volume, freshness, budget, team size.
2. **Define metrics.** Both offline (proxy) and online (business).
3. **Data.** Sources, features, labels, update cadence.
4. **Model.** Candidate approaches, tradeoffs, baseline.
5. **Training pipeline.** Where, how often, how is it validated.
6. **Serving.** Online vs batch, latency budget, caching, fallback.
7. **Monitoring.** Model health, data health, drift, alerts.
8. **Experiment and rollout.** A/B test plan, kill switch.
9. **Iteration.** How would version 2 improve on version 1?

---

## Problem 1: Design a feed ranker (Twitter / X style)

### Requirements clarifying

- QPS? Millions of users, hundreds of ranked feeds per second per server.
- Latency budget for one feed render? 200 to 500 ms.
- Candidate set size? Could be 10,000 candidate items per user.
- Personalization freshness? Ideally sub-minute for breaking news.

### Metrics

- Offline: NDCG, CTR on ranked items (offline replay), dwell time prediction accuracy.
- Online: CTR, session length, revisit rate, engagement depth (likes, replies).
- Guardrail: Time-to-meaningful-action, session abandon rate, content diversity.

### Architecture

Two-stage retrieval then ranking:

1. **Retrieval (candidate generation):** For each user, produce ~1,000 candidate items from billions via fast retrievers: two-tower model (user embedding vs item embedding, approximate nearest neighbor), recent-content index, social graph.
2. **Ranking:** For ~1,000 candidates, run a more expensive model (GBM or small neural) that predicts engagement score. Return top K.

### Data and features

- User features: demographics, historical engagement, recent sessions.
- Item features: content embedding (from a text or multimodal model), popularity, recency, author features.
- Cross features: user-author affinity, user-topic match, time of day.

### Training

- Labels: engagement signals (click, like, follow, dwell time). Multi-task learning (predict each signal jointly).
- Cadence: daily or weekly full retrain; online learning for fast adaptation.
- Validation: time-based split (train on older weeks, validate on recent week).

### Serving

- User embedding cached in Redis, recomputed every session or daily.
- Item embedding store in a vector DB (FAISS, Milvus).
- Retrieval stage uses ANN index, 20 ms budget.
- Ranking stage uses a GPU service or a highly optimized CPU service, 50 ms budget.
- Full feed render under 200 ms.

### Monitoring

- CTR trend by cohort (detect shift).
- Model score distribution (detect training-serving skew).
- Feature drift (PSI on each input).
- Fallback: if model fails, show reverse-chronological feed.

### Tradeoffs

- Two-stage vs one-stage: one-stage is simpler but cannot rank all items. Two-stage scales.
- Online learning vs offline training: online adapts fast but is harder to validate. Offline is safer but stale.

---

## Problem 2: Fraud detection system for a payments API

### Requirements

- Latency: must decide approve / review / reject in under 100 ms (synchronous to payment flow).
- Volume: thousands of transactions per second at peak.
- Label: fraud is confirmed asynchronously (chargebacks arrive weeks later).
- Risk asymmetry: false negative (missed fraud) costs dollars and reputation; false positive costs customer friction.

### Metrics

- Offline: Precision, recall, PR-AUC, dollars-saved at threshold.
- Online: True fraud rate (confirmed later), chargebacks per million, false-positive rate (complaints from blocked customers).
- Guardrail: Median latency, approval rate (business would flag if blocked rate spikes).

### Architecture

- **Scoring service:** Synchronous API. Input transaction features, output risk score.
- **Action policy:** Threshold-based: if score > H1, reject; if H2 < score < H1, send to manual review; else approve.
- **Feedback loop:** Review decisions and chargebacks feed back as labels.

### Data and features

- Transaction-level: amount, currency, device, IP geo, merchant category.
- User-level: historical spending, device history, velocity features (N transactions in last 10 minutes).
- Graph features: shared device across accounts, shared billing address.
- Real-time features from a feature store with low-latency reads.

### Training

- Model: gradient boosted trees (XGBoost) as baseline. Scales to tabular with mixed feature types.
- Labels: confirmed fraud (positive), confirmed legit (negative). Challenge: label lag (confirmed outcomes available weeks after transaction). Solution: train on fully-labeled window (older data), serve on fresh data; accept some staleness.
- Class imbalance: very skewed (0.1 to 1 percent fraud). Use class weighting, not oversampling.

### Serving

- Scoring service is a pod running the trained model, pre-loaded. P99 latency under 100 ms.
- Features fetched from online feature store (Redis, DynamoDB, or Feast on Redis).
- Fallback: if service is down, fall back to rule-based risk engine.

### Monitoring

- Model score distribution drift (PSI against training distribution).
- Chargeback rate by score bucket.
- Feature staleness (are feature values arriving on time?).
- Custom alerts for unusual approval rate dips.

### Tradeoffs

- Deep learning vs GBM: DL can capture rare patterns but is harder to explain and debug; fraud teams prefer explainability.
- Strict rules vs model: a blended system (rules override model for known high-risk patterns, model handles the rest) is the production norm.

---

## Problem 3: Recommender for an e-commerce site

### Requirements

- Catalog size: 10 million SKUs.
- Users: 50 million.
- Latency: home page and product page recommendations must load under 300 ms.
- Cold start: new users and new products are common.

### Metrics

- Offline: Recall@K, NDCG@K, coverage, diversity.
- Online: CTR on recommended slot, add-to-cart rate, revenue per visit, category diversity in cart.
- Guardrail: Recommendation freshness (do not show what user just bought), diversity.

### Architecture

Two-stage:

1. **Retrieval:** Generate 500 to 1,000 candidate items per user.
   - Collaborative filtering (matrix factorization or two-tower) for known users.
   - Content-based (item embedding similarity) for cold users.
   - Popular-items fallback for completely new visitors.
2. **Ranking:** GBM or neural model predicting purchase probability.
3. **Post-filtering:** Remove items already owned, out of stock, or previously shown too recently.
4. **Diversity re-ranking:** Ensure the final list has category variety (MMR or similar).

### Data and features

- User: browse history, purchase history, demographics.
- Item: title, description (text embedding), category, price, popularity, stock.
- Context: session, time, device, traffic source.
- Cross: user-category affinity, user-price-tier preference.

### Training

- Labels: clicks, add-to-cart, purchase. Positive-only implicit feedback; no explicit negatives (use random or popularity-weighted sampling).
- Weekly retrain for the retrieval model; daily for the ranking model.
- Embedding refresh for new items as they land (or zero-shot from text embedding).

### Serving

- User embeddings cached per session.
- Item embeddings in a vector DB for ANN retrieval.
- Ranking model served via low-latency inference service.

### Cold start

- New item: use text or image embedding from a pretrained model until you have interaction data.
- New user: popularity + category affinity inferred from the first session; update after each click.

### Monitoring

- Coverage (what fraction of catalog is ever recommended).
- Drift in item popularity distribution (seasonal).
- CTR and purchase rate by segment.

### Tradeoffs

- Personalization vs exploration: too personalized and you lock users into echo chambers. Include some exploration slots (popular or random-from-affinity-set).
- Real-time vs batch: real-time is fresher but expensive. Batch personalization once per session is usually enough.

---

## Problem 4: Demand forecasting for retail

### Requirements

- Horizon: 4 to 12 weeks.
- Hierarchy: SKU by store by region. Tens of thousands of combinations.
- Update cadence: weekly refresh of forecast; daily update of actuals.
- Consumers: supply-chain, finance, merchandising. They need interpretable forecasts with confidence intervals.

### Metrics

- Offline: MAPE, WAPE, bias (mean error), quantile loss if probabilistic.
- Online: Forecast vs actual gap over past N weeks. Inventory cost saved. Stockout rate.
- Guardrail: Forecast bias on high-value SKUs, forecast stability (no wild swings week over week).

### Architecture

Hierarchical ensemble:

1. **Bottom-up base models:** One model per SKU-store for high-volume series (LightGBM with lagged features, calendar features, promotions). Simpler model (ARIMA, ETS) for low-volume.
2. **Top-down reconciliation:** Aggregate to regional or category level, reconcile via MinT or similar.
3. **Intermittent-demand handling:** Croston or TSB for sparse series.
4. **Promotions uplift:** Separate model for promotional lift, added to base.

### Data and features

- History: sales per SKU-store-day.
- Calendar: day of week, holidays, school calendar, weather.
- Promotions: planned promotional calendar.
- External: marketing spend, competitor presence.

### Training

- Weekly retrain across all hierarchy.
- Use temporal cross-validation (expanding window).
- Conformal prediction or quantile regression for intervals.

### Serving

- Forecast written to a table (batch serving). No real-time requirement.
- Dashboard (Looker or Streamlit) for supply-chain team.
- API for downstream ops systems.

### Monitoring

- Bias and WAPE by hierarchy level (overall, region, category, SKU).
- Prediction interval coverage (80 percent interval should catch actual 80 percent of the time).
- Data freshness (actuals ingested on schedule).

### Tradeoffs

- Per-SKU model vs global model: global model (like LightGBM trained across all SKUs with SKU-ID feature) scales better and learns cross-SKU patterns. Per-SKU is interpretable but rarely beats global on aggregate.
- Deep learning (N-BEATS, Temporal Fusion Transformer) can beat GBM but at significant infra cost; start with GBM.

---

## Problem 5: Customer-support chatbot (LLM-powered)

### Requirements

- Channels: web chat, WhatsApp.
- Languages: Spanish (primary), English.
- Latency: first token under 2 seconds, full response under 15 seconds.
- Must not hallucinate policy. Must escalate to humans when uncertain.

### Metrics

- Offline: Accuracy on a held-out set of real support questions (expert-rated). Faithfulness (does the response stay grounded in retrieved policy docs?).
- Online: Resolution rate (bot resolves vs escalates), CSAT post-conversation, escalation rate, median handle time.
- Guardrail: Hallucination rate (periodic audits), policy-violation rate, response time.

### Architecture

Retrieval-augmented generation:

1. **Retriever:** Embedding-based search over an index of policy docs, FAQ, product documentation. Returns top 5 to 10 relevant passages.
2. **Generator:** LLM (GPT-4o, Claude, or a local Llama 3.1 70B for cost control) with a strict prompt: "Answer only from the provided context; if not in context, say I do not know."
3. **Post-processing:** Safety classifier, profanity filter, link validation.
4. **Escalation:** If confidence is low (from a separate classifier on the prompt), route to a human agent with conversation history.

### Data

- Policy docs, FAQ, internal knowledge base (chunked, embedded, indexed).
- Conversation logs (for eval, not training without consent).

### Training / prompt engineering

- Retriever: embedding model (sentence-transformers or a custom fine-tune if accuracy lags).
- Generator: prompt engineering first. Fine-tune only if the base model consistently fails on a specific task.
- Evaluation set: 100 to 500 curated question-response pairs with expected policy references.

### Serving

- Streaming response (tokens as they are generated) for user experience.
- Caching of common questions.
- Rate limiting per user.
- Observability: log every prompt, retrieved context, response for audit.

### Monitoring

- Retrieval recall on evaluation set (is the right policy doc being retrieved?).
- Response faithfulness (does the response cite the context?).
- Escalation rate over time.
- Cost per conversation (token count, inference cost).

### Tradeoffs

- Hosted LLM vs local: hosted is easier and higher quality; local is cheaper at scale and has data-residency control.
- Fine-tuning vs prompting: prompting is faster to iterate; fine-tuning is needed for consistent format or domain-specific language.
- Single LLM call vs pipeline: a pipeline (classifier + retriever + generator + validator) is more controllable but higher latency.

---

## How to practice

1. Pick one problem, set a 45-minute timer, answer out loud.
2. Record yourself or pair with a peer.
3. Review: did you cover the full lifecycle? Did you state tradeoffs?
4. Read a senior engineer's writeup of the same problem (ML system design repos, engineering blogs from Uber, Netflix, Mercado Libre, Nu).

## Common mistakes in interviews

1. Jumping to a specific model before clarifying the problem.
2. Ignoring latency, cost, or data freshness constraints.
3. No monitoring section. Interviewer will ask.
4. Over-engineering. If a simple batch system works, propose it.
5. No fallback plan. "What if the model is down?" is always asked.
6. Ignoring human-in-the-loop cases (fraud review, support escalation).
