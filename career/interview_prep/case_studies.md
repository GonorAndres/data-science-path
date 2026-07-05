# Business Case Studies for DS Interviews

Product DS and growth roles at Mercado Libre, Nu, Bitso, Kavak, Clip, Rappi, and consulting firms commonly ask open-ended business cases. The interviewer wants to see: problem decomposition, metric selection, hypothesis generation, and practical tradeoffs.

This file has eight cases with a worked-through framework for each. Read the question, give yourself 5 minutes to think, then compare to the reference answer.

---

## Framework (use on every case)

1. **Clarify the goal.** Before jumping in, confirm what "success" looks like in metrics. Do not assume.
2. **Decompose the problem.** Break into sub-questions. Which part has the most leverage?
3. **State assumptions explicitly.** "I will assume the business cares about 30-day retention, not same-day."
4. **Propose an approach.** Outline the analysis, model, or experiment you would run.
5. **Discuss tradeoffs.** What could go wrong? What would falsify the conclusion?
6. **Tie back to the business.** End with "and here is what I would recommend the team do next."

Never skip step 1. Interviewers test whether you stop to ask clarifying questions.

---

## Case 1: Product launch metric design

**Scenario:** Your product manager at a fintech app is launching a new feature next week: users can now set up recurring transfers. What metrics would you track and how would you tell if the feature is successful?

**Reference approach:**

- **Clarify:** What is the goal? User engagement? Reducing support load? Revenue growth via stickiness? Assume "increase long-term retention and transaction volume".
- **Primary metrics:** Share of users who set up a recurring transfer within 7 days of first seeing the feature. Among those, share still active after 28 days.
- **Secondary metrics:** Average transfer amount, total transfer count uplift, support ticket volume around the feature.
- **Guardrail metrics:** Day-1 uninstall rate, NPS, failed transaction rate, churn from confused users.
- **Analysis plan:** A/B test at feature level if possible (show to 50 percent cohort). If not, before-after comparison with a non-treated control cohort. Measure 28 days post-launch.
- **Tradeoffs:** Recurring transfers may cannibalize one-off transfers (total volume flat). Feature may be used more by already-active users (selection bias). Control for pre-period activity.
- **Recommendation:** Ship to 50 percent for 28 days, then decide on global rollout based on primary + guardrail metrics.

---

## Case 2: Fraud detection model in production

**Scenario:** The fraud team deployed a model last quarter. Offline AUC was 0.92; in production, fraud rate is flat. What investigate?

**Reference approach:**

- **Clarify:** What is the production metric they are measuring? "Flat fraud rate" could mean precision, recall, dollars saved, or total fraud events.
- **Hypothesis tree:**
  - **Data distribution shift.** Was the training data representative of production? Has the mix of transaction types changed? Check feature distributions.
  - **Label leakage.** Offline labels may have been resolved cases (includes future adjudication). Production operates in real time.
  - **Threshold not tuned for ops.** AUC is threshold-agnostic. If ops team uses the wrong threshold, precision or recall will not match offline.
  - **Treatment dilution.** The model only flags; human reviewers make the decision. If queues are overloaded or reviewers override, model signal is lost.
  - **Novelty / adversarial.** Fraudsters adapt. If your training set is 6 months old, new fraud patterns are not covered.
- **What to actually do:** Export production scores vs outcomes for the last 30 days. Build a confusion matrix. Compute precision and recall at the current threshold. Compare to offline on a held-out set from the same period. Visualize feature distributions train vs prod.
- **Likely findings:** Distribution shift plus queue bottleneck is the most common combination.

---

## Case 3: Retention curve dropped

**Scenario:** 30-day retention dropped from 40 percent to 32 percent over the last two months. What do you investigate?

**Reference approach:**

- **Clarify:** Retention of what? All users? New signups only? A specific segment?
- **Slice:** By signup cohort (new users only, to isolate the effect). By channel (paid vs organic). By country or platform (iOS vs Android). Segments often reveal that the drop is driven by one slice dragging the average.
- **Correlate with changes:** What launched in that window? New onboarding flow? Price change? Platform update? Marketing campaign shift?
- **External factors:** Seasonality (spring break, end of academic cycle, Easter), macro events, competitor launches.
- **Deep dive:** Pick the slice with the largest retention drop. Look at funnel: do users complete signup? Do they complete first meaningful action? Do they return on day 1, day 7?
- **Recommendation:** If one channel drives it, investigate channel-specific onboarding. If it is uniform, suspect a platform regression or macro.

---

## Case 4: Pricing optimization

**Scenario:** An e-commerce company asks you to recommend optimal prices for 10,000 SKUs. Walk through your approach.

**Reference approach:**

- **Clarify:** Is the goal revenue, margin, or units sold? Are there constraints (price floors, competitor parity, brand rules)?
- **Available data:** Historical prices and sales per SKU, competitor prices (if scraped), costs, seasonality, inventory.
- **Simple baseline:** Estimate price elasticity per SKU from historical variance. `elasticity = (% change in quantity) / (% change in price)`. If the SKU has been priced at multiple levels, you have signal.
- **Segment by data quality:** Top sellers with enough historical price variance can be modeled individually. Long-tail SKUs need hierarchical models (share strength across similar SKUs via category or brand).
- **Model options:** Bayesian hierarchical regression, uplift modeling on price-test data, reinforcement learning (only if you have the volume to explore).
- **Practical constraint:** You cannot freely move prices; marketing and brand will object. Propose a pilot: 5 percent of SKUs, 90-day test, measure lift.
- **Recommendation:** Start with elasticity estimation for top-500 SKUs, pilot pricing changes, scale to full catalog after one iteration.

---

## Case 5: Recommender system lift measurement

**Scenario:** The team shipped a new recommender two weeks ago. How do you know if it is better than the previous one?

**Reference approach:**

- **Design:** Ideal is an A/B test where half the users get the new recommender. If that was done, use it.
- **If no A/B:** Compare on held-out historical data (offline metrics: NDCG, Recall@K, diversity, coverage). Then monitor production KPIs: CTR on recommended items, downstream conversion, time to purchase, revenue per visit.
- **Caveat:** Offline metrics can improve while online metrics do not. The interviewer likely wants to hear: "Offline is a proxy; a proper online A/B is the only ground truth."
- **Confounding:** Novelty effect (new recs appear fresh; CTR bumps) can fade. Measure for at least 4 weeks.
- **Recommendation:** Propose an A/B test going forward, even if the system is already live. Backfill via holdout users if possible.

---

## Case 6: Diagnose a bad forecast

**Scenario:** The demand forecast for Category X has been consistently 20 percent too low for three months. What do you do?

**Reference approach:**

- **First:** Validate the complaint. Is the error genuinely biased low, or is it high variance with no bias? Compute WAPE and MAPE broken down by week.
- **Hypotheses:**
  - **Structural shift.** Demand for Category X grew (new trend, promotion effect). Model does not learn fast enough.
  - **Missing covariates.** If the model does not include marketing spend, and spend increased, the model is miscalibrated.
  - **Data pipeline bug.** Category X input features have a stale join somewhere.
  - **Model miscalibration on this segment.** Global model trained to minimize aggregate error; local error can be biased.
- **Quick fix:** Apply a calibration correction (bias shift) while investigating.
- **Root fix:** Retrain with more recent data, add missing covariates, validate pipeline.
- **Recommendation:** Short-term bias correction now, full investigation within two weeks, retrain with updated features.

---

## Case 7: Customer segmentation for a marketing team

**Scenario:** The marketing team wants customer segments to target campaigns. How do you approach it?

**Reference approach:**

- **Clarify:** What will the marketing team actually do with the segments? Email campaigns? In-app notifications? How many segments can they act on (usually 3 to 7, not 20)?
- **Approach 1 (descriptive):** RFM segmentation (recency, frequency, monetary). Simple, interpretable, easy for the business to use.
- **Approach 2 (machine learning):** k-means or Gaussian mixture on behavioral features (time on app, session count, categories browsed). Often produces less interpretable segments.
- **Approach 3 (supervised):** Predict customer lifetime value, then segment on the prediction.
- **Tradeoff:** ML segmentation often does not beat RFM on real business outcomes because the business team cannot translate latent dimensions into campaigns.
- **Recommendation:** Start with RFM + one behavioral axis (e.g., primary category). Validate segments have different conversion rates on a pilot campaign before scaling.

---

## Case 8: Model deployment disagreement

**Scenario:** You built a churn model. The product manager wants to use it to decide which users to contact with a retention campaign. Your offline F1 is 0.58. Should it ship?

**Reference approach:**

- **Clarify:** What is the PM's cost function? False positives (wasted campaign spend) vs false negatives (missed churners). What is the expected revenue from saving a churner? What is the campaign cost per user contacted?
- **Translate F1 to business:** At the current threshold, what is precision and recall? If precision is 0.30, then 70 percent of contacts are wasted spend. Estimate expected value per contact.
- **Decision framework:** If expected value per contact is positive, ship with the current threshold. If not, tune threshold to raise precision even at the cost of recall.
- **Alternative:** Ship as a ranker. Contact the top N users by predicted churn score where N matches the campaign budget. This avoids the threshold question.
- **Monitor:** Set up a control group that does not get contacted, so you can measure the true uplift from the campaign plus model.
- **Recommendation:** Ship as a ranker with a 10 percent control group, measure 30-day retention lift in treatment vs control, iterate.

---

## How to answer in the interview

- Talk through your thinking. Interviewers cannot read your mind; they can only hear what you say.
- Ask clarifying questions early. It is never wrong to confirm the goal.
- When you do not know, say so, then say what you would do to find out.
- Summarize at the end: "To recap, I would recommend X, because Y, with a check for Z."

## Common mistakes

1. Diving into a specific model before clarifying the problem.
2. Missing the business-cost translation. Technical metrics without a business tie-in feel detached.
3. Proposing an A/B test for everything. Sometimes it is not feasible; know the alternatives.
4. Over-engineering. If a simple baseline would answer the question in a week, propose it.
5. Ignoring guardrail metrics. The interviewer will ask "what could this break?"; preempt it.
