# Recommender System Foundations

## The problem

Given a set of users, a set of items, and a history of interactions, predict what each user will want next. Recommenders drive engagement at Netflix, Spotify, Amazon, Mercado Libre, TikTok.

Types of feedback:

- **Explicit.** Ratings, thumbs up/down, reviews.
- **Implicit.** Clicks, views, dwell time, purchases. Far more abundant but noisier.

## Families of recommenders

### 1. Popularity

Recommend the most popular items. Baseline. Useful for cold-start users. Diverse enough only if you respect segments.

### 2. Content-based

Use item features (text, metadata, images) to find similar items to what the user liked. Embed items; measure similarity.

Strengths: no cold-start for new items (once you can embed them).
Weaknesses: filter bubble; limited diversity.

### 3. Collaborative filtering

Use the interaction matrix $R$ (users x items) without item content. Two classic approaches:

- **User-based CF.** Find users similar to you; recommend what they liked.
- **Item-based CF.** Find items similar to ones you liked (similarity from co-occurrence, not content).

Scales poorly with large users/items sets without factorization.

### 4. Matrix factorization

Decompose $R \approx U V^\top$ with latent factors. User $u$ is a vector $U_u \in \mathbb{R}^k$; item $i$ is $V_i \in \mathbb{R}^k$. Predicted rating is $\hat{R}_{ui} = U_u^\top V_i$.

Implicit feedback variants: Bayesian Personalized Ranking (BPR), Alternating Least Squares (ALS), Weighted Regularized Matrix Factorization (WRMF).

Library: `implicit` (Python), `surprise`, `LightFM` (supports hybrid content + collaborative).

### 5. Two-tower / neural CF

User tower encodes user features into an embedding; item tower encodes item features into an embedding. Score is dot product (or cosine similarity). Trained end-to-end on (user, item, label) triples.

Advantages: scales to production (serve via approximate nearest neighbor); incorporates content features; handles cold-start via features.

Disadvantages: cannot model complex user-item interactions beyond dot product (extensions like DIN, DIEN, cross-networks address this).

### 6. Sequential recommenders

Model the user's interaction sequence. GRU4Rec, SASRec, BERT4Rec. Adapt ideas from language modeling to item sequences.

Strong for sessions (what to recommend next given what was just clicked).

### 7. Graph-based recommenders

Build a user-item bipartite graph (or extended social graph). Use GNN layers (GraphSAGE, PinSage) to learn node embeddings.

Effective when relationships between items matter (music co-listening, product complements).

### 8. LLM-era recommenders

Prompt or fine-tune LLMs to recommend. Still early; strong for explanations and cold-start but computationally heavy.

## ALS update rule

For implicit feedback, the standard objective is weighted MSE with confidence weights $c_{ui} = 1 + \alpha r_{ui}$:

$$
\min_{U, V} \sum_{u, i} c_{ui} (p_{ui} - U_u^\top V_i)^2 + \lambda \left( \sum_u \|U_u\|^2 + \sum_i \|V_i\|^2 \right)
$$

where $p_{ui} = 1$ if user $u$ interacted with item $i$ and $0$ otherwise. **Alternating Least Squares** fixes one factor matrix and solves the other in closed form, then alternates:

$$
U_u = (V^\top C_u V + \lambda I)^{-1} V^\top C_u p_u
$$

$$
V_i = (U^\top C_i U + \lambda I)^{-1} U^\top C_i p_i
$$

where $C_u$ is the diagonal matrix of confidences for user $u$ over all items, and $p_u$ is the indicator vector of interactions. Each subproblem is a regularized least squares; the closed form makes each step deterministic and the alternating structure makes it embarrassingly parallel: every user's update is independent of every other user's, and likewise for items. ALS is the production default for million-scale matrix factorization in `implicit` and Spark MLlib.

## BPR pairwise loss

For implicit feedback the natural framing is ranking: for a user $u$ who interacted with item $i$ but not item $j$, the score for $i$ should exceed the score for $j$. **Bayesian Personalized Ranking** (Rendle et al. 2009) optimizes the log-likelihood that this ordering holds:

$$
\mathcal{L}_{\text{BPR}} = \sum_{(u, i, j) \in D_S} \log \sigma(\hat{x}_{ui} - \hat{x}_{uj}) - \lambda \|\Theta\|^2
$$

where $D_S$ is the set of triples with $i$ observed and $j$ unobserved, $\hat{x}_{ui}$ is the model score, and $\sigma$ is the logistic function.

Why pairwise beats pointwise for implicit feedback. Pointwise losses (binary cross-entropy treating non-clicks as zero) require treating every unobserved pair as a hard negative, which is a misspecification: most unobserved pairs are simply unseen, not disliked. Pairwise losses only require relative ordering on observed-vs-unobserved pairs, which is the actual signal in implicit data. Empirically BPR and its variants beat pointwise MSE/BCE on ranking metrics (NDCG, Recall@K) by wide margins.

## Negative sampling strategies

The triple $(u, i, j)$ in BPR requires sampling negatives $j$. The choice is a major production differentiator.

- **Uniform random.** Sample $j$ uniformly from items the user has not interacted with. Simple, low signal: most random negatives are easy and the model learns slowly.
- **Popularity-weighted.** Sample $j$ with probability proportional to its overall popularity (or popularity to a fractional power, e.g., $\hat{p}(j)^{0.75}$, the word2vec trick). Forces the model to distinguish the user's true preference from generally popular items.
- **In-batch negatives.** In a two-tower model trained on a batch of $B$ positive pairs, the other $B - 1$ items in the batch act as negatives for each user. Free, popularity-biased (heavily sampled items appear more often as negatives), and the standard for two-tower retrieval models.
- **Hard negatives.** Items that the model currently ranks high but the user did not interact with. Implementation: periodically score candidate negatives with the current model, keep the top-K hardest. Hard negatives produce the steepest gradients and the largest accuracy gains in production. They also risk false negatives (items the user would actually like but has not seen yet).
- **Mixed strategies.** Mix uniform, popularity-weighted, and hard at a fixed ratio (e.g., 50/30/20). The 2026 production default for two-tower retrieval at scale.

The negative-sampling strategy often matters more than the model architecture: switching from uniform to mixed hard-negative sampling on a fixed two-tower commonly delivers a larger Recall@10 gain than swapping the architecture for a fancier one.

## Calibration of recommendations

A scoring model produces a number in $[0, 1]$ that the business interprets as predicted CTR. If the model is uncalibrated, this interpretation is wrong: predicted 0.10 might correspond to observed 0.06 or 0.20 in production. Two consequences:

- **Bidding and pricing break.** Ad bidders that bid as $\text{predicted CTR} \times \text{value per click}$ over-bid or under-bid in proportion to the calibration error.
- **Mixed-objective ranking breaks.** When the ranker combines predicted CTR with predicted conversion rate or revenue, miscalibrated probabilities mean the weighted sum optimizes the wrong tradeoff.

The standard production fix is **isotonic recalibration**: collect (predicted score, observed outcome) pairs from a holdout or recent production traffic, fit a non-decreasing piecewise-constant calibrator, apply at inference. Refresh weekly; drift in calibration is a leading indicator of model decay before raw accuracy drops. See `techniques/model-calibration` for the full treatment.

## Off-policy evaluation

Online A/B tests are slow and risky. Off-policy evaluation estimates how a candidate ranking policy $\pi$ would have performed using logs collected under a different policy $\pi_0$.

- **Inverse propensity scoring (IPS).** For each logged event, weight the observed reward by $\pi(a \mid x) / \pi_0(a \mid x)$. The expected weighted reward is an unbiased estimate of the candidate policy's value, provided every action under $\pi$ has positive probability under $\pi_0$ (overlap). High variance when propensities are small.
- **Doubly-robust (DR) estimators.** Combine an IPS term with a learned reward model: the reward model handles cases where IPS is high-variance, and IPS corrects model bias. Unbiased if either the reward model is correct or the propensities are correct. Far lower variance in practice.

Use cases: comparing 10 candidate ranking policies offline before promoting one or two to A/B; counterfactual debugging ("what would the system have done if we had not down-ranked these items"). Logging propensities at serving time is mandatory for any of this to work; bake it into the serving stack.

## Architecture in production

Two-stage:

1. **Candidate generation (retrieval).** From millions of items, produce 500 to 1,000 candidates. Methods: two-tower + ANN, collaborative filtering, popularity-by-segment.
2. **Ranking.** Score the candidates with a more expensive model. GBM or neural with rich features. Return top K.
3. **Re-ranking.** Apply business rules, diversity constraints, deduplication, freshness.

## Embeddings and approximate nearest neighbor (ANN)

To serve retrieval, pre-compute item embeddings and index them in a vector database. At request time, compute the user embedding, query the ANN index.

Libraries: FAISS (CPU/GPU), Milvus, Pinecone, Chroma, pgvector.

Common indexes: HNSW (graph-based, high recall), IVF (inverted file), PQ (product quantization, memory-efficient).

## Offline evaluation

Split time-ordered interactions into train and test. Standard metrics:

- **Recall@K.** Of the items the user interacted with in test, how many are in the top K recommendations?
- **NDCG@K.** Discounted cumulative gain normalized by ideal ordering.
- **MAP@K.** Mean average precision.
- **Hit rate.** Binary: did the top K contain any test interaction.
- **Coverage.** What fraction of the catalog is ever recommended.
- **Diversity.** Intra-list diversity (how different are items within a list).
- **Novelty.** How rare are recommended items (inverse popularity).

Leave-one-last-out is the canonical split for offline eval: for each user, the most recent interaction is in test, rest is train.

## Online evaluation

Offline metrics correlate loosely with online performance. Always A/B test.

Online metrics: CTR, conversion rate, revenue per session, session length, return visits, long-term retention.

## Cold start

- **New user.** No history. Use popularity, segment-based defaults, explicit preference elicitation (onboarding questions), or content-based using signup features.
- **New item.** No interactions. Use content embeddings (text, image) via a content model; once enough interactions accumulate, switch to CF.
- **New platform.** No users or items yet. Bootstrap from external signals (editorial curation, popularity from similar platforms).

## Serving considerations

- **Latency.** Retrieval + ranking under 200 ms.
- **Freshness.** Items added in the last minute may not be in the index yet. Hybrid retrieval (ANN + recency) helps.
- **Caching.** User embeddings cached for a session. Static top-K per segment as a fallback.
- **Fallback.** If the personalizer fails, serve popularity-by-segment.

## Exploration vs exploitation

Pure exploitation leads to feedback loops: the model only sees items it already liked, so it never learns about others.

Options:

- **Epsilon-greedy.** With probability epsilon, show a random item.
- **Thompson sampling.** Bayesian bandit approach.
- **Contextual bandit.** Model the context-dependent reward.
- **Explicit "Discovery" slots.** UI slots for exploration, separate from personalized slots.

## Diversity and responsible design

- **Echo chambers.** Pure personalization polarizes. Include diverse items.
- **Popularity bias.** Popular items get more recommended, more clicks, more recommended. Correct via inverse propensity weighting or explicit popularity reduction at re-ranking.
- **Fairness to producers.** In multi-sided marketplaces, recommending only a few sellers hurts others. Constrained re-ranking to ensure coverage.

## Common mistakes

1. Ignoring the implicit-feedback challenge. All non-clicks are not negatives.
2. Random split on time-ordered data (time leakage).
3. Optimizing offline metrics that do not correlate with online behavior.
4. No diversity measurement. The system can be highly accurate and terribly boring.
5. Coarse cold-start (always recommend popular) when you have signup features you could use.
6. Overlooking position bias. CTR at position 1 is higher than position 10 even for identical items. Control via IPW or counterfactual logging.

## Linking to the portfolio

- `recsys-movielens`: implements two-tower with `implicit` and `LightFM` baselines on MovieLens 25M.
- `technique/recommender-systems`: companion deep dive.
- `technique/entity-embeddings`: the embedding techniques used inside recommenders.

## References

- Koren, Bell, Volinsky, "Matrix Factorization Techniques for Recommender Systems", IEEE 2009.
- Hu, Koren, Volinsky, "Collaborative Filtering for Implicit Feedback Datasets", 2008.
- Rendle et al., "BPR: Bayesian Personalized Ranking from Implicit Feedback", 2009.
- He et al., "Neural Collaborative Filtering", 2017.
- Kang, McAuley, "Self-Attentive Sequential Recommendation" (SASRec), 2018.
- Ying et al., "Graph Convolutional Neural Networks for Web-Scale Recommender Systems" (PinSage), 2018.
- Covington, Adams, Sargin, "Deep Neural Networks for YouTube Recommendations", 2016.

## Exercises

1. Write the ALS update equation for user factors given fixed item factors.
2. Why does BPR outperform MSE on implicit-feedback data?
3. Your two-tower model trains with in-batch negatives at batch size 256. How many negatives does each positive example see, and what is one limitation of this approach?
4. The predicted CTR of your model for slot 1 is $0.35$ but the observed CTR is $0.21$. What does this indicate, and how do you fix it?
5. Explain why position bias is a problem in recommender systems and name one debiasing technique.
- Dudik, Langford, Li, "Doubly Robust Policy Evaluation and Learning", ICML 2011.
- Swaminathan, Joachims, "Counterfactual Risk Minimization: Learning from Logged Bandit Feedback", JMLR 2015.
