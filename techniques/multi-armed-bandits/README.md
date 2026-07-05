---
tags: [bandits, online-learning, decision-theory, experimentation]
status: draft
created: 2026-04-25
updated: 2026-04-25
---

# Multi-Armed Bandits

## What it is

A multi-armed bandit (MAB) is the simplest non-trivial sequential decision problem: at each round $t$, choose one of $K$ arms, observe a reward, and update beliefs. The goal is to maximize cumulative reward over $T$ rounds, balancing **exploration** (trying arms whose value is uncertain) against **exploitation** (pulling the arm currently believed best).

The setting captures the structure of A/B testing, dynamic pricing, ad selection, treatment assignment, content recommendation, and any situation where you must commit to actions one at a time and learn from the outcomes.

## Why it matters

- **Better than fixed A/B tests when traffic is finite.** A 50/50 split throws half of users at a known-bad variant for the entire experiment. A bandit allocates traffic to better-performing arms as evidence accumulates, reducing the cost of experimentation.
- **The decision-theoretic foundation for personalization.** Contextual bandits generalize MAB to incorporate features; this is the recommendation problem, the pricing problem, and the next-best-action problem in CRM.
- **A clean conceptual frame.** Many sloppy production systems are bandits in disguise (an ops team manually moving traffic between models is implementing a noisy human bandit). Naming the problem opens the door to better algorithms.

## The exploration-exploitation tradeoff

Greedy policies (always pull the empirically best arm) lock in early bad luck: an arm that produced two failures by coincidence is never tried again, even if it is actually the best. Pure exploration (uniform random) wastes pulls on known-bad arms.

The performance metric is **cumulative regret**:

$$
R(T) = T \cdot \mu^* - \sum_{t=1}^T \mu_{a_t}
$$

where $\mu^*$ is the best arm's mean reward and $a_t$ is the arm chosen at round $t$. A good bandit algorithm has regret that grows sub-linearly in $T$; the optimal rate is $O(\sqrt{T})$ for the worst case and $O(\log T)$ for nice (well-separated) instances.

## Epsilon-greedy

The simplest algorithm: with probability $\epsilon$ pick a uniformly random arm, otherwise pick the arm with highest empirical mean.

```
for t in 1..T:
    if uniform(0,1) < epsilon_t:
        a_t = uniform random arm
    else:
        a_t = argmax_a empirical_mean(a)
    pull a_t, observe reward, update mean
```

**Decay schedules.** A fixed $\epsilon$ leaves cumulative regret growing linearly. Standard fix: $\epsilon_t = \min(1, c K / t)$ for a constant $c$, which yields $O(\log T)$ regret on well-separated instances.

**Limitations.** Treats all sub-optimal arms identically; explores arms that are clearly bad as much as arms that are close-to-best.

## UCB (Upper Confidence Bound)

UCB embodies the **principle of optimism in the face of uncertainty**: pick the arm whose plausible upper bound on the mean is highest. The UCB1 algorithm (Auer-Cesa-Bianchi-Fischer 2002) chooses

$$
a_t = \arg\max_a \left( \hat{\mu}_a + \sqrt{\frac{2 \ln t}{n_a}} \right)
$$

where $\hat{\mu}_a$ is the empirical mean of arm $a$ and $n_a$ is the number of times it has been pulled. The bonus term shrinks for arms pulled often (low uncertainty) and grows logarithmically with $t$ (we get more confident as we collect data).

UCB1 has $O(K \log T / \Delta)$ regret on instances where the gap between best and second-best is $\Delta$. Empirically robust, no hyperparameters, the default frequentist baseline.

## Thompson sampling

Bayesian posterior sampling. Maintain a posterior over each arm's reward distribution. At each round, sample a value from each arm's posterior; pick the arm with the highest sample; observe and update.

For Bernoulli rewards (clicks, conversions) with a Beta prior, the update is closed form:

$$
\text{Beta}(\alpha_a, \beta_a) \xrightarrow{\text{success}} \text{Beta}(\alpha_a + 1, \beta_a)
$$

$$
\text{Beta}(\alpha_a, \beta_a) \xrightarrow{\text{failure}} \text{Beta}(\alpha_a, \beta_a + 1)
$$

Algorithm:

```python
alpha = [1] * K
beta  = [1] * K
for t in range(T):
    samples = [np.random.beta(alpha[a], beta[a]) for a in range(K)]
    a_t = int(np.argmax(samples))
    reward = pull(a_t)
    if reward:
        alpha[a_t] += 1
    else:
        beta[a_t] += 1
```

Thompson sampling has matching $O(K \log T / \Delta)$ regret to UCB but typically lower empirical regret because the random sampling explores adaptively rather than via a deterministic confidence bound. Chapelle and Li (2011) demonstrated this in a head-to-head benchmark on Yahoo News data; subsequent industrial deployments confirmed it.

## Contextual bandits

In real applications the reward depends on observable context $x_t$ (the user's features, the current time, the ad creative being shown). The arm chosen at round $t$ should depend on $x_t$.

- **LinUCB** (Li et al. 2010) assumes the reward of arm $a$ is linear in $x$: $r_{t,a} = x_t^\top \theta_a + \varepsilon$. Maintains a Bayesian linear regression per arm; the UCB rule includes a confidence bonus from the predictive variance.
- **Neural contextual bandits.** Replace the linear model with a neural network and approximate uncertainty via dropout, deep ensembles, or NTK-based confidence bounds (NeuralUCB, Zhou et al. 2020).
- **Logistic and Poisson contextual bandits.** Generalized linear contextual bandits (Filippi et al. 2010) for binary and count rewards.

The connection to personalization. A contextual bandit that picks among $K$ candidate ranking models or content variants given user features is **doing personalization with online learning**. Production recommender systems at Netflix, Spotify, and Yahoo use bandits at the candidate-selection layer for exactly this reason.

## Bandits vs A/B tests

| Property | Fixed A/B | Bandit |
|----------|-----------|--------|
| Assignment | 50/50 (or any fixed split) for the duration | Adaptive; better arms get more traffic |
| Stopping | Fixed sample size from power analysis | Continuous; converges as evidence accumulates |
| Regret | Maximal (half the traffic on the loser) | Sub-linear |
| Statistical guarantee | Clean, easy to explain to a regulator | Requires sequential-testing corrections |
| Best when | Two close-to-equal arms; need clean statistical claim | Many arms; cost of exploration matters; quick iteration |

Hybrid: run a short fixed A/B for the first $N$ users to establish baselines and assignment quality, then switch to a bandit. This is the pattern most experimentation platforms (Optimizely, Iterable, Eppo) actually implement under the hood.

## Bandits for dynamic pricing

Pricing is naturally a bandit problem: each price point is an arm, the reward is revenue per visitor (price times conversion), and the seller learns by observing demand at the prices it offers.

Frame:

- Arms: a discrete grid of candidate prices.
- Reward: $r_t = p_{a_t} \cdot \mathbb{1}[\text{purchased at price } p_{a_t}]$.
- Posterior: Beta on the per-price conversion rate; Thompson sample, then compute expected revenue.

Worked example: simulate a logistic demand curve and run Thompson sampling.

```python
import numpy as np

rng = np.random.default_rng(0)
prices = np.array([9, 12, 15, 18, 21, 24])
true_p_conv = 1 / (1 + np.exp((prices - 17) / 3))   # conversion drops as price rises
T = 5000

alpha = np.ones_like(prices, dtype=float)
beta  = np.ones_like(prices, dtype=float)
chosen = np.zeros(T, dtype=int)
revenue = np.zeros(T)

for t in range(T):
    sampled_conv = rng.beta(alpha, beta)
    expected_rev = prices * sampled_conv
    a = int(np.argmax(expected_rev))
    converted = rng.uniform() < true_p_conv[a]
    reward = prices[a] * converted
    if converted:
        alpha[a] += 1
    else:
        beta[a]  += 1
    chosen[t] = a
    revenue[t] = reward

# Inspect: late-trial allocation should concentrate on the revenue-optimal price
late_allocation = np.bincount(chosen[-1000:], minlength=len(prices)) / 1000
print(dict(zip(prices, np.round(late_allocation, 3))))
```

The bandit converges to the revenue-optimal price (typically not the lowest price that maximizes conversion; not the highest price that maximizes margin). It does so without ever running a formal A/B test and without specifying the demand model; the price-revenue tradeoff is learned directly.

## Common gotchas

1. **Non-stationarity.** Vanilla bandits assume reward distributions are constant. Real environments drift (user tastes change, seasonality, competitive pricing). Use sliding-window UCB, discounted Thompson sampling, or change-point detection on top.
2. **Delayed feedback.** Conversion rewards arrive minutes to days after the action. Maintain pending-pull queues and update posteriors when feedback arrives; do not pause learning.
3. **Off-policy correction for offline replay.** Production logs are biased by the policy that generated them. To replay a candidate policy on logs, weight by inverse propensity (see `docs/theory/12_recommender_system_foundations.md` off-policy section).
4. **Cold-start.** Beta(1,1) priors mean the first few pulls per arm are uninformative; bandits over-explore initially. Warm-start with informative priors derived from offline analysis.
5. **Multi-objective rewards.** When the reward is a weighted sum of click, conversion, and dwell time, the weights are policy decisions, not data. Document them.

## Resources

- `docs/theory/13_mlops_architecture_patterns.md` for bandit deployment patterns.
- `docs/theory/08_experimentation_causal_primer.md` for the experimentation context.
- `techniques/reinforcement-learning/README.md` for the broader RL framework that subsumes bandits.

## References

- Sutton, Barto, *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press 2018, chapter 2.
- Auer, Cesa-Bianchi, Fischer, "Finite-time Analysis of the Multiarmed Bandit Problem", Machine Learning 2002.
- Thompson, "On the likelihood that one unknown probability exceeds another in view of the evidence of two samples", Biometrika 1933.
- Chapelle, Li, "An Empirical Evaluation of Thompson Sampling", NeurIPS 2011.
- Li, Chu, Langford, Schapire, "A Contextual-Bandit Approach to Personalized News Article Recommendation" (LinUCB), WWW 2010.
- Russo et al., "A Tutorial on Thompson Sampling", Foundations and Trends in Machine Learning 2018.
- Lattimore, Szepesvari, *Bandit Algorithms*, Cambridge University Press 2020 (free online).
