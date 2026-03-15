# Causal ML for Underwriting & Pricing

## Purpose

Move beyond prediction to causation. Standard ML models answer "what will happen?" but cannot answer "what would happen if we changed X?" -- and that second question is what actually drives business decisions. Causal ML provides the tools to estimate the effect of interventions: what happens to retention if we offer a 10% discount? Does this risk factor actually cause higher claims, or is it just correlated? Which marketing channel truly drives new policy sales vs. just capturing people who would have bought anyway? This is where actuarial science and modern ML intersect most naturally.

---

## What It Unlocks

- **Career paths**: Causal Inference ML Engineer, Pricing Data Scientist, Marketing Science Engineer, Experimentation Platform Engineer
- **Skills**: Treatment effect estimation, observational causal inference, experiment design and analysis, uplift modeling, counterfactual reasoning, sensitivity analysis for unobserved confounders
- **Domain leverage**: Actuarial pricing is fundamentally about isolating risk factors. Causal ML formalizes this with modern statistical tools. Your training in GLMs, credibility theory, and risk classification maps directly onto causal inference concepts.

---

## Key Technologies

| Technology | What It Does |
|---|---|
| **DoWhy** | Microsoft's causal inference library. Provides a principled 4-step workflow: model, identify, estimate, refute. Forces you to state assumptions explicitly. |
| **EconML** | Microsoft's library for heterogeneous treatment effect estimation. Implements causal forests, double ML, instrumental variables. Production-grade. |
| **CausalML** | Uber's library focused on uplift modeling and individual treatment effect estimation. Strong on marketing/pricing use cases. |
| **Causal Forests** | Non-parametric method for estimating heterogeneous treatment effects. Extension of random forests to causal problems. |
| **Double/Debiased ML (DML)** | Method that uses ML for nuisance parameter estimation while preserving valid causal inference. Best of both worlds. |
| **Instrumental Variables** | Technique for estimating causal effects when you have unobserved confounders but can find a "natural experiment." |
| **Synthetic Control** | Method for estimating causal effects of interventions on aggregate units (states, regions, market segments) using a weighted combination of control units. |
| **DAGs (Directed Acyclic Graphs)** | Visual tool for encoding causal assumptions. Foundation of the structural causal model framework. |

---

## Project Ideas

### 1. Discount Impact Analysis

**Scope**: Estimate the causal effect of premium discounts on policy retention, segmented by customer type. Answer: "If we offer a 15% renewal discount to segment X, how much does retention improve? Is the discount cost-effective given the lifetime value difference?"

**Dataset**: Use the Kaggle "Health Insurance Cross Sell Prediction" dataset or the "Insurance Company Benchmark" dataset as a base. Augment with synthetic discount assignment and renewal outcomes. Key: simulate realistic confounding -- customers who get discounts are typically those the company thinks are at risk of leaving, creating selection bias.

**Deliverables**:
- Causal DAG mapping the relationships between customer features, discount assignment, and retention outcome. Identify confounders, mediators, and colliders.
- Naive analysis showing why simple "discount group vs. no-discount group" comparison gives biased results
- Propensity score matching/weighting to estimate the Average Treatment Effect (ATE)
- Double ML (using EconML) for heterogeneous treatment effects: which customer segments benefit most from discounts?
- Causal forest for individual-level treatment effect estimation: personalized discount recommendations
- Sensitivity analysis (using DoWhy's refutation methods) to assess robustness to unobserved confounders
- Business recommendation: optimal discount targeting strategy with expected retention lift and ROI calculation
- Comparison against naive uplift model to show when causal rigor matters

**Complexity**: Medium-high. The causal modeling is the core challenge. Communicating results to business stakeholders requires clear visualization of treatment effects.

### 2. Risk Factor Decomposition for Auto Insurance Pricing

**Scope**: Separate causal risk factors from confounded correlations in auto insurance pricing. For example: does vehicle age directly cause higher claim frequency, or is it confounded by driver demographics? Does credit score causally predict claims, or does it proxy for unobserved socioeconomic factors that are legally protected?

**Dataset**: French Motor TPL dataset from OpenML (real insurance data with policy and claims information). Supplement with synthetic variables to create realistic confounding structures.

**Deliverables**:
- Causal DAG for auto insurance risk factors based on actuarial domain knowledge and data exploration
- Traditional GLM pricing model (Poisson for frequency, Gamma for severity) as baseline
- Causal analysis using DML to estimate the direct causal effect of each rating factor after controlling for confounders
- Comparison: which factors have large predictive power but small causal effect? These are proxy variables.
- Fairness analysis: use causal mediation analysis to decompose whether protected-class effects operate through legitimate (causal) vs. illegitimate (proxy) pathways
- Instrumental variable analysis for at least one factor where a natural instrument exists
- Policy recommendation: which rating factors should be included in a pricing model based on causal (not just predictive) validity?
- Write-up connecting causal analysis to actuarial pricing regulation (when regulators ask "is this factor related to risk?", they implicitly mean causally)

**Complexity**: High. Requires deep actuarial domain knowledge to construct credible causal DAGs. The fairness analysis adds legal and ethical complexity.

### 3. Marketing Attribution for Insurance

**Scope**: Measure the causal effect of each marketing channel (paid search, social media, email, agent referral, direct mail) on policy acquisition. Move beyond last-click attribution to true causal measurement.

**Dataset**: Synthetic marketing touchpoint data following realistic insurance purchase journeys. Model multi-touch attribution with realistic patterns: paid search captures demand, email nurtures, agent converts. Include seasonal effects and marketing budget variation as potential instruments.

**Deliverables**:
- Causal DAG for insurance marketing attribution, encoding assumptions about channel interactions and customer journeys
- Naive attribution models (last-click, first-click, linear) as baselines -- show their biases
- Observational causal analysis: DML estimating incremental effect of each channel controlling for customer intent signals
- Synthetic control analysis: estimate the effect of a "marketing blackout" in a region/segment using other regions as synthetic controls
- Geo-experiment design: propose a randomized geo-experiment to validate the observational estimates
- Budget optimization: given causal channel effects, recommend reallocation of a fixed marketing budget to maximize policy acquisitions
- Incrementality framework: for each channel, estimate what percentage of conversions are truly incremental (caused by the marketing) vs. organic
- Sensitivity analysis on the "no unobserved confounders" assumption

**Complexity**: High. Marketing attribution is a genuinely hard causal inference problem due to complex customer journeys and selection effects.

---

## Career Impact & Salary Context

- **Demand**: Growing rapidly. Causal inference teams are expanding at Uber, Netflix, Airbnb, Meta, Microsoft, Booking.com, and financial institutions. Insurance companies are beginning to adopt these methods for pricing fairness and marketing effectiveness.
- **Where to target**: Experimentation/causal inference teams at tech companies, pricing analytics at insurers and financial institutions, marketing science teams, economic consulting firms (Analysis Group, Cornerstone Research).
- **Salary range (US)**: $150K-$230K for causal inference ML engineers and applied scientists. Companies with dedicated causal inference teams (Netflix, Uber) pay at the top of this range.
- **Mexico/LATAM remote**: $60K-$110K USD for remote roles. Causal inference skills are rare in LATAM, making you highly differentiated.
- **Academic bridge**: Causal ML sits at the intersection of statistics, economics, and ML. Strong causal inference skills open doors to research scientist roles and PhD programs if that path interests you.

---

## How Your Actuarial Background Provides an Edge

- This is the specialization where your actuarial training provides the strongest advantage. Actuarial pricing has always been about isolating risk factors -- causal ML gives you rigorous modern tools for what actuaries do intuitively with GLMs and credibility.
- Your experience with GLMs (Poisson, Gamma, Tweedie) for pricing means you already understand the parametric side. Causal ML extends this with non-parametric methods and explicit causal assumptions.
- Credibility theory in actuarial science is conceptually related to treatment effect estimation -- weighting between individual and group experience.
- You understand insurance regulation around rating factors. When a regulator asks "is credit score related to risk?", they are asking a causal question. You can now answer it with rigorous methods.
- Actuarial reserving deals with censored data and time-varying effects -- concepts that appear in causal survival analysis.
- Your mathematical training (probability, statistics, stochastic processes) exceeds what most ML engineers bring to causal inference. You can engage with the theory, not just the library APIs.

---

## Prerequisites

- **Phase 1** (Python ML Foundations): Must be complete -- you need solid modeling fundamentals.
- **Phase 5** (Statistics & Experimentation): Strongly recommended -- A/B testing and statistical inference are the foundation for causal ML.
- Solid understanding of probability and statistics (you have this from actuarial training).
- Familiarity with GLMs (you have this from actuarial pricing).
- Basic understanding of potential outcomes framework or structural causal models (this will be learned during the specialization).

---

## Estimated Timeline

- **Weeks 1-2**: Causal inference fundamentals (potential outcomes, DAGs, identification, estimation). Read Brady Neal's Introduction to Causal Inference.
- **Weeks 3-4**: DoWhy and EconML walkthrough. Implement textbook examples (smoking/cancer, returns to education).
- **Weeks 5-7**: Project 1 (Discount Impact Analysis)
- **Weeks 8-10**: Project 2 (Risk Factor Decomposition)
- **Weeks 11-13**: Project 3 (Marketing Attribution) + synthetic control methods
- **Week 14**: Integration, portfolio write-up, connecting causal ML to your actuarial pricing knowledge
- **Total: 12-14 weeks** at 15-20 hours/week

---

## Resources

### Courses
- [Brady Neal: Introduction to Causal Inference](https://www.bradyneal.com/causal-inference-course) -- best free course on causal inference for ML practitioners
- [Stanford CS 7792: Counterfactual Machine Learning](https://cs7792.netlify.app/) -- advanced but excellent
- [Causal Inference for the Brave and True](https://matheusfacure.github.io/python-causality-handbook/) -- free online book with Python code, practical focus

### Papers
- "Double/Debiased Machine Learning for Treatment and Structural Parameters" (Chernozhukov et al., 2018) -- the DML paper
- "Estimation and Inference of Heterogeneous Treatment Effects using Random Forests" (Wager & Athey, 2018) -- causal forests
- "Using Synthetic Controls: Feasibility, Data Requirements, and Methodological Aspects" (Abadie, 2021) -- synthetic control overview
- "Metalearners for Estimating Heterogeneous Treatment Effects" (Kunzel et al., 2019) -- S-learner, T-learner, X-learner

### Repos & Tools
- [DoWhy](https://github.com/py-why/dowhy) -- causal inference library
- [EconML](https://github.com/py-why/EconML) -- heterogeneous treatment effects
- [CausalML](https://github.com/uber/causalml) -- uplift modeling
- [CausalNex](https://github.com/quantumblacklabs/causalnex) -- Bayesian network-based causal inference
- [dagitty](http://www.dagitty.net/) -- web tool for drawing and analyzing causal DAGs
