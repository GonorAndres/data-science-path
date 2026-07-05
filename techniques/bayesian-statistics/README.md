# Bayesian Statistics

## What it is

Bayesian statistics is a framework for updating your beliefs about the world as new evidence arrives. Instead of treating a parameter (like the probability of a customer churning) as a single fixed but unknown number, Bayesian methods represent it as a *distribution* -- a range of plausible values with associated probabilities. You start with a **prior** distribution (what you believed before seeing data), observe data, and then compute a **posterior** distribution (your updated belief). This "learning from data" loop is the engine behind Bayesian thinking, and it maps naturally to how actuaries and data scientists refine assumptions over time.

## Why it matters

For an actuarial or data science practitioner, Bayesian statistics matters for several concrete reasons:

- **Incorporating prior knowledge**: Actuaries frequently have expert judgment, industry tables, or regulatory guidance that should inform analysis. Bayesian methods give you a principled, transparent way to blend that knowledge with observed data, rather than ignoring it (as frequentist methods do) or inserting it informally.
- **Small-sample reasoning**: Insurance portfolios are often thin. A new product line may have only two years of claims. Bayesian methods gracefully handle small data by leaning more on the prior when data is scarce, then letting the data dominate as volume grows.
- **Intuitive uncertainty quantification**: A 95% credible interval means "there is a 95% probability the parameter lies in this range." That is what most stakeholders *think* a confidence interval means, but it is not. Bayesian intervals answer the question people actually ask.
- **Decision-making under uncertainty**: Bayesian methods plug directly into decision theory. You can compute the expected cost of each action given the full posterior distribution of the unknown parameter, rather than relying on point estimates.
- **Foundation for credibility theory**: The Buhlmann credibility formula -- a pillar of actuarial exam material -- is a special case of Bayesian updating. Understanding Bayesian statistics makes credibility theory feel natural rather than formulaic.

## When to use it

- You need to combine prior expert judgment with observed data (e.g., blending industry mortality tables with your company's limited experience).
- You are running an A/B test and want to monitor results continuously without inflating false-positive rates (Bayesian A/B testing).
- You have hierarchical or grouped data (policyholders within territories within states) and want to share information across groups.
- You need to quantify full parameter uncertainty, not just a point estimate, for risk management or reserve ranges.
- Your sample size is small and you want to borrow strength from external information.
- You want to make probabilistic statements like "there is an 87% probability that the new treatment is better."
- You are building a credibility model and want to understand what is happening under the hood.

## How it works (intuitive)

### The core idea: belief updating

Imagine you are an underwriter evaluating a new commercial fleet account. Before seeing any claims data, you have a prior belief about the expected loss ratio based on industry benchmarks. After one year of experience, you observe the account's actual losses. Bayesian statistics tells you exactly how to blend your prior belief and the observed data into an updated (posterior) belief.

The weight given to data vs. prior depends on two things:
1. **How confident your prior is** -- a vague prior lets the data speak; a strong prior resists change.
2. **How much data you have** -- more data shifts the posterior further from the prior toward the data.

This is exactly the intuition behind the actuarial credibility formula:

$$\text{Estimate} = Z \times \text{Observed} + (1 - Z) \times \text{Prior}$$

where $Z$ (the credibility factor) increases with more data.

### Step-by-step walkthrough

**Step 1: Choose a prior distribution.**
This encodes what you believe before seeing data. It can be:
- *Informative*: Based on industry data, expert opinion, or regulatory tables.
- *Weakly informative*: Constrains the parameter to a reasonable range but does not pin it down.
- *Non-informative / flat*: Expresses near-total ignorance.

**Step 2: Write down the likelihood.**
The likelihood is the probability of the observed data given a specific parameter value. This is the same quantity used in maximum likelihood estimation (MLE); Bayesian methods just combine it with the prior.

**Step 3: Apply Bayes' theorem to get the posterior.**

$$P(\theta \mid \text{data}) = \frac{P(\text{data} \mid \theta) \times P(\theta)}{P(\text{data})}$$

In words: Posterior is proportional to Likelihood times Prior.

The denominator $P(\text{data})$ is a normalizing constant that ensures the posterior integrates to 1. In practice, you often work with the proportional form:

$$P(\theta \mid \text{data}) \propto P(\text{data} \mid \theta) \times P(\theta)$$

**Step 4: Summarize the posterior.**
Once you have the posterior distribution, you can extract:
- The posterior mean (point estimate).
- Credible intervals (uncertainty ranges).
- Posterior predictive distributions (forecasts for new observations).

### An analogy: the courtroom

Think of Bayesian updating like a trial:
- **Prior**: The juror's initial impression based on general background knowledge.
- **Likelihood (evidence)**: Testimony and exhibits presented in court.
- **Posterior**: The juror's updated belief after considering the evidence.
- A strong prior (deep-seated belief) requires overwhelming evidence to shift. A weak prior shifts easily.

## Key concepts

### Bayes' theorem

$$P(\theta \mid D) = \frac{P(D \mid \theta) \cdot P(\theta)}{P(D)}$$

| Symbol | Meaning |
|--------|---------|
| $\theta$ | Parameter of interest (e.g., claim frequency) |
| $D$ | Observed data |
| $P(\theta)$ | Prior -- your belief about $\theta$ before seeing data |
| $P(D \mid \theta)$ | Likelihood -- probability of the data given $\theta$ |
| $P(\theta \mid D)$ | Posterior -- updated belief about $\theta$ after seeing data |
| $P(D)$ | Evidence (marginal likelihood) -- ensures the posterior integrates to 1 |

### Conjugate priors

A prior is *conjugate* to a likelihood if the resulting posterior belongs to the same distribution family as the prior. This gives closed-form solutions and makes hand calculations tractable.

**Beta-Binomial (for proportions):**

| Component | Distribution |
|-----------|-------------|
| Prior | $\theta \sim \text{Beta}(\alpha, \beta)$ |
| Likelihood | $X \mid \theta \sim \text{Binomial}(n, \theta)$ |
| Posterior | $\theta \mid X \sim \text{Beta}(\alpha + x, \beta + n - x)$ |

Interpretation: $\alpha$ and $\beta$ act as "pseudo-counts." If your prior is Beta(2, 8), it is as if you have already seen 2 successes and 8 failures before collecting data. Observing $x$ successes in $n$ trials simply adds to these counts.

**Normal-Normal (for means):**

| Component | Distribution |
|-----------|-------------|
| Prior | $\mu \sim N(\mu_0, \sigma_0^2)$ |
| Likelihood | $\bar{X} \mid \mu \sim N(\mu, \sigma^2 / n)$ |
| Posterior | $\mu \mid \bar{X} \sim N\!\left(\frac{\frac{\mu_0}{\sigma_0^2} + \frac{n\bar{X}}{\sigma^2}}{\frac{1}{\sigma_0^2} + \frac{n}{\sigma^2}},\; \frac{1}{\frac{1}{\sigma_0^2} + \frac{n}{\sigma^2}}\right)$ |

The posterior mean is a precision-weighted average of the prior mean and the sample mean. Precision = 1/variance. More data or smaller variance means higher precision and more weight.

**Gamma-Poisson (for rates):**

| Component | Distribution |
|-----------|-------------|
| Prior | $\lambda \sim \text{Gamma}(\alpha, \beta)$ |
| Likelihood | $X_i \mid \lambda \sim \text{Poisson}(\lambda)$ for $i=1,\ldots,n$ |
| Posterior | $\lambda \mid X \sim \text{Gamma}(\alpha + \sum x_i, \beta + n)$ |

This is directly useful for modeling claim counts.

### Credible intervals vs. confidence intervals

| Feature | 95% Credible Interval (Bayesian) | 95% Confidence Interval (Frequentist) |
|---------|----------------------------------|---------------------------------------|
| Interpretation | "There is a 95% probability that $\theta$ lies in this interval, given the data." | "If we repeated this experiment many times, 95% of the constructed intervals would contain the true $\theta$." |
| Fixed quantity | The interval is random before data is seen; after data, the probability statement is about $\theta$. | $\theta$ is fixed; the interval is random across hypothetical repetitions. |
| Depends on prior? | Yes | No |
| Intuitive? | Yes -- answers the question people actually ask | Frequently misinterpreted |

The two can coincide when a flat prior is used, but they answer fundamentally different questions.

### Bayesian A/B testing

In a frequentist A/B test, you fix a sample size, collect data, then compute a p-value. Peeking at results early inflates false positives.

In a Bayesian A/B test:
1. Place priors on the conversion rates of variants A and B: $\theta_A \sim \text{Beta}(\alpha_A, \beta_A)$, $\theta_B \sim \text{Beta}(\alpha_B, \beta_B)$.
2. After observing conversions and non-conversions, compute posteriors.
3. Estimate $P(\theta_B > \theta_A \mid \text{data})$ via simulation: draw samples from both posteriors and count how often the B sample exceeds A.
4. You can check this probability at any time without a multiple-testing penalty.

This approach naturally quantifies "how confident are we that B is better?" rather than "can we reject the null that they are equal?"

### MCMC intuition

When conjugate priors are not available, the posterior does not have a closed form. Markov Chain Monte Carlo (MCMC) methods approximate the posterior by generating a large number of samples from it.

**Analogy**: Imagine you are lost in a hilly landscape in thick fog. You cannot see the full terrain (the posterior), but at each step you can measure the altitude where you stand. MCMC algorithms (like the Metropolis-Hastings algorithm or the No-U-Turn Sampler) propose random steps. If a proposed step goes uphill (higher posterior density), you always accept it. If it goes downhill, you sometimes accept it (with probability proportional to the density ratio). Over thousands of steps, the places you visit most frequently trace out the shape of the posterior distribution.

Key MCMC concepts:
- **Chains**: Multiple independent walks to check consistency.
- **Burn-in / warm-up**: Initial samples discarded because the chain has not yet reached the high-density region.
- **Convergence diagnostics**: R-hat statistic (should be close to 1.0), trace plots (should look like "fuzzy caterpillars," not trending).
- **Effective sample size (ESS)**: Accounts for autocorrelation between samples. Higher is better.

## MCMC in depth

When the posterior has no closed form, the standard tool is Markov Chain Monte Carlo. Three algorithms cover most practical cases.

### Metropolis-Hastings

Metropolis-Hastings (MH) is the prototype MCMC algorithm. To draw samples from a target $\pi(\theta) \propto p(D \mid \theta) p(\theta)$:

```
Initialize theta_0
For t = 1, 2, ..., N:
    Propose theta* ~ q(theta* | theta_{t-1})
    Compute acceptance ratio
        alpha = min(1, pi(theta*) q(theta_{t-1} | theta*) /
                       pi(theta_{t-1}) q(theta* | theta_{t-1}))
    Draw u ~ Uniform(0, 1)
    If u < alpha:
        theta_t = theta*
    Else:
        theta_t = theta_{t-1}
```

When $q$ is symmetric (random walk MH) the proposal terms cancel and the rule simplifies to "accept with probability $\min(1, \pi(\theta^*)/\pi(\theta_{t-1}))$". Tuning the proposal scale is the operational challenge: too small means high acceptance but slow exploration; too large means most proposals rejected.

### Gibbs sampling

When the conditional posterior of each parameter given the others has a closed form, Gibbs sampling cycles through parameters and samples each from its conditional in turn:

```
For t = 1, 2, ..., N:
    For each parameter j:
        theta_j^{(t)} ~ p(theta_j | theta_{-j}^{(t-1)}, D)
```

Gibbs is exact (no acceptance step) when the conditionals are tractable. The classical setting is conjugate hierarchical models: each variance, each random effect, each fixed effect has a known conditional and Gibbs cycles through them. Without conjugacy, fall back to Metropolis-within-Gibbs.

### NUTS (No-U-Turn Sampler)

NUTS (Hoffman, Gelman 2014) is the sampler used by Stan, PyMC, and NumPyro for continuous-parameter models. It is a self-tuning variant of **Hamiltonian Monte Carlo (HMC)**: the parameter is augmented with a momentum, simulated forward in time using the gradient of the log posterior, and the trajectory provides a globally informed proposal.

Why it dominates for high-dimensional continuous models:

- Gradient information lets the sampler take long, well-targeted moves; random-walk MH gets stuck in funnels.
- NUTS automatically chooses the trajectory length, removing the most fragile HMC hyperparameter.
- Acceptance rates near 0.8 with effective sample sizes per second 10x to 1000x random-walk MH on realistic models.

NUTS requires log-posterior gradients, so models must be implemented in a framework that supports automatic differentiation (Stan, PyMC v5 on PyTensor, NumPyro on JAX).

### Convergence diagnostics

A single chain can look fine and still be wrong. Run **at least four chains** from over-dispersed starting points and check:

- **R-hat ($\hat{R}$).** Compares between-chain to within-chain variance. Should be below 1.01 for every parameter; values above 1.05 indicate the chains have not mixed.
- **Effective sample size (ESS).** Accounts for autocorrelation. Total ESS should be at least 400 for stable inference; bulk ESS and tail ESS (Vehtari et al. 2021) report different parts of the distribution.
- **Trace plots.** Visual: should look like a "fuzzy caterpillar" with no trend. Trends indicate non-convergence.
- **Divergent transitions** (HMC/NUTS specific). Each divergence is a step where the sampler failed to integrate the trajectory accurately, typically in pathological geometry (the funnel of a hierarchical model). More than a handful of divergences invalidates the run; reparameterize.

ArviZ provides all of these via `az.summary` and `az.plot_trace`.

## Variational inference

When MCMC is too slow for the application (real-time updating, very high-dimensional models, scaling to streaming data), variational inference (VI) approximates the posterior with a tractable family $q_\phi(\theta)$ and minimizes the KL divergence to the true posterior. Equivalently, VI maximizes the **evidence lower bound (ELBO)**:

$$
\text{ELBO}(\phi) = \mathbb{E}_{q_\phi}\!\left[ \log p(D, \theta) - \log q_\phi(\theta) \right]
$$

The first term is the expected joint log-density (data fit + prior); the second is the entropy of the variational approximation. The ELBO equals $\log p(D)$ minus the KL, so maximizing the ELBO is equivalent to minimizing the KL.

**Mean-field approximation.** Assume $q_\phi(\theta) = \prod_j q_{\phi_j}(\theta_j)$ factorizes across parameters. Cheap, fast, and dramatically underestimates posterior covariance: parameters that are correlated in the true posterior come out independent in the approximation, and the marginal variances are biased low.

**Full-rank Gaussian VI.** $q_\phi(\theta) = \mathcal{N}(\mu_\phi, \Sigma_\phi)$ with a learned full covariance. Captures correlations; quadratic in dimension.

VI is faster than MCMC by 10x to 1000x but approximate; the approximation error is usually invisible in posterior means but biases posterior variances and tails. Use VI for:

- Model exploration (rapid iteration on prior or likelihood specifications).
- Scaling to large data (`pm.fit(method="advi")` in PyMC, SVI in NumPyro).
- Initialization of MCMC for hard models.

For final inference on consequential decisions prefer MCMC.

## Hierarchical models and partial pooling

A hierarchical model places a prior over groups whose hyperparameters are themselves estimated from data. The simplest example is many policyholder cohorts sharing a common population mean and variance:

$$
\mu \sim \mathcal{N}(0, 10), \quad \tau \sim \text{HalfCauchy}(2.5)
$$

$$
\theta_g \sim \mathcal{N}(\mu, \tau^2) \quad \text{for each group } g
$$

$$
y_{g, i} \sim \mathcal{N}(\theta_g, \sigma^2) \quad \text{for observations } i \text{ in group } g
$$

The posterior of $\theta_g$ shrinks toward the population mean $\mu$ by an amount that depends on (a) how much data group $g$ has and (b) how much between-group variance $\tau^2$ the data supports. This is **partial pooling**: groups with little data borrow strength from the population; groups with abundant data dominate their own posterior.

The connection to **Buhlmann credibility** is direct: the credibility factor $Z = n / (n + k)$ is the partial-pooling weight in the conjugate normal-normal hierarchical model with $k = \sigma^2 / \tau^2$. Buhlmann credibility is Bayesian hierarchical modeling under the conjugate normal special case; hierarchical Bayesian models generalize it to any likelihood (Poisson claim counts, Tweedie pure premium, Bernoulli lapse) with a prior structure that maps to actuarial intuition.

Nested random effects (drivers within agents within branches within regions) extend the same machinery to multi-level groupings. The same posterior shrinkage occurs at every level; it is the right framework for a portfolio of portfolios.

## Code: PyMC Bayesian GLM for claim frequency

A claim-frequency Poisson GLM with a Gamma prior on the rate, hierarchical across territories.

```python
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

# data: claims, exposure (log-offset), territory (categorical), age, gender
df = pd.read_parquet("policy_year.parquet")
n_terr = df["territory"].nunique()
terr_idx = df["territory"].astype("category").cat.codes.values

with pm.Model() as freq_model:
    # Population-level intercept
    mu = pm.Normal("mu", mu=-2.0, sigma=1.0)
    # Territory-level standard deviation
    tau = pm.HalfCauchy("tau", beta=1.0)
    # Per-territory deviations (non-centered parameterization)
    z = pm.Normal("z", mu=0, sigma=1, shape=n_terr)
    alpha_terr = pm.Deterministic("alpha_terr", mu + tau * z)
    # Fixed effects
    beta_age    = pm.Normal("beta_age", mu=0, sigma=0.5)
    beta_gender = pm.Normal("beta_gender", mu=0, sigma=0.5)

    eta = (alpha_terr[terr_idx]
           + beta_age * df["age_z"].values
           + beta_gender * df["is_male"].values
           + np.log(df["exposure"].values))
    rate = pm.math.exp(eta)
    pm.Poisson("claims", mu=rate, observed=df["claims"].values)

    trace = pm.sample(2000, tune=2000, target_accept=0.95,
                      chains=4, return_inferencedata=True)

print(az.summary(trace, var_names=["mu", "tau", "beta_age", "beta_gender"]))
az.plot_forest(trace, var_names=["alpha_terr"])
```

The non-centered parameterization (`alpha_terr = mu + tau * z`) is the standard fix for the funnel pathology in hierarchical models. The `target_accept=0.95` raises NUTS adaptation to handle the geometry.

## Code: NumPyro version of the same model

NumPyro is a JAX-based reimplementation of Pyro. Identical semantics to PyMC, often 5x to 50x faster on GPU.

```python
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC

def model(terr_idx, age_z, is_male, log_exposure, claims, n_terr):
    mu = numpyro.sample("mu", dist.Normal(-2.0, 1.0))
    tau = numpyro.sample("tau", dist.HalfCauchy(1.0))
    z = numpyro.sample("z", dist.Normal(0.0, 1.0).expand([n_terr]))
    alpha_terr = mu + tau * z
    beta_age = numpyro.sample("beta_age", dist.Normal(0, 0.5))
    beta_gender = numpyro.sample("beta_gender", dist.Normal(0, 0.5))
    eta = (alpha_terr[terr_idx] + beta_age * age_z
           + beta_gender * is_male + log_exposure)
    numpyro.sample("claims", dist.Poisson(jnp.exp(eta)), obs=claims)

kernel = NUTS(model, target_accept_prob=0.95)
mcmc = MCMC(kernel, num_warmup=2000, num_samples=2000, num_chains=4)
mcmc.run(jax.random.PRNGKey(0),
         terr_idx=terr_idx, age_z=df["age_z"].values, is_male=df["is_male"].values,
         log_exposure=jnp.log(df["exposure"].values), claims=df["claims"].values,
         n_terr=n_terr)
mcmc.print_summary()
```

The model code is essentially the same; the runtime difference is dramatic on JAX-supported hardware. For production Bayesian inference at scale (hundreds of thousands of policies, hierarchical at multiple levels) NumPyro on a GPU is the 2026 default.

## Cross-reference

A future essay `docs/theory/16_bayesian_methods.md` will tie these techniques together at theory level, including model comparison (WAIC, LOO-PSIS), posterior predictive checks as a unified diagnostic, and the connection to penalized likelihood and James-Stein estimation. The technique-level material here remains the operational reference.

## Python implementation

**Key libraries:**

- `pymc` (v5+): Full-featured probabilistic programming. Define priors, likelihoods, sample posteriors via NUTS (a variant of Hamiltonian Monte Carlo).
- `arviz`: Posterior diagnostics and visualization (trace plots, forest plots, posterior predictive checks).
- `scipy.stats`: Useful for conjugate prior calculations (e.g., `scipy.stats.beta` for Beta-Binomial updates).
- `bambi`: Bayesian regression models with a formula interface similar to R's `brms`.

```python
# Bayesian A/B test with conjugate Beta-Binomial
import numpy as np
from scipy import stats

# Prior: Beta(1, 1) = Uniform on [0, 1]
alpha_prior, beta_prior = 1, 1

# Observed data
conversions_A, trials_A = 120, 1000
conversions_B, trials_B = 145, 1000

# Posterior distributions
post_A = stats.beta(alpha_prior + conversions_A, beta_prior + trials_A - conversions_A)
post_B = stats.beta(alpha_prior + conversions_B, beta_prior + trials_B - conversions_B)

# P(B > A) via Monte Carlo
samples_A = post_A.rvs(100_000)
samples_B = post_B.rvs(100_000)
prob_B_better = (samples_B > samples_A).mean()
print(f"P(B > A) = {prob_B_better:.4f}")
```

```python
# PyMC example: estimating a Poisson rate
import pymc as pm
import arviz as az

with pm.Model() as claims_model:
    # Prior for claim rate
    lam = pm.Gamma("lambda", alpha=2, beta=1)
    # Likelihood
    obs = pm.Poisson("claims", mu=lam, observed=[3, 5, 4, 2, 6, 3, 4])
    # Sample posterior
    trace = pm.sample(2000, tune=1000, cores=2)

az.summary(trace)
az.plot_trace(trace)
```

## R implementation (if applicable)

- `rstan` / `brms`: Stan-based Bayesian modeling. `brms` provides a formula interface on top of Stan.
- `rstanarm`: Pre-compiled Stan models for common regression tasks.
- `BayesianFirstAid`: Drop-in Bayesian replacements for common frequentist tests.

```r
# brms example: Bayesian Poisson regression
library(brms)
fit <- brm(claims ~ age + territory,
           data = claims_data,
           family = poisson(),
           prior = c(prior(normal(0, 5), class = "b"),
                     prior(normal(0, 10), class = "Intercept")),
           chains = 4, iter = 2000)
summary(fit)
plot(fit)
```

## Gotchas & lessons learned

1. **Prior sensitivity**: Always run a sensitivity analysis. If your conclusions change dramatically with a different reasonable prior, your data is not informative enough to support strong claims. Report results under multiple priors.

2. **Flat priors are not always "non-informative"**: A flat prior on a parameter can become highly informative after a transformation. For example, a flat prior on $\sigma$ is not flat on $\sigma^2$. Use weakly informative priors (e.g., half-Cauchy for scale parameters) rather than truly flat priors.

3. **Conjugate priors are convenient but limited**: They force you to use specific distributional families. Real problems often require more flexible models, which is where MCMC becomes necessary.

4. **MCMC diagnostics are not optional**: Always check R-hat, ESS, trace plots, and divergent transitions. A model that has not converged gives meaningless results. Divergent transitions in particular indicate the sampler is struggling with the posterior geometry.

5. **Computational cost**: MCMC is slower than frequentist methods, sometimes by orders of magnitude. For production systems, consider whether the Bayesian benefits justify the compute time. Variational inference (VI) is a faster approximation, but it can underestimate uncertainty.

6. **Posterior predictive checks**: After fitting, simulate data from the posterior predictive distribution and compare it to the observed data. If the simulated data looks nothing like the real data, your model is misspecified regardless of how clean the MCMC diagnostics look.

7. **Interpreting credible intervals correctly**: A 95% credible interval does *not* mean "95% of future observations will fall here." That would be a prediction interval. The credible interval is about the *parameter*, not future data.

8. **The denominator problem**: $P(D)$ (the evidence) is often intractable. This is why MCMC exists -- it sidesteps computing $P(D)$ entirely. If someone asks you to compute $P(D)$, you are probably doing something wrong (unless you are doing formal model comparison via Bayes factors).

9. **Bayesian does not mean subjective**: The prior is explicitly stated, auditable, and sensitivity-tested. Frequentist methods also embed assumptions (distributional assumptions, asymptotic arguments) that are often less transparent.

10. **Start simple**: Use conjugate models first to build intuition. Move to MCMC only when the problem demands it.

## Resources

- [Statistical Rethinking by Richard McElreath](https://xcelab.net/rm/statistical-rethinking/) -- The best introductory Bayesian textbook. Focuses on intuition and modeling.
- [Bayesian Data Analysis (BDA3) by Gelman et al.](http://www.stat.columbia.edu/~gelman/book/) -- The comprehensive reference.
- [PyMC documentation](https://www.pymc.io/welcome.html) -- Tutorials, examples, and API reference.
- [ArviZ documentation](https://python.arviz.org/en/stable/) -- Bayesian diagnostics and visualization.
- [Kruschke, Doing Bayesian Data Analysis](https://sites.google.com/site/doingbayesiandataanalysis/) -- "The puppy book." Excellent for beginners.
- [Bayesian Methods for Hackers](https://camdavidsonpilon.github.io/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/) -- Free online book using PyMC.
- [Think Bayes by Allen Downey](https://greenteapress.com/wp/think-bayes/) -- Short, code-first introduction.
- [Conjugate prior table (Wikipedia)](https://en.wikipedia.org/wiki/Conjugate_prior) -- Quick reference for conjugate families.
- SOA Exam STAM / CAS Exam MAS-I materials -- Bayesian estimation and credibility sections.
