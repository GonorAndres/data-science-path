# Probability and Statistical Estimation Primer

This essay establishes the probability and inference vocabulary that the rest of the theory library assumes. The actuarial reader will recognize most of this from undergraduate coursework; the goal here is to fix notation, name the asymptotic results that ML papers cite without proof, and surface the bridges from classical estimation to credibility theory.

## Random variables and joint distributions

A random variable $X$ is a measurable function from a probability space to $\mathbb{R}$. Its distribution is fully described by either its probability mass function $p(x)$ (discrete case) or its density $f(x)$ (continuous case). The expectation is

$$
E[X] = \int x f(x) \, dx
$$

with the integral replaced by a sum in the discrete case. The variance is $\operatorname{Var}(X) = E[(X - E[X])^2]$.

For two random variables, the joint distribution $P(X, Y)$ encodes everything: marginals are obtained by integrating out, the conditional distribution is $P(Y \mid X) = P(X, Y) / P(X)$, and independence is the factorization $P(X, Y) = P(X) P(Y)$.

Supervised learning is, at its core, the study of $P(Y \mid X)$: the conditional distribution of a target given features. The bias-variance decomposition (essay 01), the regularization story (essay 02), and the entire causal-inference apparatus (essay 08b) all rest on careful management of $P(X, Y)$.

## Maximum likelihood estimation

Suppose we observe $n$ independent draws $X_1, \dots, X_n$ from a parametric family $f(x; \theta)$, $\theta \in \Theta \subseteq \mathbb{R}^p$. The log-likelihood is

$$
\ell(\theta) = \sum_{i=1}^n \log f(X_i; \theta).
$$

The maximum likelihood estimator is $\hat\theta_{\mathrm{MLE}} = \arg\max_\theta \ell(\theta)$.

### Score function

The score is the gradient of the log-likelihood:

$$
U(\theta) = \nabla_\theta \ell(\theta) = \sum_{i=1}^n \nabla_\theta \log f(X_i; \theta).
$$

Under regularity conditions, $E_\theta[U(\theta)] = 0$. The MLE satisfies $U(\hat\theta) = 0$ at an interior maximum.

### Fisher information

The Fisher information matrix is

$$
I(\theta) = -E_\theta\left[\nabla^2_\theta \log f(X; \theta)\right] = E_\theta[U(\theta) U(\theta)^\top] / n
$$

(when normalized per observation). It measures how sharply the likelihood curves around the truth: high information means the data are highly informative about $\theta$.

### Asymptotic normality

Under standard regularity assumptions (identifiability, smoothness, finite information),

$$
\sqrt{n}\,(\hat\theta_{\mathrm{MLE}} - \theta_0) \xrightarrow{d} \mathcal{N}\!\left(0,\, I(\theta_0)^{-1}\right).
$$

This is the workhorse result that justifies Wald confidence intervals, $z$-tests on coefficients, and the standard errors that statsmodels prints next to a GLM coefficient. The Cramer-Rao lower bound says no unbiased estimator can do better than $I(\theta_0)^{-1}$ asymptotically, so the MLE is asymptotically efficient.

### Practical computation

In ML practice the MLE is found by gradient-based optimization (essay 00b). Logistic regression, GLMs, and the conditional likelihood of a transformer language model are all MLE problems; cross-entropy loss is the negative log-likelihood of a categorical model.

## The delta method

Suppose $\hat\theta$ is asymptotically normal and we want the distribution of $g(\hat\theta)$ for a smooth function $g$. The first-order Taylor expansion gives

$$
\sqrt{n}\,(g(\hat\theta) - g(\theta_0)) \xrightarrow{d} \mathcal{N}\!\left(0, \nabla g(\theta_0)^\top I(\theta_0)^{-1} \nabla g(\theta_0)\right).
$$

The delta method is how we get standard errors for derived quantities: predicted probabilities from a logistic regression, ratios of coefficients, elasticities. It also justifies the variance formula for log-odds in epidemiology and the standard error for an exponentiated GLM coefficient.

## The bootstrap

When asymptotic theory is unavailable or untrustworthy (small $n$, irregular loss, complex estimator), the bootstrap gives a computational substitute.

### Nonparametric bootstrap

1. Resample $n$ observations with replacement from the data.
2. Compute the estimator $\hat\theta^{*(b)}$ on the resample.
3. Repeat $B$ times (typically 1,000 to 10,000).
4. The empirical distribution of $\{\hat\theta^{*(b)}\}$ approximates the sampling distribution of $\hat\theta$.

Confidence intervals: the percentile method takes the $\alpha/2$ and $1 - \alpha/2$ quantiles of the bootstrap distribution. The BCa (bias-corrected and accelerated) method is more accurate when the sampling distribution is skewed.

### Parametric bootstrap

When a model is fit and trusted, simulate new data from the fitted model and re-estimate. This is the engine behind ODP bootstrap reserving in actuarial loss reserving.

### When the bootstrap fails

Heavy-tailed distributions, dependence (time series, clusters), and parameters on the boundary of the parameter space all break the standard bootstrap. Block bootstrap and stationary bootstrap are the time-series fixes; explicit clustering is the dependence fix.

## Hypothesis testing

A null hypothesis $H_0$ specifies a parameter value or set; the alternative $H_1$ specifies its complement. A test statistic $T$ is computed from the data and compared to its sampling distribution under $H_0$.

### p-values

The $p$-value is $P_{H_0}(T \geq t_{\mathrm{obs}})$: the probability of seeing a statistic at least as extreme as observed, under the null. A $p$-value is a measure of evidence against $H_0$, not the probability that $H_0$ is true.

### Type I and Type II errors, power

- **Type I error rate ($\alpha$).** Probability of rejecting $H_0$ when it is true.
- **Type II error rate ($\beta$).** Probability of failing to reject $H_0$ when $H_1$ is true.
- **Power.** $1 - \beta$: probability of correctly rejecting under a specific alternative.

Power increases with sample size, effect size, and signal-to-noise ratio. The classical sample-size formula for a two-sample test (essay 08a) inverts the power relation.

### Confidence intervals

A $(1-\alpha)$ confidence interval $[L, U]$ is a random interval that, in repeated sampling, contains the true parameter with probability $1 - \alpha$. The Wald interval

$$
\hat\theta \pm z_{1-\alpha/2}\, \widehat{\mathrm{SE}}(\hat\theta)
$$

is the asymptotic-normality go-to. Likelihood-ratio and bootstrap intervals can be more accurate.

A confidence interval is **not** a probability statement about the parameter. The parameter is fixed; the interval is random. The probability is over the data-generating process across hypothetical replications.

## Bayesian credible intervals

A credible interval is the Bayesian counterpart: $P(\theta \in [L, U] \mid \mathrm{data}) = 1 - \alpha$, computed from the posterior $\pi(\theta \mid x) \propto f(x \mid \theta) \pi(\theta)$. Unlike a confidence interval, the parameter is the random object and the interval is fixed once the data are observed.

In large samples with weak priors, Bayesian credible intervals and frequentist confidence intervals usually agree numerically (Bernstein-von Mises). They diverge when priors carry meaningful information or when the sample is small. For an actuarial reader, the credibility-theory connection is direct: Buhlmann credibility is the Bayesian posterior mean of the claim frequency under a conjugate prior.

## Multiple testing

If you run $m$ independent tests at level $\alpha$, the family-wise probability of at least one false positive is $1 - (1 - \alpha)^m$, which approaches 1 quickly. Two standard corrections:

### Bonferroni

Reject if $p_i < \alpha / m$. Controls the family-wise error rate (FWER) at $\alpha$. Conservative; loses power when $m$ is large.

### Benjamini-Hochberg (BH)

Sort $p$-values $p_{(1)} \leq \dots \leq p_{(m)}$. Find the largest $k$ such that $p_{(k)} \leq k\alpha / m$. Reject all hypotheses with $p$-value $\leq p_{(k)}$. Controls the false discovery rate (FDR), the expected proportion of false positives among rejections.

BH is the right correction for screening many features (genomics, marketing campaigns, many A/B tests); Bonferroni is the right correction when any single false positive is unacceptable.

## Actuarial bridge: MLE meets credibility

Buhlmann credibility writes the credibility-weighted estimate of an individual's true mean as

$$
\widehat{\mu_i} = Z \bar X_i + (1 - Z) \mu, \qquad Z = \frac{n}{n + k},
$$

where $\bar X_i$ is the individual's observed mean over $n$ exposures, $\mu$ is the population mean, and $k = E[\text{within-class variance}] / \operatorname{Var}(\text{between-class means})$.

This estimator is exactly the posterior mean under a normal-normal conjugate model with a properly chosen prior variance: $\mu_i \mid \text{data} \sim \mathcal{N}(\widehat{\mu_i}, \cdot)$. Empirical Bayes plugs in MLEs of the hyperparameters $(\mu, k)$ to make credibility a fully data-driven procedure. The bridge from MLE to credibility runs through Bayesian analysis: estimate the prior from the data via likelihood, then take the posterior mean.

The same logic extends to target encoding in feature engineering (essay 05), where the smoothed mean encoding $\widehat{y}_c = (n_c \bar y_c + m \bar y) / (n_c + m)$ is precisely the credibility-weighted average with $k = m$. A James-Stein-style empirical Bayes target encoder is the principled actuarial generalization.

## How this primer is used downstream

- Essay 01 takes for granted the expectation operator over training sets and the joint distribution $P(X, Y)$.
- Essay 02 leans on the Bayesian-prior interpretation of L2 and L1 penalties; the MAP estimator is the optimum of a penalized log-likelihood.
- Essay 03 cites the no-unbiased-CV-variance-estimator result, which is a sampling-distribution claim about a statistic.
- Essay 06 treats SHAP values as expectations under different value-function distributions; the conditional vs interventional debate is a probability-theory debate.
- Essays 08a and 08b make repeated use of asymptotic normality, the delta method, and clustered standard errors.
- Essay 18 derives conformal coverage guarantees from exchangeability, a probability-theory property strictly weaker than IID.

## References

- Casella, Berger, *Statistical Inference*, 2nd ed., 2002.
- Murphy, *Probabilistic Machine Learning: An Introduction*, MIT Press, 2022.
- Gelman, Carlin, Stern, Dunson, Vehtari, Rubin, *Bayesian Data Analysis*, 3rd ed., 2013.
- van der Vaart, *Asymptotic Statistics*, Cambridge University Press, 1998.
- Efron, Tibshirani, *An Introduction to the Bootstrap*, 1993.
- Benjamini, Hochberg, "Controlling the false discovery rate: a practical and powerful approach to multiple testing", JRSS-B, 1995.
- Buhlmann, Gisler, *A Course in Credibility Theory and its Applications*, Springer, 2005.
