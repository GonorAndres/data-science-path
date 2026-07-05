---
tags: [bayesian, regression, uncertainty-quantification]
status: draft
created: 2026-04-25
updated: 2026-04-25
---

# Gaussian Processes

## What it is

A Gaussian Process (GP) is a distribution over functions. Where a parametric model assumes a fixed function class (linear, neural network with fixed architecture) and learns parameters, a GP places a prior directly on the space of functions and conditions on observed data to obtain a posterior that is itself a distribution over functions.

Formally, a GP on input space $\mathcal{X}$ is specified by a mean function $m(x)$ and a covariance (kernel) function $k(x, x')$ such that for any finite set of inputs $X = \{x_1, \ldots, x_n\}$ the function values $\mathbf{f} = [f(x_1), \ldots, f(x_n)]$ follow a multivariate normal:

$$
\mathbf{f} \sim \mathcal{N}\big(m(X),\ K(X, X)\big)
$$

with $K_{ij} = k(x_i, x_j)$. A sample path from a GP prior is one possible function the model considers a priori plausible; sampling many such paths shows the prior's flexibility. Smooth kernels produce smooth sample paths; rougher kernels produce rougher ones.

## Why it matters

- **Native uncertainty quantification.** The posterior is a Gaussian over function values; predictive intervals come for free, calibrated under the model's assumptions.
- **Small-data regime.** GPs perform well from dozens to a few thousand observations, where a neural network would overfit and a linear model would underfit.
- **Bayesian.** Hyperparameters are learned by maximizing the marginal likelihood, which automatically penalizes overfitting (Occam's razor for kernels).
- **Actuarial fit.** Mortality rate smoothing across age, geographic risk surfaces, premium uncertainty for thin segments are exactly the small-data, intervals-required problems GPs were built for.

## Kernels

The kernel encodes prior assumptions about the function: smoothness, periodicity, scale of variation. Choosing the kernel is the GP analog of choosing a model architecture.

### Squared exponential (RBF)

$$
k(x, x') = \sigma_f^2 \exp\!\left( -\frac{\|x - x'\|^2}{2 \ell^2} \right)
$$

Sample paths are infinitely differentiable. The default kernel for very smooth phenomena. Hyperparameters: signal variance $\sigma_f^2$, length-scale $\ell$. Often too smooth for real data; use Matern instead.

### Matern

A family parameterized by smoothness $\nu$:

- $\nu = 1/2$: $k(r) = \sigma_f^2 \exp(-r/\ell)$. Sample paths are continuous but not differentiable. Equivalent to an Ornstein-Uhlenbeck process.
- $\nu = 3/2$: $k(r) = \sigma_f^2 (1 + \sqrt{3} r / \ell) \exp(-\sqrt{3} r / \ell)$. Once-differentiable sample paths.
- $\nu = 5/2$: $k(r) = \sigma_f^2 (1 + \sqrt{5} r / \ell + 5 r^2 / 3 \ell^2) \exp(-\sqrt{5} r / \ell)$. Twice-differentiable. The 2026 default for general regression on real-world data; smooth enough to interpolate, rough enough to track real noise structure.

As $\nu \to \infty$, Matern converges to the squared exponential.

### Periodic

$$
k(x, x') = \sigma_f^2 \exp\!\left( -\frac{2 \sin^2(\pi |x - x'| / p)}{\ell^2} \right)
$$

Encodes a known period $p$. Useful for daily, weekly, yearly seasonality. Combined additively or multiplicatively with Matern to model "smooth trend plus weekly seasonality".

### Dot product (linear)

$$
k(x, x') = \sigma_b^2 + \sigma_v^2 (x - c)^\top (x' - c)
$$

A GP with this kernel reduces to Bayesian linear regression. Useful as an additive component when the data has a global linear trend on top of nonlinear local structure.

### Compositions

Kernels are closed under addition and multiplication, both of which yield valid kernels. A common recipe for time series is

$$
k = k_{\text{Matern},\,5/2} + k_{\text{periodic, weekly}} \cdot k_{\text{Matern},\,5/2}
$$

a smooth long-term trend plus a slowly changing weekly cycle. The Automatic Statistician (Lloyd et al. 2014) automates kernel composition search.

## GP regression

Observe noisy data $y_i = f(x_i) + \varepsilon_i$ with $\varepsilon_i \sim \mathcal{N}(0, \sigma_n^2)$. The joint distribution of training labels $\mathbf{y}$ and a test value $f_*$ at $x_*$ is multivariate normal; conditioning on $\mathbf{y}$ gives the predictive posterior:

$$
\mu_*(x_*) = k(x_*, X) [K(X, X) + \sigma_n^2 I]^{-1} \mathbf{y}
$$

$$
\sigma_*^2(x_*) = k(x_*, x_*) - k(x_*, X) [K(X, X) + \sigma_n^2 I]^{-1} k(X, x_*)
$$

Both have closed forms. The predictive mean is a linear smoother of the training labels with weights determined by the kernel; the predictive variance grows away from training data and contracts near it.

### Hyperparameter learning

Hyperparameters $\theta = (\sigma_f, \ell, \sigma_n, \ldots)$ are learned by maximizing the **log marginal likelihood**:

$$
\log p(\mathbf{y} \mid X, \theta) = -\frac{1}{2} \mathbf{y}^\top (K_\theta + \sigma_n^2 I)^{-1} \mathbf{y} - \frac{1}{2} \log |K_\theta + \sigma_n^2 I| - \frac{n}{2} \log 2\pi
$$

The first term is a data fit; the second is a complexity penalty (large kernels are penalized). The interplay is the Bayesian Occam's razor: a kernel that is too flexible is penalized even if it fits the data, and a too-rigid kernel fits poorly. No held-out validation set is needed.

## GP classification

For binary classification with $y_i \in \{-1, +1\}$, place a GP prior on a latent function $f$ and use a sigmoid (or probit) link $p(y_i = 1 \mid f(x_i)) = \Phi(f(x_i))$. The posterior over $f$ is no longer Gaussian because the likelihood is non-conjugate. Two approximations:

- **Laplace approximation.** Find the posterior mode and approximate the posterior as Gaussian centered there with covariance from the inverse Hessian. Fast, modestly accurate.
- **Expectation propagation (EP).** Iteratively match moments of cavity distributions. More accurate, more expensive.

For modern applications, GP classification is less common than GP regression because deep neural classifiers dominate; GPs are still preferred when calibrated probabilities on small data matter, e.g., medical risk stratification with hundreds of patients.

## Sparse GPs

The naive GP requires inverting an $n \times n$ matrix, costing $O(n^3)$. For $n > 5000$ this is impractical.

Sparse GPs introduce $m$ **inducing points** $Z = \{z_1, \ldots, z_m\}$ (with $m \ll n$) that summarize the training data. Two main variants:

- **FITC** (Fully Independent Training Conditional, Snelson-Ghahramani 2006). Approximates the prior by assuming training latents are conditionally independent given inducing latents.
- **VFE / SVGP** (Variational Free Energy, Titsias 2009; Stochastic Variational GP, Hensman et al. 2013). Treats inducing values as variational parameters and optimizes a variational lower bound on the marginal likelihood.

Both reduce inference to $O(n m^2)$. SVGP additionally supports stochastic mini-batch training, enabling GPs on millions of points with $m = 100$ to $1000$ inducing points.

## Code: scikit-learn worked example

Synthetic mortality smoothing problem: observed crude mortality rates by single year of age in a small population, with sampling noise that grows where exposure is small. Goal is a smoothed mortality curve with credible intervals.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

# Synthetic data: ages 20 to 90, true log-mortality is a smooth Gompertz curve
rng = np.random.default_rng(0)
ages = np.arange(20, 91).reshape(-1, 1)
true_logmu = -10 + 0.09 * ages.ravel()
true_mu = np.exp(true_logmu)

# Exposure varies; deaths follow Poisson; observed crude rate is noisy
exposure = 1000 * np.exp(-0.5 * ((ages.ravel() - 55) / 30) ** 2)
deaths = rng.poisson(true_mu * exposure)
crude_logmu = np.log(np.maximum(deaths, 0.5) / exposure)

# Sample weights inversely proportional to expected variance of log-rate
# Var(log_mu_hat) approx 1 / deaths
weights = np.maximum(deaths, 1).astype(float)

kernel = ConstantKernel(1.0) * Matern(length_scale=10.0, nu=2.5) + WhiteKernel(1e-2)
gp = GaussianProcessRegressor(kernel=kernel, alpha=1.0 / weights,
                              n_restarts_optimizer=5, normalize_y=True)
gp.fit(ages, crude_logmu)

age_grid = np.linspace(20, 90, 200).reshape(-1, 1)
mean, std = gp.predict(age_grid, return_std=True)

plt.figure(figsize=(9, 5))
plt.scatter(ages, crude_logmu, s=10, alpha=0.5, label="Crude observed")
plt.plot(age_grid, mean, "k-", label="GP smoothed")
plt.fill_between(age_grid.ravel(), mean - 1.96 * std, mean + 1.96 * std,
                 alpha=0.2, label="95% credible interval")
plt.plot(ages, true_logmu, "r--", label="Truth")
plt.xlabel("Age")
plt.ylabel("log mortality rate")
plt.legend()
plt.tight_layout()
```

The GP smoothes the noisy crude rates while widening its credible interval at ages with low exposure (small `weights`), exactly the right behavior for actuarial mortality work.

## Actuarial applications

- **Mortality rate smoothing.** Whittaker-Henderson is the classical actuarial method; a Matern-5/2 GP is a Bayesian generalization that produces credible intervals natively.
- **Lee-Carter extensions.** Replace the parametric age-period structure with a GP over (age, year) for a smoother and more honest extrapolation.
- **Small-area estimation.** Per-municipality loss ratios in a small Mexican state with thin data; the GP shares strength across municipalities via a kernel on geographic distance.
- **Premium uncertainty for thin segments.** When pricing a risk class with only 50 historical observations, a GP gives a posterior over the pure premium with intervals that widen for unusual feature combinations.
- **Bayesian optimization for actuarial model tuning.** Scikit-optimize and Optuna's GP surrogate use exactly this machinery to tune hyperparameters.

## Common gotchas

1. **Length-scale identifiability.** With small datasets the marginal likelihood can be flat in $\ell$. Initialize from multiple starting points (`n_restarts_optimizer=5` or more) and inspect.
2. **Numerical conditioning.** $K + \sigma_n^2 I$ becomes near-singular when $\sigma_n$ is small relative to the data. The `alpha` parameter in scikit-learn adds a small jitter; increase if you see Cholesky errors.
3. **Anisotropic length-scales.** With multivariate inputs, use one length-scale per input dimension (ARD: automatic relevance determination). Inputs with large fitted $\ell$ are effectively ignored, which is a useful feature-selection diagnostic.
4. **Predictive variance is conditional on the kernel.** The intervals are calibrated under the assumed model; if the kernel is wrong the intervals are wrong. Compare different kernels via marginal likelihood and pick by Bayes factor or, more pragmatically, by held-out log predictive density.
5. **Scaling beyond a few thousand points.** Use SVGP via `gpytorch` or `gpflow`. Scikit-learn's GP does not implement sparse approximations.

## Resources

- `techniques/bayesian-statistics/README.md` for the Bayesian background.
- `techniques/bayesian-credibility/README.md` for the actuarial connection (credibility theory is a special-case GP).
- `docs/theory/11_time_series_foundations.md` for the time-series context where GPs meet state-space models.

## References

- Rasmussen, Williams, *Gaussian Processes for Machine Learning*, MIT Press 2006 (the canonical reference, free online at gaussianprocess.org).
- Titsias, "Variational Learning of Inducing Variables in Sparse Gaussian Processes", AISTATS 2009.
- Hensman, Fusi, Lawrence, "Gaussian Processes for Big Data", UAI 2013.
- Snelson, Ghahramani, "Sparse Gaussian Processes using Pseudo-inputs", NeurIPS 2006.
- Lloyd, Duvenaud, Grosse, Tenenbaum, Ghahramani, "Automatic Construction and Natural-Language Description of Nonparametric Regression Models" (Automatic Statistician), AAAI 2014.
- Gpytorch documentation, gpytorch.ai.
- GPflow documentation, gpflow.github.io.
