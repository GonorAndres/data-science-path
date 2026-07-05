---
tags: [time-series, bayesian, state-space]
status: draft
created: 2026-04-25
updated: 2026-04-25
---

# State-Space Models and the Kalman Filter

## What it is

A state-space model decomposes an observed time series into an unobserved (latent) state that evolves over time and a noisy observation generated from that state. The framework subsumes ARIMA, ETS, structural time-series, and many control-theoretic models. The Kalman filter is the closed-form recursive Bayesian inference algorithm for the linear Gaussian special case; the smoother extends it to full-sequence inference; particle filters and EKF/UKF extend the same machinery to nonlinear and non-Gaussian models.

## Why it matters

- **The right level of abstraction.** State-space framing makes assumptions explicit (level, trend, seasonality, regression effects) and lets you mix them additively. ARIMA looks parsimonious but is the same model class with most of the structure hidden in lag coefficients.
- **Online updates.** The filter ingests one observation at a time and updates the posterior; no need to refit on the full history. Critical for streaming forecasting and real-time anomaly detection.
- **Native uncertainty.** Posterior covariances propagate forward; predictive intervals are immediate.
- **Causal impact.** BSTS-based CausalImpact uses state-space models to estimate counterfactual outcomes for marketing and product launches; the actuarial analog is estimating the counterfactual loss experience that would have prevailed without a regulatory change.

## The linear Gaussian state-space model

The canonical form:

$$
\text{State equation:}\quad x_t = F_t\, x_{t-1} + B_t\, u_t + w_t,\qquad w_t \sim \mathcal{N}(0, Q_t)
$$

$$
\text{Observation equation:}\quad y_t = H_t\, x_t + v_t,\qquad v_t \sim \mathcal{N}(0, R_t)
$$

- $x_t \in \mathbb{R}^n$ is the latent state (level, trend, seasonality components, regression coefficients).
- $y_t \in \mathbb{R}^m$ is the observed measurement.
- $F_t, B_t, H_t$ are known (often time-invariant) matrices specifying the dynamics and the measurement.
- $u_t$ is an exogenous input (control signal, holiday indicator).
- $Q_t$ and $R_t$ are the process and observation noise covariances.

A **local level model** has $x_t = x_{t-1} + w_t$ and $y_t = x_t + v_t$: a random walk in the unobserved level, observed with noise. A **local linear trend model** adds a slope state. A **basic structural model** adds a seasonal component as a sum-to-zero set of dummies that evolves over time.

## The Kalman filter

The filter maintains the posterior mean $\hat{x}_{t \mid t}$ and covariance $P_{t \mid t}$ given observations through time $t$. Each new observation triggers a two-step update.

### Predict (prior at time $t$ given data through $t-1$)

$$
\hat{x}_{t \mid t-1} = F_t\, \hat{x}_{t-1 \mid t-1} + B_t\, u_t
$$

$$
P_{t \mid t-1} = F_t\, P_{t-1 \mid t-1}\, F_t^\top + Q_t
$$

### Update (posterior at time $t$ given the new observation)

The innovation (prediction error) and its covariance:

$$
\tilde{y}_t = y_t - H_t\, \hat{x}_{t \mid t-1}
$$

$$
S_t = H_t\, P_{t \mid t-1}\, H_t^\top + R_t
$$

The **Kalman gain** is

$$
K_t = P_{t \mid t-1}\, H_t^\top\, S_t^{-1}
$$

and the posterior update is

$$
\hat{x}_{t \mid t} = \hat{x}_{t \mid t-1} + K_t\, \tilde{y}_t
$$

$$
P_{t \mid t} = (I - K_t H_t)\, P_{t \mid t-1}
$$

The Kalman gain is the optimal weighting between the prior prediction and the new observation: it is small when the measurement is noisy ($R_t$ large) and large when the prior is uncertain ($P_{t \mid t-1}$ large). The derivation drops out of minimizing the trace of $P_{t \mid t}$.

The filter is exactly the Bayesian posterior update for the linear Gaussian model. No approximation.

## The Kalman smoother (RTS)

The filter produces $\hat{x}_{t \mid t}$, the posterior given **only past data**. For full-sequence inference (using future observations to refine past states) run the Rauch-Tung-Striebel smoother backward after the filter:

$$
J_t = P_{t \mid t}\, F_{t+1}^\top\, P_{t+1 \mid t}^{-1}
$$

$$
\hat{x}_{t \mid T} = \hat{x}_{t \mid t} + J_t\, (\hat{x}_{t+1 \mid T} - \hat{x}_{t+1 \mid t})
$$

$$
P_{t \mid T} = P_{t \mid t} + J_t\, (P_{t+1 \mid T} - P_{t+1 \mid t})\, J_t^\top
$$

For retrospective analysis (counterfactual estimation, anomaly tagging on historical data) always use the smoother; the filter is for online operation.

## Connection to ARIMA, ETS, and statsforecast

ARIMA, SARIMA, and the full ETS family (Hyndman, Koehler, Snyder, Grose 2008) have exact state-space representations. The `statsforecast` library and `statsmodels.tsa.statespace` use Kalman filters under the hood for both estimation (via the prediction-error decomposition of the log likelihood) and forecasting. Knowing the state-space form means you can:

- Write custom models (level + slowly varying trend + seasonal + regressor + AR(1) noise) by composing component matrices.
- Inspect the filtered states for diagnostic plots (filtered level vs filtered trend) instead of treating the model as a black box.
- Combine multiple time series under shared latent factors (dynamic factor models) using the same filter.

## Extended Kalman filter (EKF)

When the dynamics or observation are nonlinear, $f$ and $h$ replace $F$ and $H$:

$$
x_t = f(x_{t-1}, u_t) + w_t,\qquad y_t = h(x_t) + v_t
$$

The EKF linearizes around the current estimate at each step. Compute Jacobians

$$
F_t = \frac{\partial f}{\partial x}\bigg|_{\hat{x}_{t-1 \mid t-1}}, \qquad H_t = \frac{\partial h}{\partial x}\bigg|_{\hat{x}_{t \mid t-1}}
$$

and apply the linear Kalman recursion using these Jacobians. EKF works when the nonlinearity is mild and the posterior remains roughly Gaussian. It fails when the dynamics are highly nonlinear or when the posterior is multimodal; common failure modes in tracking and SLAM.

The unscented Kalman filter (UKF) replaces linearization with deterministic sigma-point sampling and propagates the points through $f$ and $h$ exactly. Higher accuracy on mildly to moderately nonlinear models.

## Particle filters

For arbitrary nonlinear or non-Gaussian models, represent the posterior as a weighted ensemble of samples (particles). At each step:

1. **Predict:** propagate each particle through $f$ with simulated process noise.
2. **Update:** reweight each particle by the observation likelihood $p(y_t \mid x_t^{(i)})$.
3. **Resample** when effective sample size drops, drawing particles in proportion to their weights.

Particle filters handle multimodal posteriors and arbitrary noise distributions; the cost is $O(N)$ per step in the number of particles, and the ensemble can degenerate (all particles collapse onto a single trajectory) without careful resampling.

## Bayesian Structural Time Series (BSTS)

BSTS (Scott, Varian 2014) is a state-space framework specifically for forecasting with regressors. The state has interpretable components:

- **Trend:** local level + local linear trend.
- **Seasonality:** weekly, yearly, or custom periods.
- **Regression:** a time-varying or time-invariant linear contribution from external regressors with a spike-and-slab prior on coefficients (sparse selection).

The CausalImpact application (Brodersen et al. 2015) uses BSTS to estimate the counterfactual: fit on pre-intervention data, project forward, compare projections to observed post-intervention values, attribute the gap to the intervention. Used for marketing campaign measurement and policy-effect estimation when randomized control is infeasible.

## Code: Kalman filter on a noisy sine wave

```python
import numpy as np
import matplotlib.pyplot as plt

# Synthetic data: noisy observations of an underlying sine
T = 200
t = np.arange(T)
true_signal = np.sin(0.1 * t)
y = true_signal + np.random.normal(0, 0.5, T)

# Local linear trend model: state = [level, slope]
F = np.array([[1, 1], [0, 1]])
H = np.array([[1, 0]])
Q = np.array([[1e-4, 0], [0, 1e-5]])
R = np.array([[0.25]])

x = np.array([0.0, 0.0])
P = np.eye(2)

filtered = np.zeros(T)
for k in range(T):
    # Predict
    x = F @ x
    P = F @ P @ F.T + Q
    # Update
    innov = y[k] - (H @ x)[0]
    S = (H @ P @ H.T + R)[0, 0]
    K = (P @ H.T / S).ravel()
    x = x + K * innov
    P = (np.eye(2) - np.outer(K, H)) @ P
    filtered[k] = x[0]

plt.plot(t, y, ".", alpha=0.4, label="Observed")
plt.plot(t, true_signal, "g-", label="True signal")
plt.plot(t, filtered, "r-", label="Kalman filtered")
plt.legend(); plt.tight_layout()
```

The filtered series tracks the underlying sine without lag because the local linear trend model can express short-run drift. Reducing $Q$ produces smoother output (more trust in the model); increasing $Q$ tracks faster (more trust in observations). The tuning is exactly the bias-variance tradeoff.

## Code: BSTS sketch with TFP

```python
import tensorflow as tf
import tensorflow_probability as tfp

sts = tfp.sts

observed = tf.constant(y, dtype=tf.float32)
trend = sts.LocalLinearTrend(observed_time_series=observed)
seasonal = sts.Seasonal(num_seasons=7, observed_time_series=observed)
model = sts.Sum([trend, seasonal], observed_time_series=observed)

# Variational inference
variational_posteriors = tfp.sts.build_factored_surrogate_posterior(model=model)
loss = tfp.vi.fit_surrogate_posterior(
    target_log_prob_fn=model.joint_distribution(observed_time_series=observed).log_prob,
    surrogate_posterior=variational_posteriors,
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    num_steps=200,
)

forecast_dist = tfp.sts.forecast(
    model, observed_time_series=observed,
    parameter_samples=variational_posteriors.sample(50),
    num_steps_forecast=30,
)
```

The factored surrogate posterior is a mean-field variational approximation to the parameter posterior; for production work prefer HMC via `tfp.mcmc` despite the cost.

## Common gotchas

1. **Identifiability.** Component variances ($Q$ entries vs $R$) are often weakly identified. Use informative priors or fix one to a sensible value.
2. **Initialization of $P_0$.** Set the initial state covariance large (diffuse prior) so the filter quickly forgets the (arbitrary) initial mean. Statsmodels and PyKalman both support diffuse initialization.
3. **Numerical issues in $S^{-1}$.** Use the Joseph form of the covariance update for numerical stability with long sequences and small noise.
4. **Mismatch between state dynamics and reality.** A local linear trend model on a series with sharp regime breaks will lag through the break; add a structural break component or use particle filters.
5. **Time-varying parameters.** When matrices $F_t$, $H_t$ should change slowly over time, augment the state with the parameters and add a small process noise; this turns parameter learning into a state-estimation problem.

## Resources

- `docs/theory/11_time_series_foundations.md` for the broader time-series context.
- `techniques/temporal-fusion-transformer/README.md` for the deep-learning alternative when n is large and covariates are rich.
- `techniques/garch-volatility/README.md` for the heteroscedastic-noise extension.

## References

- Durbin, Koopman, *Time Series Analysis by State Space Methods*, 2nd ed., Oxford 2012 (the canonical reference).
- Hyndman, Koehler, Snyder, Grose, *Forecasting with Exponential Smoothing: The State Space Approach*, Springer 2008.
- Kalman, "A New Approach to Linear Filtering and Prediction Problems", Trans. ASME 1960.
- Rauch, Tung, Striebel, "Maximum Likelihood Estimates of Linear Dynamic Systems", AIAA Journal 1965.
- Scott, Varian, "Predicting the present with Bayesian structural time series", International Journal of Mathematical Modelling and Numerical Optimisation 2014.
- Brodersen et al., "Inferring causal impact using Bayesian structural time-series models" (CausalImpact), Annals of Applied Statistics 2015.
