# Regularization and the Art of Generalization

## What regularization is

Regularization is anything that discourages the model from memorizing the training set: a penalty on the loss, a constraint on the optimization, or a procedural rule that limits flexibility. The goal is always to shift the bias-variance trade-off toward lower variance at the cost of higher bias, so test error drops.

This essay assumes the optimization vocabulary in essay 00b (gradient descent, SGD, Adam, AdamW) and the probability vocabulary in essay 00 (MAP estimation, conjugate priors).

## Taxonomy

Six families appear in practice:

1. **Norm penalties on parameters.** L1, L2, elastic net, group lasso, SCAD, MCP.
2. **Early stopping.** Halt before validation loss rises.
3. **Capacity constraints.** Tree depth, neural width, max features per split, attention head count.
4. **Noise injection.** Dropout, data augmentation, label smoothing, stochastic depth.
5. **Ensembling.** Averaging models reduces variance.
6. **Data-level regularization.** More data, cleaner labels, balanced sampling, transfer learning.

Each family targets a different failure mode.

## L1 vs L2 in linear models

With squared loss and penalty strength $\lambda$:

- **L2 (Ridge):** $\min \|y - X\beta\|_2^2 + \lambda \|\beta\|_2^2$. Shrinks all coefficients proportionally. Stable with correlated features.
- **L1 (Lasso):** $\min \|y - X\beta\|_2^2 + \lambda \|\beta\|_1$. Produces sparse solutions; coefficients can be exactly zero.
- **Elastic Net:** linear combination of L1 and L2. Gets the feature selection of L1 with the stability of L2 under correlation.

Geometrically, L2's penalty contour is a ball (no corners; optima do not snap to axes). L1's contour is a diamond with corners on the axes, where many solutions concentrate.

Practical rule:

- Many correlated features and you want all of them shrunk: L2.
- Many features and you want a sparse model for interpretability: L1.
- Default baseline: Elastic Net with $\alpha = 0.5$.

## Bayesian-prior interpretation (the actuarial bridge)

L2 and L1 penalties have a clean Bayesian interpretation that connects directly to credibility theory.

### L2 = Gaussian prior

Suppose $y \mid X, \beta \sim \mathcal{N}(X\beta, \sigma^2 I)$ and $\beta \sim \mathcal{N}(0, \tau^2 I)$. The log-posterior is

$$
\log \pi(\beta \mid y) = -\tfrac{1}{2\sigma^2} \|y - X\beta\|_2^2 - \tfrac{1}{2\tau^2} \|\beta\|_2^2 + \mathrm{const}.
$$

The MAP estimator is exactly the ridge solution with $\lambda = \sigma^2 / \tau^2$. The penalty strength is the noise-to-prior ratio: high noise or tight prior pulls coefficients toward zero.

### L1 = Laplace prior

If $\beta_j \sim \mathrm{Laplace}(0, b)$ with density $\propto \exp(-|\beta_j|/b)$, the log-prior is $-\|\beta\|_1 / b$ and the MAP recovers the lasso with $\lambda = \sigma^2 / b$. The Laplace distribution has heavier tails near zero than a Gaussian, which is why some coefficients are pushed exactly to zero.

### Connection to credibility theory

Buhlmann credibility writes the credibility-weighted mean as $Z \bar X_i + (1 - Z) \mu$ with $Z = n/(n+k)$. This is precisely the posterior mean of $\mu_i$ under a normal-normal conjugate model. Ridge regression is the multivariate generalization: each coefficient is shrunk toward zero (or toward a prior mean) by an amount controlled by the relative weight of data and prior. For the actuarial reader, **regularization is credibility theory applied to a regression coefficient vector**.

## Beyond L1 and L2: SCAD and MCP

L1 introduces a non-trivial bias on large coefficients (it shrinks even strong signals). Two non-convex penalties fix this.

### SCAD (Smoothly Clipped Absolute Deviation)

Fan & Li (2001). The penalty is L1 for small coefficients, transitions to a quadratic, and becomes flat for large coefficients:

$$
p_\lambda'(|\beta|) = \lambda \mathbb{1}\{|\beta| \leq \lambda\} + \frac{(a\lambda - |\beta|)_+}{a - 1}\mathbb{1}\{|\beta| > \lambda\}, \quad a > 2.
$$

Large coefficients are not shrunk. Achieves the oracle property under regularity conditions: estimates are asymptotically equivalent to the OLS fit on the true active set.

### MCP (Minimax Concave Penalty)

Zhang (2010). Similar shape:

$$
p_\lambda(|\beta|) = \int_0^{|\beta|} \left(\lambda - t/\gamma\right)_+ dt.
$$

MCP transitions more smoothly than SCAD; both are implemented in `ncvreg` (R) and a few Python packages.

These appear in actuarial pricing literature where the L1 bias on the dominant rating variables (driver age, vehicle group) is unwanted.

## Group lasso and structured sparsity

When categorical levels expand into many one-hot columns, L1 can drop some levels but keep others, producing an incoherent feature. Group lasso penalizes the L2 norm of each group of coefficients:

$$
\sum_g \sqrt{p_g}\, \|\beta_g\|_2,
$$

so an entire group is either zero or non-zero. Standard for categorical encodings, time-series lag groups, and grouped covariates.

## Weight decay in deep learning

Weight decay adds $\lambda \|\theta\|_2^2 / 2$ to the loss. The classical Hinton formulation adds $\lambda \theta$ to the gradient before each step.

In **vanilla SGD** with a fixed learning rate, weight decay is mathematically equivalent to L2 penalty in the loss; the two formulations produce identical updates.

In **adaptive optimizers (Adam, RMSProp)**, the equivalence breaks. The per-parameter learning rate $1/\sqrt{\hat v_t}$ rescales the gradient component $\lambda \theta$, so different parameters experience different effective decay strengths. Loshchilov & Hutter's AdamW restores the regularization-strength interpretation by decoupling the decay step from the gradient update (see essay 00b for the update rule). **The "weight decay equals L2" identity holds only in vanilla SGD**; under Adam, treat them as different operations.

## Early stopping as implicit L2

Yao, Rosasco, and Caponnetto (2007) showed that gradient descent on a convex loss, stopped early, behaves like an L2-penalized solution. The number of iterations $T$ plays the role of an inverse penalty: more iterations means a smaller effective $\lambda$. This is one reason early stopping works as a regularizer without any explicit penalty term, and why compute-aware practitioners often use shorter training runs as a cheap regularization knob.

In neural networks, early stopping on a validation metric is the canonical implementation: monitor validation loss each epoch, save the best checkpoint, stop after $N$ patience epochs without improvement.

## Dropout

During training, randomly zero a fraction $p$ of activations per layer. At inference, scale all activations by $(1-p)$, or use inverted dropout where the zeroing step itself includes a $1/(1-p)$ rescale.

The original motivation (Srivastava et al. 2014) frames dropout as "training an ensemble of $2^n$ subnetworks sharing weights". Treat this as a heuristic interpretation, not a theorem: the mathematical equivalence to true ensembling holds only in restricted linear models. Empirically dropout regularizes; the precise mechanism is still a research question.

Typical $p$: 0.1 to 0.5 for fully connected layers, 0.0 to 0.2 for convolutional layers. Disabled at eval time (`model.eval()` in PyTorch). Stochastic depth and DropConnect are variants on the same idea applied to layers and weights respectively.

## Data augmentation

- **Images.** Random crops, flips, color jitter, rotation, mixup, cutmix, RandAugment.
- **Text.** Synonym replacement, back-translation, paraphrase, EDA.
- **Time series.** Jittering, scaling, warping, mixup.
- **Tabular.** SMOTE for minority classes, Gaussian noise.

Augmentation expands the effective training distribution. Apply only during training. Test-time augmentation (TTA) is an inference-time variant: average predictions over augmented copies of a test input.

## Label smoothing

Replace one-hot labels with $1 - \alpha$ for the true class and $\alpha / (K-1)$ for others. Prevents over-confident predictions, which improves both generalization and calibration (essay 18). Standard in modern image classification and machine translation.

## Ensemble methods as regularization

Averaging $M$ independent models trained on bootstrapped data reduces variance by a factor of roughly $M$ (with diminishing returns due to correlation between learners). Bagging is the mechanism behind Random Forest.

Boosting (XGBoost, LightGBM) is sequential and primarily reduces bias, but early stopping rounds and column subsampling within boosting act as regularizers.

## Cross-validation as tuning regularization strength

Hyperparameters that control regularization strength ($\lambda$, max depth, dropout $p$, patience, stochastic depth probability) must be chosen on held-out data. Essay 03 covers cross-validation in detail. The output of CV is a regularization strength, not a model.

## Choosing $\lambda$ in practice

1. Train at several $\lambda$ values, spanning orders of magnitude ($10^{-4}, 10^{-3}, \dots, 10^{0}$).
2. For each $\lambda$, compute CV error.
3. Pick the $\lambda$ at the minimum, or the largest $\lambda$ within one standard error of the minimum (Hastie et al.'s **1-SE rule** for a sparser, more stable model).

For tree models, sweep `max_depth`, `min_child_samples`, `subsample`, `colsample_bytree`. For deep nets, sweep weight decay $\lambda$, dropout $p$, and the EMA decay if used.

## When regularization alone is not enough

If the training set is small relative to feature count and noise, regularization buys some generalization but cannot manufacture signal. Options in that regime:

- Collect more data (first-best).
- Reduce feature count to match data.
- Use transfer learning (import signal from a larger pretraining corpus).
- Use domain priors (Bayesian models with informative priors, hand-crafted features, monotonic constraints).

## Common mistakes

1. Tuning regularization on the test set. Test set becomes a validation set; you need a new test set.
2. Using L1 for feature selection when features are correlated. L1 picks one of a correlated group arbitrarily; selections are unstable.
3. Applying the same dropout rate to all layers without tuning. Different layers benefit from different rates.
4. Stacking dropout on top of batch norm without thought. They interact; sometimes one is enough.
5. Treating "more regularization" as universally safe. Over-regularized models underfit; bias rises above noise floor.
6. Believing weight decay and L2 are interchangeable in Adam. They are not (use AdamW).

## Linking to the portfolio

- `nlp-reviews-multilingual`: weight decay on the transformer adapter, early stopping on validation macro-F1.
- `timeseries-retail-forecast`: LightGBM with `min_child_samples`, `reg_alpha`, `reg_lambda`, subsampling.
- `cv-defect-detection`: data augmentation as the primary regularizer, weight decay secondary.
- `mlops-model-service`: monitors test-set performance over time; if production drift exceeds a threshold, retrain (a system-level analogue of regularization).

## References

- Hastie, Tibshirani, Friedman, *The Elements of Statistical Learning*, Chapter 3 (Ridge/Lasso) and Chapter 8 (ensembling).
- Goodfellow, Bengio, Courville, *Deep Learning*, Chapter 7.
- Loshchilov, Hutter, "Decoupled Weight Decay Regularization" (AdamW), ICLR 2019.
- Yao, Rosasco, Caponnetto, "On Early Stopping in Gradient Descent Learning", Constructive Approximation, 2007.
- Fan, Li, "Variable Selection via Nonconcave Penalized Likelihood and its Oracle Properties" (SCAD), JASA, 2001.
- Zhang, "Nearly Unbiased Variable Selection under Minimax Concave Penalty" (MCP), Annals of Statistics, 2010.
- Srivastava et al., "Dropout: A Simple Way to Prevent Neural Networks from Overfitting", JMLR, 2014.
- Szegedy et al., "Rethinking the Inception Architecture for Computer Vision" (label smoothing), CVPR 2016.

## Exercises

1. Show algebraically that L2 regularization in the MAP framework corresponds to a Gaussian prior on the weights, and identify the relationship between the prior variance and the regularization strength $\lambda$.
2. You train a Ridge regression with $\lambda = 0.01$ using AdamW. Is this equivalent to L2 regularization? Explain why or why not.
3. State the 1-SE rule for selecting $\lambda$ in cross-validation and explain when you would prefer it over the minimum-CV-error rule.
4. A dropout rate of $0.5$ is used during training. At inference, should you scale the activations? By what factor and why?
5. Explain early stopping as implicit regularization and state which classical regularizer it approximates.
