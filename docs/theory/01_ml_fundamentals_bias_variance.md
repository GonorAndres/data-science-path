# ML Fundamentals: Bias, Variance, and Generalization

## The decomposition

For a squared-error regression problem, expected prediction error at a point $x$ decomposes as:

$$
E\left[ (y - \hat{f}(x))^2 \right] = \underbrace{\left( E[\hat{f}(x)] - f(x) \right)^2}_{\text{bias}^2} + \underbrace{\text{Var}[\hat{f}(x)]}_{\text{variance}} + \underbrace{\sigma^2}_{\text{irreducible}}
$$

Three terms:

1. **Bias.** The systematic gap between the average prediction over different training sets and the true function. Comes from model inflexibility: assuming linearity when the truth is curved.
2. **Variance.** How much the prediction changes across different training sets drawn from the same distribution. Comes from model flexibility: each training set produces a different fit.
3. **Irreducible noise.** The floor. Labels have genuine randomness we cannot predict no matter what.

## Why this matters

Every model selection choice is an implicit bias-variance tradeoff. Regularization strength, tree depth, number of features, training-set size: all of them move the model along the tradeoff curve.

In the **underparameterized regime** (model has fewer parameters than the effective complexity of the data), more parameters and flexibility pushes bias down and variance up. More data pushes variance down (the sampling noise of any single training set matters less) with bias unchanged. The classical claim does not extend cleanly to the **overparameterized regime** discussed under double descent below.

## What the expectation is taken over

A common student misconception is that the expectation in the decomposition averages over the noise $\epsilon$ alone. It does not. The expectation is over training sets: i.i.d. samples of size $n$ drawn from $P(X, Y)$. Each training set produces a different fitted $\hat{f}$, and bias and variance describe how that fitted function behaves on average and how much it fluctuates across hypothetical retrainings on independent samples from the same population. The noise term is separate and reflects label randomness given $X$.

This is why bootstrapping (resampling the training set) is the standard empirical estimator: each bootstrap sample is a stand-in for "another draw from $P$".

## The Bayes-optimal predictor and the noise floor

For squared loss, the function that minimizes expected error is the conditional mean:

$$
f^*(x) = E[Y \mid X = x]
$$

This is the **Bayes-optimal regressor**. The irreducible noise $\sigma^2(x) = \text{Var}(Y \mid X = x)$ is the gap between $Y$ and $f^*(x)$ that no model can close, because the same $x$ corresponds to a distribution of $y$ values, not a single one. For 0-1 classification loss the analog is the Bayes classifier $f^*(x) = \arg\max_y P(Y = y \mid X = x)$ and the Bayes error rate.

Mentioning $f^*$ explicitly clarifies what "irreducible" means: it is irreducible relative to a fixed feature set $X$. Adding informative features can lower it; better models cannot.

## A caveat on loss functions

The clean additive bias-plus-variance-plus-noise decomposition shown above is specific to **squared loss**. For 0-1 loss (classification accuracy) the decomposition does not hold cleanly. Domingos (2000) proposed a unified bias-variance decomposition that works for general loss functions but the bias and variance terms are no longer additive in the simple sense; they interact through the sign of the gap to the Bayes-optimal prediction. A high-variance classifier can have lower 0-1 error than a low-variance one if its variance happens to push predictions toward the correct side of the decision boundary.

In practice this means: do not expect a tidy bias-variance plot for a classifier evaluated by accuracy. Use squared error on the predicted probabilities (the Brier score) if you want a decomposition that behaves classically, or work directly with cross-entropy.

## Intuition through three examples

### Linear regression on quadratic data

If the true relationship is $y = x^2 + \epsilon$ and you fit a straight line, every training set produces roughly the same wrong slope. Bias is high, variance is low.

### Deep decision tree on small data

If the true relationship is $y = x^2 + \epsilon$ and you fit an unrestricted decision tree on 30 points, each training set produces a wildly different tree. Bias is low (tree is flexible enough), variance is high.

### The sweet spot

A quadratic regression or a pruned random forest. Some bias (still some assumption), some variance (some dependence on the training set), but the sum is minimized.

## How to diagnose

Build two learning curves: training error and validation error as a function of training-set size.

- If both plateau high: high bias (underfitting). Add complexity or features.
- If there is a large and persistent gap: high variance (overfitting). Regularize, add data, simplify.
- If both converge to a low level: you are at the good regime.

## Practical moves

### When bias is the problem

- Use a more flexible model (polynomial features, deeper tree, more neurons).
- Add features that capture the relationship the simpler model could not.
- Reduce regularization strength.
- Check if the features themselves are signal-poor (the model cannot fix what the data does not have).

### When variance is the problem

- Get more training data (the single most effective move).
- Reduce model complexity (prune trees, shrink networks, drop features).
- Apply regularization (L2, weight decay, early stopping, dropout).
- Average multiple models (bagging, ensembling).
- Reduce noise in labels if possible (relabel, aggregate).

### When irreducible noise is the problem

- Accept the floor; you cannot beat it.
- Improve the label itself (better measurement, better definition).
- Change the target variable to something more predictable.

## Why deep networks break the textbook

The classical bias-variance view predicts that very flexible models overfit. Modern deep networks with millions or billions of parameters defy this: they can fit random noise on the training set yet still generalize to held-out data when trained on real signal.

The current explanation involves **implicit regularization** of stochastic gradient descent, the structure of natural data, and the **double-descent** phenomenon: test error decreases, then rises sharply, then decreases again as model capacity grows.

The peak in the double-descent curve sits at the **interpolation threshold**: the smallest model capacity at which the model can perfectly fit the training data. For squared loss with $n$ training points, this corresponds roughly to the point where the number of effective parameters equals $n$. Below the threshold the classical bias-variance tradeoff applies. Above it, increasing capacity continues to lower test error because among the many functions that interpolate the training set, gradient descent with appropriate initialization tends to select low-norm (smoother) ones; this is an implicit regularization effect.

**Grokking** (Power et al. 2022) is a related phenomenon in the overparameterized regime: training accuracy reaches 100 percent quickly but test accuracy remains near chance for a long time before suddenly improving. The current interpretation is that the network first memorizes, then over many additional training steps the implicit regularization of the optimizer pulls the solution toward a generalizing one along the interpolating manifold. Grokking and double descent are both signs that "more parameters = more variance" is the wrong mental model in this regime.

For interview purposes: the classical tradeoff holds for most practical ML (tabular, trees, linear models); deep networks operate in an overparameterized regime where scale of data, architecture, and optimization interact non-classically.

## Connecting to the portfolio

Every project in `projects/` makes implicit choices on this tradeoff:

- `nlp-reviews-multilingual` uses a pretrained transformer: most parameters are fixed, we fine-tune on a limited dataset. Bias is constrained by pretraining; variance is kept down by limiting fine-tune epochs.
- `timeseries-retail-forecast` uses LightGBM with strong regularization. Trees are flexible but regularization (min_child_samples, lambda) controls variance.
- `causal-uplift-marketing` explicitly addresses bias introduced by non-random treatment assignment.

Whenever you write a model in these projects, name the bias-variance decision you made.

## References

- Hastie, Tibshirani, Friedman, *The Elements of Statistical Learning*, Chapter 7.
- Goodfellow, Bengio, Courville, *Deep Learning*, Chapter 5.
- Belkin et al., "Reconciling modern machine learning practice and the bias-variance trade-off", 2019 (for double-descent).
- Domingos, "A Unified Bias-Variance Decomposition and its Applications", ICML 2000 (for 0-1 loss extension).
- Power, Burda, Edwards, Babuschkin, Misra, "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets", 2022.
- Shalev-Shwartz, Ben-David, *Understanding Machine Learning*, Chapter 5.

## Exercises

1. Derive the bias-variance decomposition for squared loss starting from $E[(y - \hat{f}(x))^2]$, making explicit which expectation is over the noise and which is over the training set.
2. A model achieves 98 percent training accuracy and 65 percent test accuracy. Is this bias-dominated or variance-dominated, and what intervention do you try first?
3. Explain in one paragraph why adding more training data reduces variance but not the irreducible error.
4. Sketch (do not code) the double-descent curve for a linear model as the number of parameters crosses $n$, and label the interpolation threshold.
5. For 0-1 loss, explain why the classical bias-variance decomposition breaks down and describe what Domingos (2000) proposes instead.
