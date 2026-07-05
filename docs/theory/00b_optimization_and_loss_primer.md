# Optimization and Loss-Function Primer

This essay is the second prerequisite of the theory library. It establishes the optimization vocabulary (gradient descent and its variants) and the loss-function zoo that the regularization, deep-learning, ranking, and contrastive-learning essays all draw on.

## Why optimization shows up everywhere

Most ML models are estimators of the form $\hat\theta = \arg\min_\theta L(\theta; \mathrm{data})$. Linear regression has a closed-form solution; everything else is solved by iterative optimization. Understanding the iteration matters because the optimizer choice changes which minimum you find, how stable training is, and how regularization interacts with the update rule (the AdamW vs L2 distinction is exactly this).

## Gradient descent from first principles

Suppose $L: \mathbb{R}^p \to \mathbb{R}$ is differentiable. The first-order Taylor expansion at $\theta$ is

$$
L(\theta + d) \approx L(\theta) + \nabla L(\theta)^\top d + O(\|d\|^2).
$$

To make $L$ decrease as fast as possible per unit step length, choose $d \propto -\nabla L(\theta)$. The update rule

$$
\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)
$$

with learning rate $\eta > 0$ is gradient descent. For convex $L$ with Lipschitz gradient (constant $L_{\mathrm{lip}}$), $\eta < 2 / L_{\mathrm{lip}}$ guarantees convergence to the global minimum. For strongly convex $L$ with parameter $\mu$, the convergence rate is geometric: $L(\theta_t) - L(\theta^*) \leq (1 - \mu/L_{\mathrm{lip}})^t \cdot \mathrm{const}$.

In ML, neither convexity nor exact Lipschitz constants hold globally, but the local picture and the intuition transfer.

## Stochastic gradient descent

When the loss is a sum over $n$ examples, $L(\theta) = \tfrac{1}{n} \sum_i \ell_i(\theta)$, computing the full gradient costs $O(n)$ per step. Stochastic gradient descent (SGD) replaces the full gradient by an estimate computed on a single example or a mini-batch:

$$
\theta_{t+1} = \theta_t - \eta \nabla \ell_{i_t}(\theta_t), \qquad i_t \text{ random}.
$$

The estimate is unbiased: $E[\nabla \ell_{i_t}] = \nabla L$. Variance is the cost.

Mini-batch SGD averages the gradient over $B$ examples; variance scales as $1/B$. Larger $B$ gives a better gradient estimate per step but fewer steps per epoch. Empirically, $B$ between 32 and 1024 dominates in deep learning, with linear scaling rules (Goyal et al. 2017) for very large $B$.

## Momentum

SGD oscillates in narrow valleys (high curvature in one direction, low in another) because each step is a noisy gradient. Momentum fixes this by keeping a running average of past gradients:

$$
v_{t+1} = \beta v_t + \nabla L(\theta_t), \qquad \theta_{t+1} = \theta_t - \eta v_{t+1}.
$$

With $\beta \approx 0.9$, the effective step size in a consistent direction grows toward $\eta / (1 - \beta)$, while oscillating components cancel out. Nesterov momentum evaluates the gradient at $\theta_t - \eta \beta v_t$ instead of $\theta_t$, giving a small look-ahead correction.

## Adam and AdamW

Adaptive methods scale each parameter's learning rate by the inverse of a running estimate of its gradient magnitude. Adam (Kingma & Ba 2014) maintains two exponential moving averages:

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \qquad
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2,
$$

and updates with the bias-corrected estimates $\hat m_t = m_t / (1 - \beta_1^t)$, $\hat v_t = v_t / (1 - \beta_2^t)$:

$$
\theta_{t+1} = \theta_t - \eta \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}.
$$

Defaults $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$ work in most cases.

### AdamW: decoupled weight decay

Adding $\lambda \|\theta\|_2^2$ to the loss in Adam does not produce true weight decay; the $\sqrt{\hat v_t}$ rescaling shrinks effective decay differently per parameter. Loshchilov & Hutter's AdamW separates the decay from the gradient update:

$$
\theta_{t+1} = \theta_t - \eta\!\left(\frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} + \lambda \theta_t\right).
$$

This restores the regularization-strength interpretation of $\lambda$. Modern transformer training (BERT, Llama, ViT) defaults to AdamW. The equivalence "weight decay = L2 penalty" holds in vanilla SGD only; with adaptive optimizers, decoupled weight decay is the correct choice.

## Learning-rate schedules

A constant learning rate is rarely optimal. Three patterns dominate.

### Linear warmup

For the first few hundred or few thousand steps, ramp $\eta$ from a small value (often zero) up to its target. Stabilizes early training, especially for large models with adaptive optimizers where bias-correction terms are noisy at $t \approx 0$.

### Cosine annealing

After warmup, decay according to

$$
\eta_t = \eta_{\min} + \tfrac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\!\left(\pi \frac{t - t_{\mathrm{warm}}}{T - t_{\mathrm{warm}}}\right)\right).
$$

The schedule is smooth, ends at a small learning rate, and is the de-facto default for transformer pretraining. SGDR (Loshchilov & Hutter) adds warm restarts.

### Cyclic schedules

Triangular or 1cycle policies (Smith) oscillate the learning rate within a range. Useful for shorter training runs and small models; less common for foundation-model training.

## The loss-function zoo

A loss function encodes the optimization target. Choice matters: it determines what "best" means.

### Regression

- **Squared loss** $\ell(y, \hat y) = (y - \hat y)^2$. The MLE under Gaussian noise. Sensitive to outliers because errors are squared.
- **Mean absolute error (MAE)** $|y - \hat y|$. The MLE under Laplace noise. Robust to outliers; corresponds to median regression.
- **Huber loss.** Quadratic for $|y - \hat y| \leq \delta$, linear beyond. Combines squared-loss efficiency near zero with MAE robustness in the tails. A standard choice for regression with occasional outliers.
- **Quantile (pinball) loss.** $\rho_\tau(u) = u(\tau - \mathbb{1}\{u < 0\})$ where $u = y - \hat y$. Trains a model to estimate the $\tau$-th conditional quantile; foundation of quantile regression and conformal-flavored prediction intervals.

### Classification

- **Cross-entropy.** $\ell = -\sum_k y_k \log \hat p_k$. The negative log-likelihood of a categorical model. Standard for both binary (with the sigmoid) and multiclass (with the softmax) classification.
- **Focal loss** (Lin et al. 2017). $\ell = -(1 - \hat p_y)^\gamma \log \hat p_y$. Down-weights well-classified examples; designed for severe class imbalance in dense object detection. Now used in fraud, churn, and rare-event modeling.
- **Hinge loss.** $\max(0, 1 - y \hat f)$ for $y \in \{-1, +1\}$. The SVM loss; sparser updates than cross-entropy.
- **Label-smoothed cross-entropy.** Soft labels of $1 - \alpha$ on the true class and $\alpha / (K-1)$ elsewhere. Improves calibration and regularizes confidence (essay 02).

### Contrastive and ranking

- **Contrastive (InfoNCE).** Given an anchor $a$, a positive $p$, and negatives $\{n_j\}$:

$$
\ell = -\log \frac{\exp(\mathrm{sim}(a, p)/\tau)}{\exp(\mathrm{sim}(a, p)/\tau) + \sum_j \exp(\mathrm{sim}(a, n_j)/\tau)}.
$$

Foundation of self-supervised representation learning (SimCLR, MoCo, CLIP). Temperature $\tau$ controls how sharp the softmax over negatives is.

- **Triplet loss.** $\max(0, \mathrm{dist}(a, p) - \mathrm{dist}(a, n) + m)$. Older but still useful in face verification.

- **BPR (Bayesian Personalized Ranking)** for recommenders:

$$
\ell = -\log \sigma(\hat y_{u, i^+} - \hat y_{u, i^-})
$$

over observed positive interactions $i^+$ and sampled negatives $i^-$. Pairwise rank loss; decouples the absolute score scale from the ranking objective.

## Convex vs non-convex optimization

A function is convex if the line segment between any two points on its graph lies above the graph. Convex problems have a unique global minimum (when strictly convex) and many algorithms converge to it.

Non-convex problems (deep networks, mixtures, latent-variable models) have:

- **Multiple local minima.** Different initializations land in different basins.
- **Saddle points.** Gradient is zero but the point is neither minimum nor maximum. In high dimensions, saddle points dominate the loss landscape (Dauphin et al. 2014); strict local minima are rare. Momentum and adaptive methods help escape saddles via noise.
- **Plateaus.** Long flat regions where progress is slow.

Yet deep networks are routinely trained to near-zero loss. Why? Empirical and theoretical work (NTK regime, mode connectivity) suggests that wide networks have many global minima of similar generalization quality, and SGD finds them robustly.

## Convergence intuition

Two regularity conditions matter:

- **Lipschitz smoothness.** The gradient does not change too fast: $\|\nabla L(\theta) - \nabla L(\theta')\| \leq L_{\mathrm{lip}} \|\theta - \theta'\|$. Bounds how aggressive the learning rate can be.
- **Strong convexity (parameter $\mu$).** The Hessian's eigenvalues are bounded below by $\mu > 0$. Ensures geometric convergence; absent in deep nets but present in regularized linear models.

The condition number $\kappa = L_{\mathrm{lip}} / \mu$ controls iteration count: vanilla GD needs $O(\kappa \log(1/\epsilon))$ iterations to reach precision $\epsilon$. Preconditioning (Adam's per-parameter scaling, second-order methods like L-BFGS) implicitly reduces $\kappa$ and speeds convergence.

## How this primer is used downstream

- Essay 02 (regularization) cites AdamW vs L2 equivalence; this essay derives both.
- Essay 09 (LLMs) names cross-entropy, contrastive losses, and cosine schedules; this essay defines them.
- Essay 17 (DL fundamentals) builds on the gradient and update vocabulary established here.
- Essay 12 (recsys) cites BPR and InfoNCE; this essay writes them down.
- Essay 18 (calibration) cites quantile loss for prediction intervals.

## References

- Kingma, Ba, "Adam: A Method for Stochastic Optimization", ICLR 2015.
- Loshchilov, Hutter, "Decoupled Weight Decay Regularization" (AdamW), ICLR 2019.
- Loshchilov, Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts", ICLR 2017.
- Goyal et al., "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour", 2017.
- Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
- Smith, "Cyclical Learning Rates for Training Neural Networks", WACV 2017.
- Dauphin et al., "Identifying and attacking the saddle point problem in high-dimensional non-convex optimization", NeurIPS 2014.
- Bottou, Curtis, Nocedal, "Optimization Methods for Large-Scale Machine Learning", SIAM Review, 2018.
