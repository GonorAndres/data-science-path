# ML Theory Flashcards

Core ML theory questions that appear in almost every data science interview loop in Mexico. Format: question, crisp answer (the "what"), short elaboration (the "why and how"), and a common follow-up trap.

Use spaced repetition: work through 10 cards per day for a week before an interview.

---

## Bias, variance, generalization

### Q1. What is the bias-variance tradeoff?

**Crisp:** Expected test error decomposes into squared bias + variance + irreducible noise. More flexible models reduce bias but increase variance.

**Why:** Bias is error from wrong assumptions (underfitting). Variance is sensitivity to training-set fluctuation (overfitting). Irreducible noise is the floor set by label noise or unknowable features.

**Trap:** Interviewers ask "what if we add more data?" Answer: variance drops (model sees more signal), bias is unchanged (still same functional form).

### Q2. Define overfitting and how to detect it.

**Crisp:** Model performs well on training data, poorly on held-out data. Detect via a gap between training and validation metrics.

**Why:** Usually caused by model complexity exceeding what the data supports, or data leakage inflating training scores.

**Trap:** "If training and validation both score poorly, is that overfitting?" No, that is underfitting or a bug. Overfitting specifically means the gap.

### Q3. How do you detect underfitting?

**Crisp:** High error on both train and validation, with train error nowhere near irreducible noise. Fix: more flexibility (deeper model, more features, fewer regularization).

---

## Regularization

### Q4. L1 vs L2 regularization.

**Crisp:** L1 (Lasso) adds sum of absolute weights; L2 (Ridge) adds sum of squared weights. L1 produces sparse solutions; L2 shrinks but does not zero out.

**Why:** L1 geometry is diamond (corners at axes), so the optimum frequently lands on a corner where weights are exactly zero. L2 geometry is a circle, so weights shrink proportionally.

**Trap:** "Does L1 always select features well?" Not if features are highly correlated; L1 picks one of a correlated group arbitrarily. Elastic Net (L1 + L2) handles this better.

### Q5. Why does weight decay in neural networks differ from L2 in linear models?

**Crisp:** In SGD with weight decay, the weight decay term is added to the gradient; with adaptive optimizers (Adam), it is often applied after the gradient step (AdamW). In linear regression, both are equivalent.

**Trap:** Asked as a deep-learning follow-up. The right answer is "it depends on the optimizer."

### Q6. Dropout intuition.

**Crisp:** Randomly zeroing activations during training forces the network to distribute representations across units, acting as a noise-injection regularizer.

**Why:** Can be interpreted as training an ensemble of subnetworks sharing weights. At inference, activations are scaled so the expected output matches training.

---

## Cross-validation

### Q7. When does k-fold cross-validation fail?

**Crisp:** When data has structure that random splits violate. Common cases: time series (leakage from future into train), grouped data (same user in both train and val), rare labels (some folds may have zero positives).

**Fix:** TimeSeriesSplit, GroupKFold, StratifiedKFold respectively.

### Q8. What is nested cross-validation and when do you need it?

**Crisp:** An inner CV loop for hyperparameter tuning, an outer CV loop for generalization estimation.

**Why:** If you tune on the same CV you use for reporting, the reported score is optimistic (you selected the hyperparameters that happened to work best on those folds).

**Trap:** Expensive. In practice, people use one held-out test set for final reporting.

### Q9. Temporal cross-validation for forecasting.

**Crisp:** Train on [t0, t1], validate on [t1, t2], roll forward. Never shuffle time.

**Why:** A model trained on future data predicting past data overstates performance. Temporal splits simulate real deployment conditions.

---

## Evaluation metrics

### Q10. When is accuracy misleading?

**Crisp:** Class imbalance. 99 percent negative class gives 99 percent accuracy for a model that always predicts negative.

**Better:** Precision, recall, F1, ROC-AUC, PR-AUC, log loss.

### Q11. Precision vs recall in fraud detection.

**Crisp:** Fraud is rare; false negatives (missed fraud) are costly; false positives (blocking a real transaction) are also costly. The tradeoff depends on business weights.

**Trap:** Interviewer expects "it depends". Do not pick one without asking about costs.

### Q12. ROC-AUC vs PR-AUC.

**Crisp:** ROC-AUC treats false positive rate symmetrically with recall; PR-AUC zooms in on positive-class performance. Under heavy class imbalance, PR-AUC is more informative.

**Why:** ROC-AUC of 0.9 on a dataset with 1 percent positives can still mean most predicted positives are wrong. PR-AUC exposes that.

### Q13. Calibration vs discrimination.

**Crisp:** Discrimination is whether high-scored instances are truly more likely to be positive (measured by AUC). Calibration is whether a predicted probability of 0.7 truly means 70 percent positive rate (measured by reliability diagrams or Brier score).

**Why it matters:** Risk pricing, A/B testing, expected-value decisions all require calibrated probabilities. A well-discriminating but miscalibrated model is fine for ranking but wrong for decisions.

### Q14. How to calibrate a model.

**Crisp:** Platt scaling (logistic regression on raw scores) or Isotonic regression (monotonic step function). Use on a held-out calibration set, not on training data.

---

## Class imbalance

### Q15. Techniques for imbalanced classification.

**Crisp:** Class weights in the loss; resampling (SMOTE, oversampling the minority, undersampling the majority); threshold tuning; one-class or anomaly detection framing.

**Trap:** SMOTE does not always help. In high-dimensional, highly imbalanced settings, it can introduce noise. Measure, do not assume.

### Q16. When to adjust the decision threshold.

**Crisp:** Default 0.5 is arbitrary. Choose threshold based on business cost function or to hit a target precision or recall.

---

## Linear models

### Q17. What does a coefficient in logistic regression mean?

**Crisp:** `exp(beta)` is the odds ratio: a one-unit increase in the feature multiplies the odds of the positive outcome by `exp(beta)`.

**Trap:** This is odds ratio, not risk ratio. Confusing them is a common mistake.

### Q18. Assumptions of linear regression.

**Crisp:** Linearity, independence of errors, homoscedasticity, normality of residuals, no perfect multicollinearity.

**Trap:** The output being normal is not required; the residuals being normal is only needed for small-sample inference.

### Q19. Multicollinearity: detection and fix.

**Crisp:** Detect via VIF (variance inflation factor) > 5 or 10. Fix: drop redundant features, combine them, or use Ridge regression.

---

## Trees and ensembles

### Q20. Why do decision trees overfit?

**Crisp:** Trees recursively partition until leaves are pure, which memorizes training noise. Pruning, max depth, min samples per leaf all regularize.

### Q21. Bagging vs boosting intuition.

**Crisp:** Bagging (Random Forest) trains many independent models on bootstrapped samples and averages them; reduces variance. Boosting (XGBoost, LightGBM) trains models sequentially, each fixing the previous one's errors; reduces bias primarily, variance secondarily.

### Q22. Why does XGBoost handle missing values natively?

**Crisp:** At each split, it learns a default direction for missing values. The default direction is chosen to minimize the training loss.

### Q23. When to prefer Random Forest over XGBoost.

**Crisp:** When you need a robust baseline with little tuning, Random Forest is hard to beat. XGBoost usually wins with careful tuning but is more finicky.

### Q24. LightGBM vs XGBoost.

**Crisp:** Similar performance. LightGBM is faster on large datasets (leaf-wise growth, histogram-based splits) but can overfit on small datasets. XGBoost is more stable out of the box.

---

## Feature engineering

### Q25. Encoding categorical features: options and tradeoffs.

**Crisp:** One-hot (simple, sparse, blows up cardinality), target / mean encoding (leakage risk, use with CV), entity embeddings (learned in a neural net), frequency encoding (simple, preserves rarity signal).

### Q26. What is target leakage?

**Crisp:** Feature derived from the target that would not be available at prediction time. Classic example: "was contacted by collections" when predicting default, where collections only happens after default.

**Fix:** Audit every feature against a timeline. Ask: at the moment of prediction, would this be known?

---

## Neural networks

### Q27. Why use batch normalization?

**Crisp:** Normalizes activations per mini-batch, stabilizes training, allows higher learning rates. Also has a regularization effect via mini-batch noise.

### Q28. Vanishing and exploding gradients.

**Crisp:** Deep networks struggle when gradients shrink (vanish) or grow (explode) through many layers. Fixes: careful initialization (He, Xavier), normalization (batch, layer), skip connections (ResNet), gradient clipping.

### Q29. Why Adam is not always the best optimizer.

**Crisp:** Adam adapts per-parameter learning rates, which can hurt generalization in some regimes (vision classification often prefers SGD with momentum). Default for most tasks, but benchmark.

### Q30. Learning rate scheduling rationale.

**Crisp:** Start high to explore, decay to refine. Common schedules: step decay, cosine annealing, one-cycle (Smith).

---

## Recommenders and RL (rare in junior interviews, common in mid)

### Q31. Cold start problem.

**Crisp:** New users or items have no interaction history, so collaborative filtering fails. Solutions: content-based features, hybrid models, explicit preference elicitation at signup.

---

## Causal and experimentation (product DS interviews)

### Q32. Correlation vs causation in practice.

**Crisp:** Observational data can only show correlation unless you can identify a causal graph and satisfy conditions (backdoor, instrumental variables, randomization).

### Q33. Why randomize an A/B test?

**Crisp:** Randomization balances unobserved confounders in expectation, so the difference between groups is attributable to the treatment.

### Q34. When randomization is not possible.

**Crisp:** Use quasi-experimental methods: difference-in-differences, regression discontinuity, propensity score matching, instrumental variables, synthetic control.

---

## Modern ML (LLMs, generative)

### Q35. What is attention in transformers (intuitive)?

**Crisp:** Each output token is a weighted sum of input tokens, where weights are computed by comparing a query to keys. Lets the model selectively look at relevant context.

### Q36. Why do LLMs need fine-tuning at all if they are pretrained?

**Crisp:** Pretraining gives general language capability. Fine-tuning specializes behavior (style, format, domain), teaches tool use, or aligns with human preferences.

### Q37. RAG vs fine-tuning for domain knowledge.

**Crisp:** RAG retrieves relevant documents at query time; cheap to update, good for factual grounding. Fine-tuning bakes knowledge into weights; expensive to update, good for style or reasoning patterns. Often combine.

---

## Interviewer-favored follow-ups

- "How does your answer change if the data is non-IID?" (answer: blocked sampling, group-aware CV)
- "Walk me through how you would debug a model whose offline metric is 0.92 but online A/B is flat." (answer: data leakage, distribution shift, metric mismatch, treatment effect dilution, novelty effect)
- "A stakeholder says your model is biased against women. How do you investigate?" (answer: disaggregate metrics by subgroup, check training data representation, audit features for proxies, engage the stakeholder on the specific metric they care about)

Be ready to turn every theoretical answer into an operational one: "Here is how I would test this claim in code."
