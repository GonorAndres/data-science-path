---
name: DS Curriculum Glossary
type: reference
updated: 2026-04-25
---

# Glossary

One-line definitions of acronyms and terms used across the theory library. Sorted alphabetically. Forward references (See: filename) point to essays whose final form lives elsewhere or is being written by a parallel agent.

- **ALE** (Accumulated Local Effects): plot of the local effect of a feature on the prediction, integrated along the feature's distribution; preferred over PDP when features are correlated. See: [06_model_interpretability_shap_lime.md](06_model_interpretability_shap_lime.md).
- **ATE** (Average Treatment Effect): population-average difference in potential outcomes, $E[Y(1) - Y(0)]$. See: [08_experimentation_causal_primer.md](08_experimentation_causal_primer.md).
- **ATT** (Average Treatment effect on the Treated): $E[Y(1) - Y(0) \mid T = 1]$, the average effect among units that actually received treatment. See: [08_experimentation_causal_primer.md](08_experimentation_causal_primer.md).
- **AUROC** (Area Under the Receiver Operating Characteristic): probability that a random positive scores higher than a random negative; threshold-free ranking quality. See: [01_ml_fundamentals_bias_variance.md](01_ml_fundamentals_bias_variance.md).
- **BH** (Benjamini-Hochberg): step-up procedure controlling the false discovery rate at level $q$ across multiple hypothesis tests. See: docs/theory/00_probability_estimation_primer.md.
- **BPR** (Bayesian Personalized Ranking): pairwise ranking loss for implicit-feedback recommender systems; optimizes the probability that an observed item ranks above an unobserved one. See: [12_recommender_system_foundations.md](12_recommender_system_foundations.md).
- **BSTS** (Bayesian Structural Time Series): state-space time-series model with explicit trend, seasonal, and regression components inferred via MCMC; the engine behind Google CausalImpact. See: [11_time_series_foundations.md](11_time_series_foundations.md).
- **CATE** (Conditional Average Treatment Effect): treatment effect conditional on covariates, $E[Y(1) - Y(0) \mid X = x]$; the target of uplift modeling and causal forests. See: [08_experimentation_causal_primer.md](08_experimentation_causal_primer.md).
- **CoT** (Chain of Thought): prompting pattern asking the model to produce intermediate reasoning steps before the final answer. See: [09_llm_foundations_2026.md](09_llm_foundations_2026.md).
- **CRPS** (Continuous Ranked Probability Score): proper scoring rule for probabilistic forecasts of a continuous variable; generalizes MAE to predictive distributions. See: [11_time_series_foundations.md](11_time_series_foundations.md).
- **CUPED** (Controlled-experiment Using Pre-Experiment Data): variance-reduction technique for A/B tests using a pre-period covariate to residualize the outcome. See: [08_experimentation_causal_primer.md](08_experimentation_causal_primer.md).
- **DML** (Double / Debiased Machine Learning): orthogonalization approach to causal inference that cross-fits nuisance functions to obtain $\sqrt{n}$-consistent treatment-effect estimates. See: [08_experimentation_causal_primer.md](08_experimentation_causal_primer.md).
- **DPO** (Direct Preference Optimization): RLHF alternative that fits a policy to preference pairs without an explicit reward model. See: [09_llm_foundations_2026.md](09_llm_foundations_2026.md).
- **ECE** (Expected Calibration Error): weighted average gap between confidence and accuracy across confidence bins. See: docs/theory/18_calibration_and_conformal.md.
- **ELBO** (Evidence Lower Bound): the objective maximized by variational inference; equals $\log p(x) - \mathrm{KL}(q \| p)$. See: docs/theory/00_probability_estimation_primer.md.
- **ESG** (Economic Scenario Generator): stochastic simulator of macro-financial paths used in capital and Solvency modeling. See: techniques/extreme-value-theory/README.md.
- **EVT** (Extreme Value Theory): asymptotic theory for the maximum of a sample; underpins GEV and peaks-over-threshold methods. See: techniques/extreme-value-theory/README.md.
- **FDR** (False Discovery Rate): expected proportion of false positives among rejected nulls; the BH procedure controls FDR. See: docs/theory/00_probability_estimation_primer.md.
- **GARCH** (Generalized Autoregressive Conditional Heteroskedasticity): time-series model in which conditional variance depends on past squared returns and past variances. See: [11_time_series_foundations.md](11_time_series_foundations.md).
- **GLM** (Generalized Linear Model): regression family with exponential-family responses, linear predictor, and link function. See: techniques/glm/README.md.
- **GP** (Gaussian Process): distribution over functions where every finite collection of values is jointly Gaussian; used for non-parametric Bayesian regression. See: techniques/gaussian-processes/README.md.
- **HTE** (Heterogeneous Treatment Effect): variation in treatment effect across covariate subgroups; same target as CATE. See: [08_experimentation_causal_primer.md](08_experimentation_causal_primer.md).
- **IID** (Independent and Identically Distributed): standard sampling assumption; violated by time-series, hierarchical, and network data. See: [03_cross_validation_strategies.md](03_cross_validation_strategies.md).
- **IPW** (Inverse Propensity Weighting): re-weight observational samples by $1 / P(T = t \mid X)$ to mimic randomization. See: [08_experimentation_causal_primer.md](08_experimentation_causal_primer.md).
- **IV** (Instrumental Variable): variable correlated with treatment but affecting the outcome only through treatment; identifies the effect under exclusion restriction. See: [08_experimentation_causal_primer.md](08_experimentation_causal_primer.md).
- **KL divergence** (Kullback-Leibler divergence): asymmetric measure of how far one distribution is from another, $\sum p \log(p/q)$. See: docs/theory/00_probability_estimation_primer.md.
- **LoRA** (Low-Rank Adaptation): parameter-efficient fine-tuning that learns a low-rank update $\Delta W = BA$ to a frozen weight matrix. See: [09_llm_foundations_2026.md](09_llm_foundations_2026.md).
- **MAE** (Mean Absolute Error): $\frac{1}{n}\sum |y_i - \hat{y}_i|$; robust to outliers, the optimal predictor under MAE is the conditional median. See: [01_ml_fundamentals_bias_variance.md](01_ml_fundamentals_bias_variance.md).
- **MAP** (Maximum A Posteriori): point estimate maximizing the posterior; equals MLE plus a regularizer corresponding to the prior. See: [02_regularization_and_generalization.md](02_regularization_and_generalization.md).
- **MDE** (Minimum Detectable Effect): smallest effect size an experiment can detect with given power and alpha. See: [08_experimentation_causal_primer.md](08_experimentation_causal_primer.md).
- **MLE** (Maximum Likelihood Estimation): parameter estimate maximizing the data likelihood. See: docs/theory/00_probability_estimation_primer.md.
- **MoE** (Mixture of Experts): architecture that routes each token to a small subset of expert sub-networks; gives parameter scale without proportional FLOP cost. See: [09_llm_foundations_2026.md](09_llm_foundations_2026.md).
- **NDCG** (Normalized Discounted Cumulative Gain): position-discounted ranking metric, normalized by the ideal ranking. See: [12_recommender_system_foundations.md](12_recommender_system_foundations.md).
- **NMS** (Non-Maximum Suppression): post-processing step in object detection that removes overlapping boxes by keeping the highest-scoring per cluster. See: [10_computer_vision_foundations.md](10_computer_vision_foundations.md).
- **ODP** (Over-Dispersed Poisson): GLM family used in stochastic chain-ladder reserving where variance scales with the mean. See: techniques/loss-reserving/README.md.
- **OLS** (Ordinary Least Squares): regression estimator minimizing squared residuals; the MLE under Gaussian noise. See: techniques/regression/README.md.
- **PEFT** (Parameter-Efficient Fine-Tuning): umbrella term for fine-tuning methods (LoRA, prefix-tuning, adapters) that update a small fraction of weights. See: [09_llm_foundations_2026.md](09_llm_foundations_2026.md).
- **PDP** (Partial Dependence Plot): expected prediction as a function of one feature, marginalizing over the rest. See: [06_model_interpretability_shap_lime.md](06_model_interpretability_shap_lime.md).
- **ICE** (Individual Conditional Expectation): per-observation curve underlying a PDP; reveals heterogeneity hidden by averaging. See: [06_model_interpretability_shap_lime.md](06_model_interpretability_shap_lime.md).
- **PFI** (Permutation Feature Importance): drop in score after randomly permuting one feature's values; biased under correlated features. See: [06_model_interpretability_shap_lime.md](06_model_interpretability_shap_lime.md).
- **PPO** (Proximal Policy Optimization): policy-gradient RL algorithm with a clipped surrogate objective; the workhorse of RLHF before DPO. See: [09_llm_foundations_2026.md](09_llm_foundations_2026.md).
- **PSI** (Population Stability Index): drift metric comparing two distributions across bins, $\sum (p_i - q_i) \log(p_i / q_i)$. See: [13_mlops_architecture_patterns.md](13_mlops_architecture_patterns.md).
- **QLoRA** (Quantized LoRA): LoRA fine-tuning on top of a 4-bit quantized base model; enables 65B fine-tuning on a single GPU. See: [09_llm_foundations_2026.md](09_llm_foundations_2026.md).
- **RAGAS** (Retrieval-Augmented Generation Assessment): evaluation framework for RAG systems with metrics for faithfulness, answer relevance, and context recall. See: [09_llm_foundations_2026.md](09_llm_foundations_2026.md).
- **RCT** (Randomized Controlled Trial): experiment where treatment assignment is randomized; the gold standard for causal identification. See: [08_experimentation_causal_primer.md](08_experimentation_causal_primer.md).
- **RLHF** (Reinforcement Learning from Human Feedback): alignment method that trains a reward model on human preferences and a policy via PPO. See: [09_llm_foundations_2026.md](09_llm_foundations_2026.md).
- **RoPE** (Rotary Position Embedding): positional encoding that rotates query and key vectors by position-dependent angles; enables length extrapolation. See: [09_llm_foundations_2026.md](09_llm_foundations_2026.md).
- **SRM** (Sample Ratio Mismatch): A/B test diagnostic where observed allocation deviates significantly from designed allocation; flags assignment bugs. See: [08_experimentation_causal_primer.md](08_experimentation_causal_primer.md).
- **SHAP** (SHapley Additive exPlanations): per-feature contribution to a prediction satisfying the Shapley efficiency, symmetry, and dummy axioms. See: [06_model_interpretability_shap_lime.md](06_model_interpretability_shap_lime.md).
- **SMD** (Standardized Mean Difference): difference in means scaled by pooled SD; balance diagnostic for matched / weighted observational studies. See: [08_experimentation_causal_primer.md](08_experimentation_causal_primer.md).
- **SPRT** (Sequential Probability Ratio Test): Wald sequential test that allows continuous monitoring with controlled error rates. See: [08_experimentation_causal_primer.md](08_experimentation_causal_primer.md).
- **SUTVA** (Stable Unit Treatment Value Assumption): assumption that one unit's treatment does not affect another's outcome and treatment is consistently defined. See: [08_experimentation_causal_primer.md](08_experimentation_causal_primer.md).
- **TFT** (Temporal Fusion Transformer): attention-based forecasting model with variable selection and quantile outputs. See: [11_time_series_foundations.md](11_time_series_foundations.md).
- **VC dimension** (Vapnik-Chervonenkis dimension): largest set size shatterable by a hypothesis class; classical capacity measure. See: [01_ml_fundamentals_bias_variance.md](01_ml_fundamentals_bias_variance.md).
- **VLM** (Vision-Language Model): multimodal model jointly trained on images and text (CLIP, SigLIP, LLaVA). See: [10_computer_vision_foundations.md](10_computer_vision_foundations.md).
- **WoE / IV** (Weight of Evidence / Information Value): log-odds-based encoding of a binary target into bins of a feature, with IV summarizing the feature's discriminatory power; canonical in credit and pricing. See: [05_feature_engineering_cookbook.md](05_feature_engineering_cookbook.md).
