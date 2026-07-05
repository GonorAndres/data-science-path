---
tags: [optimization, hyperparameters, optuna]
status: draft
created: 2026-04-23
updated: 2026-04-23
---

# Hyperparameter Optimization with Optuna

## What it is

Optuna is a hyperparameter optimization framework. You define an objective function (given hyperparameters, return a score), and Optuna searches for the best hyperparameters using bayesian optimization, random search, or TPE (Tree-structured Parzen Estimator).

## Why it matters

Manual tuning is slow and biased. Grid search does not scale. Random search scales but does not learn. Optuna (and similar: Hyperopt, Ray Tune, Weights & Biases Sweeps) learns from past trials to propose better ones.

## When to use it

- Tuning gradient boosting (LightGBM, XGBoost) hyperparameters.
- Neural net learning rate, weight decay, architecture choices.
- Any expensive-to-evaluate model with > 3 hyperparameters.

Skip for quick experiments (use defaults) or when compute is unlimited (just grid search).

## Concepts

- **Trial.** One evaluation of the objective.
- **Study.** A collection of trials sharing a search space.
- **Sampler.** How new trials are proposed (TPE default, CMA-ES, grid, random).
- **Pruner.** Stops unpromising trials early (ASHA, Hyperband).
- **Storage.** Persistent SQLite or PostgreSQL backend for distributed studies.

## Gotchas

- Suggest_* calls define the search space; always the same in every trial.
- Pruning requires intermediate reporting during the trial.
- Conditional hyperparameters supported but increase search space complexity.

## Resources

- Optuna docs: optuna.readthedocs.io.
- Akiba et al. "Optuna: A Next-generation Hyperparameter Optimization Framework". 2019.
