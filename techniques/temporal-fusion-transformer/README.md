---
tags: [time-series, deep-learning, attention]
status: complete
created: 2026-04-25
updated: 2026-04-25
---

# Temporal Fusion Transformer (TFT)

## What it is

A deep-learning architecture for multi-horizon time-series forecasting that combines local sequential processing with global attention and produces native quantile forecasts. Proposed by Lim, Arik, Loeff and Pfister (2020). Used in production at retail, energy, and finance shops as the strongest single-model baseline when the training set has thousands of related series and a mixture of static, past-known, and future-known covariates.

## Why it matters

Three properties that no single competing method provides at once.

1. Handles three covariate types correctly. Static (store id, customer segment), past-only known (historical promotions, weather observed), and future-known (calendar holidays, scheduled price drops). Most deep-learning baselines (DeepAR, NHITS) collapse the distinction; TFT keeps separate input streams.
2. Probabilistic by construction. Quantile outputs ($q_{0.1}, q_{0.5}, q_{0.9}$) trained with the pinball loss; intervals come for free, no Monte Carlo postprocessing.
3. Interpretable. Variable selection networks emit per-input importance weights at every timestep; attention weights over time tell you which past timesteps the forecast leaned on.

## Architecture walkthrough

The forward path through TFT, in five blocks. Each block solves one specific problem the others do not.

### 1. Variable selection networks

For each input stream (static, past, future), a separate variable selection network computes a softmax over the input variables and a gated transformation of each. The network learns to suppress useless inputs by setting their softmax weight near zero. This is the input-gating mechanism.

Why it exists. Real production datasets carry tens to hundreds of candidate features and many are noise. Naive concatenation forces the encoder to learn-then-ignore; an explicit gate at the input lets the model spend capacity on signal.

### 2. Static covariate encoders

Static features (entity id embedding, market segment, region) are encoded once and broadcast through the network as conditioning vectors. Four separate static encoders feed: temporal variable selection, the LSTM initial state, the static enrichment layer, and the final feed-forward.

Why it exists. Static features should bias every step of temporal processing without being recomputed at every step. A single encoder that broadcasts to four distinct conditioning sites is more parameter-efficient than re-injecting them as time-replicated inputs.

### 3. Sequence-to-sequence LSTM encoder

A bidirectional or causal LSTM (encoder-decoder pair) processes the past and future inputs sequentially. This captures local temporal patterns: short-range autocorrelation, lagged covariate effects, recent shocks.

Why it exists. Attention is poor at very local patterns; a five-step exponential decay is captured cheaply by an LSTM and expensively by attention. The LSTM does the local work, leaving attention free for the long-range work.

### 4. Multi-head attention over time

After the LSTM, a multi-head attention block attends across all timesteps in the past window. Long-range dependencies (annual seasonality, holidays from a year ago, structural breaks) are picked up here.

Causal masking ensures the model never sees future-only known inputs from beyond the current forecasting timestep when producing its prediction for that step.

Why it exists. LSTMs degrade for ranges beyond around 50 timesteps; attention scales to hundreds. The two are complementary, not redundant.

### 5. Gating, add-and-norm, and quantile output

Gated residual networks (GRNs) wrap every nonlinear transformation: input is passed through a small MLP whose output is multiplied by a sigmoid gate and added back to the input. The gate lets the model bypass nonlinearities when they are not needed, simplifying optimization. Layer normalization stabilizes training.

The final layer is a feed-forward that outputs $Q$ quantile values per forecast horizon. Training uses the pinball loss summed over quantiles and horizons.

## Quantile outputs and the pinball loss

For a target $y$ and a predicted quantile level $q \in (0, 1)$ with predicted value $\hat{y}_q$, the pinball loss is

$$
\mathcal{L}_q(y, \hat{y}_q) = \max\{ q (y - \hat{y}_q),\ (q - 1)(y - \hat{y}_q) \}
$$

Minimizing this loss yields the conditional $q$-quantile of the target distribution. Summing across $q \in \{0.1, 0.5, 0.9\}$ produces calibrated point and interval forecasts.

The pinball loss is asymmetric: predicting too low at $q = 0.9$ is penalized 9x more than predicting too high. This is exactly the asymmetric loss profile actuarial reserving and inventory planning need.

## When to use TFT

| Scenario | Recommended | Why |
|----------|-------------|-----|
| 10 to 100 series, no covariates | ETS / ARIMA via statsforecast | TFT will overfit; statistical baselines win at small n |
| 100 to 5000 series, rich covariates, point forecasts only | LightGBM with lags + targets | Faster to train, easier to operate, matches TFT in MAE on most tabular-style series |
| 1000+ series, rich covariates, need quantile forecasts | TFT | Quantiles are native; LightGBM needs separate models per quantile |
| 1000+ series, need feature importance for stakeholder review | TFT | Variable selection weights are interpretable per series |
| Long history (years), strong yearly seasonality, simple covariates | Prophet or NHITS | TFT is overkill |
| Hierarchical forecasting (store/SKU/region) with reconciliation | LightGBM + MinT or HierarchicalForecast | TFT does not natively reconcile hierarchies |

Default heuristic for the portfolio: use LightGBM as the first-pass model and TFT as the challenger. Promote to TFT only when (a) you need probabilistic forecasts that LightGBM-quantile cannot produce calibrated, or (b) you have specific interpretability demands the variable-selection layer satisfies.

## Worked example with pytorch-forecasting

```python
import lightning.pytorch as pl
from pytorch_forecasting import (
    TimeSeriesDataSet,
    TemporalFusionTransformer,
    QuantileLoss,
)
from pytorch_forecasting.data import GroupNormalizer
import pandas as pd

# data: long-format dataframe with columns
# [series_id, time_idx, target, price, promo, holiday_flag, store_size]
df = pd.read_parquet("retail_sales.parquet")

max_encoder_length = 60
max_prediction_length = 14

training_cutoff = df["time_idx"].max() - max_prediction_length

training = TimeSeriesDataSet(
    df[df["time_idx"] <= training_cutoff],
    time_idx="time_idx",
    target="target",
    group_ids=["series_id"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=["store_size"],
    time_varying_known_reals=["price", "promo", "holiday_flag", "time_idx"],
    time_varying_unknown_reals=["target"],
    target_normalizer=GroupNormalizer(groups=["series_id"]),
    add_relative_time_idx=True,
    add_target_scales=True,
)

validation = TimeSeriesDataSet.from_dataset(
    training, df, predict=True, stop_randomization=True
)

train_loader = training.to_dataloader(train=True, batch_size=128, num_workers=4)
val_loader   = validation.to_dataloader(train=False, batch_size=256, num_workers=4)

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=1e-3,
    hidden_size=64,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=32,
    loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
    log_interval=10,
    reduce_on_plateau_patience=4,
)

trainer = pl.Trainer(
    max_epochs=30,
    accelerator="auto",
    gradient_clip_val=0.1,
    callbacks=[pl.callbacks.EarlyStopping(monitor="val_loss", patience=5)],
)
trainer.fit(tft, train_loader, val_loader)

# Variable importance over the validation set
raw_predictions = tft.predict(val_loader, mode="raw", return_x=True)
interp = tft.interpret_output(raw_predictions.output, reduction="sum")
# interp["encoder_variables"] -> importance of each past-known feature
# interp["decoder_variables"] -> importance of each future-known feature
# interp["static_variables"]  -> importance of static features
# interp["attention"]         -> attention weights across encoder timesteps
```

The interpretation block is what justifies the model in front of a stakeholder: "the forecast leaned 70 percent on price and promo over the last 14 days; weekly seasonality contributed the rest". A LightGBM SHAP plot answers a similar question but does not separate static, past, and future contributions.

## Common gotchas

1. `group_ids` must be present in both training and prediction sets. If you predict on a series id the model has never seen during training, the static embedding is uninitialized; the predict call will throw or silently zero out static information depending on the version. To forecast a brand-new series, either retrain from scratch or fall back to a global-only baseline.
2. Distinguish static from time-varying covariates explicitly. A "store size" that you accidentally encode as time-varying gets fed through the temporal pipeline and the model wastes capacity learning it is constant. The variable selection network will eventually figure it out but training is slower.
3. Past-known vs future-known is load-bearing. Holiday flags are future-known (you have a calendar), but raw weather observations are past-known (you only have history; future forecasts are a separate model). Misclassifying a past-known as future-known leaks future information at inference; misclassifying a future-known as past-known wastes the future signal.
4. Encoder length and prediction length interact. Encoder length should cover the longest seasonal cycle plus headroom; for daily retail data with weekly and yearly seasonality, 90 to 365 days is typical.
5. Hyperparameters are sensitive. `hidden_size`, `attention_head_size`, and `dropout` interact non-trivially. Use Optuna with a 20 to 50 trial budget; do not hand-tune.
6. Compute scales linearly in (batch * encoder_len * series). For very large catalogs, sample the training set per epoch rather than fit on the full dataset every epoch.
7. The QuantileLoss expects monotonic quantile predictions but does not enforce them. Crossing quantiles ($q_{0.5} > q_{0.9}$ for some series) is a sign of insufficient training; either train longer or apply a post-hoc isotonic projection.

## Resources

- `docs/theory/11_time_series_foundations.md` for the broader time-series context.
- `projects/timeseries-retail-forecast` for a LightGBM baseline and a TFT challenger on retail sales.
- `techniques/state-space-and-kalman/README.md` for the classical alternative when interpretability and small data dominate.

## References

- Lim, Arik, Loeff, Pfister, "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting", International Journal of Forecasting 2021.
- Salinas, Flunkert, Gasthaus, "DeepAR", International Journal of Forecasting 2020.
- pytorch-forecasting docs, pytorch-forecasting.readthedocs.io.
- Olivares et al., "Neural Hierarchical Forecasting" (NHITS), AAAI 2023.
- Hyndman, Athanasopoulos, *Forecasting: Principles and Practice*, 3rd ed., chapter 11.
