# Anomaly Visual Cues

## Why highlight anomalies

Executives ask: "What happened yesterday?" They do not want to stare at a chart for 10 minutes. They want a callout.

## Simple anomaly detection

Rolling mean +/- 2 sigma:

```python
window = 28
df["roll_mean"] = df["revenue"].rolling(window, min_periods=7).mean()
df["roll_std"] = df["revenue"].rolling(window, min_periods=7).std()
df["z"] = (df["revenue"] - df["roll_mean"]) / df["roll_std"]
df["anomaly"] = df["z"].abs() > 2
```

Simple, interpretable, robust for roughly normal distributions.

## More robust: MAD

Median absolute deviation is outlier-resistant:

```python
df["roll_med"] = df["revenue"].rolling(window).median()
df["mad"] = (df["revenue"] - df["roll_med"]).abs().rolling(window).median()
df["anomaly"] = (df["revenue"] - df["roll_med"]).abs() > 3 * 1.4826 * df["mad"]
```

## Visualizing anomalies

- **Highlight points.** Scatter markers on anomalous days, larger and colored.
- **Annotations.** Label the biggest anomaly with its date and value.
- **Callout panel.** Side panel listing "3 anomalies in the last 7 days" with links to detail.

Avoid alerting on every minor fluctuation; the threshold should be set such that 1 to 5 anomalies show up per month, not per day.

## Context

A single day's anomaly is interesting; whether it indicates a problem requires context:

- Day-of-week (Mondays and Fridays vary).
- Holidays.
- Marketing campaigns active.
- Known system events.

The dashboard should show these as annotations alongside anomalies so the exec can correlate.

## Beyond statistical

- **Isolation Forest or One-Class SVM** for multivariate anomaly detection.
- **Prophet with interval bounds.** Anomalies are points outside the 80 or 95 percent interval.
- **Business rules.** "More than 20 percent drop in revenue on a non-holiday weekday is always an anomaly."

For the exec dashboard, simple rolling +/- 2 sigma is enough; it catches the meaningful outliers.

## Cross-links

- `02_kpi_design.md`
- `04_dashboard_performance.md`
- `docs/theory/11_time_series_foundations.md`
