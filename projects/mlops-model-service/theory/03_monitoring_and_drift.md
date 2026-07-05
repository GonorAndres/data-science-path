# Monitoring and Drift Detection

## Why monitoring ML is different

Traditional web services monitor uptime, latency, and error rate. ML services add three new failure modes:

1. **Data drift.** Input distribution shifts. Model still runs but predicts on data it was not trained for.
2. **Concept drift.** The target relationship changes (what counted as "positive" a year ago may read differently now).
3. **Silent accuracy decay.** Neither the server nor the client notices; only downstream users see degraded decisions.

A service that is healthy at the operational layer can silently be broken at the ML layer. Monitoring must cover both.

## Three monitoring axes

### Operational

- **Request count** per endpoint per status code.
- **Latency histograms** (P50, P95, P99).
- **Error rate** (4xx, 5xx).
- **CPU, memory, container count.**

Standard SRE metrics; Cloud Run emits many of these for free.

### Data quality

- **Input length distribution.** Histogram of characters / tokens per request.
- **Input null or empty rate.** Should be zero; spike means upstream bug.
- **Schema validation failure rate.** Spike means contract mismatch.
- **Language distribution.** If you detect language, track the ES vs EN split; a spike away from training-time balance is drift.

### Model performance

- **Predicted class distribution.** Should match the expected production distribution (roughly 50 percent positive, 30 percent neutral, 20 percent negative). Spikes indicate a possible drift or bug.
- **Confidence distribution.** Lower mean confidence over time can indicate drift.
- **Online vs offline gap.** When ground truth arrives (user feedback, thumbs up/down), compute rolling accuracy and compare to the offline benchmark.

## This project's monitoring implementation

Metrics exposed in Prometheus text format at `/metrics`:

- `http_requests_total{endpoint, status}`
- `http_request_duration_seconds{endpoint}` (histogram)
- `predictions_total{label}` (counter by predicted class)
- `input_length_chars` (histogram)

Structured logs emit each request with:

- `request_id` (propagated from header or generated).
- `path`, `status`, `duration_ms`.
- On failure, the exception traceback.

Logs land in Cloud Logging; Prometheus scraping can be configured via a separate Prometheus server or GCP Managed Prometheus.

## Drift detection (stubbed)

Full drift detection requires a reference distribution (training data) and a monitoring window (recent production data). Compute a statistical distance:

- **Population Stability Index (PSI).** Industry standard for tabular. Bins the feature, compares bin proportions.
- **Kolmogorov-Smirnov.** For continuous features.
- **KL / JS divergence.** For categorical or probability distributions.

Decision thresholds (heuristic):

- PSI < 0.1: no significant drift.
- 0.1 to 0.25: moderate drift; investigate.
- Greater than 0.25: significant drift; retrain or intervene.

For text inputs, drift is usually on derived features (length, language, topic distribution), not on the raw text itself.

### Implementation sketch

A background job (Cloud Scheduler -> Cloud Run job, or a GitHub Actions cron):

1. Pulls yesterday's logs from Cloud Logging.
2. Computes distribution of input length, language, predicted class.
3. Compares to the training reference distribution stored in GCS.
4. Computes PSI per feature.
5. Writes a daily drift report to GCS.
6. Alerts if any PSI exceeds threshold.

`evidentlyai` and `whylabs` are off-the-shelf tools. For this portfolio, a custom script is enough.

## Alerting

Two levels:

### Operational alerts (Cloud Run native)

- P95 latency > 1 second for 5 minutes.
- Error rate > 5 percent for 5 minutes.
- Container restart loop.

Sent to Cloud Monitoring, which forwards to email / Slack / PagerDuty.

### ML alerts (custom)

- Predicted class distribution shifted by more than 10 percentage points from the reference.
- Mean confidence dropped by more than 0.1 from the reference.
- Input language distribution shifted significantly.

These fire on the daily drift report and send a Slack message or email.

## Ground-truth feedback loop

If the service is used in an app where humans provide feedback ("was this prediction correct? thumbs up/down"), a feedback endpoint captures it. Over time, a ground-truth dataset builds up automatically.

A weekly offline job computes accuracy on the new feedback and compares to the offline benchmark. When the gap widens, retraining is triggered (or at minimum flagged).

## Retraining triggers

Three retraining triggers:

1. **Scheduled.** Retrain monthly on accumulated data. Simple baseline.
2. **Performance-triggered.** Retrain when online performance drops below threshold.
3. **Drift-triggered.** Retrain when input distribution has shifted significantly.

This project does not implement retraining automation; the NLP project is the training pipeline. The hook between them would be a GitHub Actions workflow triggered by a drift or performance alert.

## What to watch per stakeholder

| Stakeholder | Cares about |
|-------------|-------------|
| SRE | Latency, error rate, uptime |
| Data Scientist | Predicted class distribution, confidence drift |
| Product Manager | Feedback-loop accuracy, adoption |
| Legal / Compliance | Per-language fairness over time |
| Finance | Cost per 1k predictions |

Dashboards slice the same metrics by audience.

## Common mistakes

1. Monitoring only operational metrics. The model can silently degrade while the service is healthy.
2. Thresholds pulled from papers, not from the actual production distribution. Always set thresholds from your own reference.
3. Alert fatigue. Too many alerts get ignored. Only alert on actionable conditions.
4. No runbook. When an alert fires, what does the on-call do? Write it down.
5. Metrics cardinality explosion. Labels with high cardinality (request IDs as labels, user IDs as labels) blow up Prometheus memory. Keep label cardinality small.

## Cross-links

- `01_serving_patterns.md`: where metrics are collected in the request path.
- `02_ci_cd_for_ml.md`: how model validation fits before deploy.
- `04_cost_and_latency.md`: the performance baseline you monitor against.
- `docs/theory/13_mlops_architecture_patterns.md`: broader monitoring patterns.
