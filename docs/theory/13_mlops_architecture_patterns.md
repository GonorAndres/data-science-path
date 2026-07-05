# MLOps Architecture Patterns

## What MLOps is

MLOps is the operational discipline for machine learning systems: versioning, reproducibility, deployment, monitoring, governance. It is not a tool; it is a set of practices borrowed from DevOps and adapted to the specific failure modes of ML.

## The ML lifecycle

1. **Problem framing.** Define the business metric and the model's role.
2. **Data.** Collect, validate, document.
3. **Features.** Engineer, store, version.
4. **Training.** Fit models, track experiments, produce artifacts.
5. **Validation.** Evaluate offline, check fairness, stress-test.
6. **Deployment.** Ship to production.
7. **Monitoring.** Watch for drift, errors, performance decay.
8. **Governance.** Log decisions; enable rollback; audit.

MLOps addresses all steps except the first, and arguably enables the first to be iterated faster.

## Levels of maturity

Google's MLOps level 0, 1, 2 framework is the most cited.

- **Level 0.** Manual process. Notebooks to production via copy-paste. Rarely reproducible.
- **Level 1.** Automated ML pipeline. CI for code, training pipelines run reliably, model versioning.
- **Level 2.** CI/CD for ML. Full automation including pipeline testing, canary rollouts, automated monitoring-triggered retraining.

Most Mexican mid-sized organizations are at Level 0 or 1 in 2026. Reaching Level 2 is a differentiator.

## Reproducibility

Every production model must be reproducible from source. This requires:

- **Code.** Git commit hash.
- **Data.** Version or snapshot of the training data.
- **Environment.** Pinned dependencies (requirements.txt, Dockerfile, lock files).
- **Configuration.** Hyperparameters, feature list, random seed.
- **Artifacts.** Trained model weights stored with metadata.

If the data scientist leaves, a new hire should be able to retrain the exact same model from these inputs.

Tools: DVC (data version control), MLflow (experiment tracking), Docker (environment), Pulumi or Terraform (infrastructure).

## Experiment tracking

Every training run should log:

- Hyperparameters.
- Input data version.
- Metrics (across folds and on holdout).
- Artifacts (model file, feature importance plot, confusion matrix).
- Tags (branch name, author, purpose).

Tools: MLflow (open-source, default), Weights & Biases, Neptune, ClearML.

Structure the experiments so you can filter by project, compare runs, and retrieve artifacts for any run.

## Feature stores

A feature store separates feature engineering from training and inference.

- **Offline store.** Historical features for training. Large, batch-computed.
- **Online store.** Real-time features for inference. Small, low-latency.
- **Point-in-time correctness.** When constructing a training set, features must reflect values known as of the label timestamp.

Tools: Feast, Tecton, Hopsworks, internal platforms (Uber's Michelangelo, Netflix's Metaflow).

Not every team needs a full feature store. Small teams can use a shared SQL schema with a feature catalog. Introduce a real feature store when feature reuse across teams becomes the bottleneck.

## Model registry

Central catalog of trained models. Each entry has:

- Name, version, stage (dev, staging, production, archived).
- Artifact link.
- Training run reference.
- Performance metrics.
- Governance metadata (approved by, date, constraints).

Tools: MLflow model registry, Sagemaker model registry, Vertex AI model registry.

Promotion flow: dev -> staging -> production requires an approval gate. Automated tests plus human sign-off.

## Deployment patterns

### 1. Batch scoring

Model runs on a schedule, writes predictions to a table. Downstream systems consume the table.

Simple, cheap, high throughput. Freshness measured in hours.

### 2. Online inference

Model wrapped in an API (FastAPI, Flask, TorchServe, Triton). Synchronous request-response.

Low latency (milliseconds). Infrastructure: load balancer, autoscaler, cache.

### 3. Streaming inference

Model consumes a Kafka or Kinesis stream, writes predictions downstream. Useful for event-driven systems (fraud, anomalies).

### 4. Edge inference

Model runs on-device (mobile, IoT). Requires model compression (quantization, distillation).

## Serving infrastructure

Modern options:

- **Cloud Run / Lambda / Cloud Functions.** Serverless. No ops. Cold starts can be a problem.
- **Kubernetes.** Flexible, complex, the enterprise default.
- **Managed ML services.** SageMaker Endpoints, Vertex AI Endpoints, Azure ML. Opinionated, fast to deploy.
- **Inference servers.** TorchServe, NVIDIA Triton, BentoML. Multi-model hosting, dynamic batching.

Choose based on:

- Team size (smaller teams prefer managed).
- Traffic profile (bursty vs steady).
- Latency budget.
- Cost sensitivity.

## Advanced deployment patterns

Beyond canary and blue-green, regulated and high-traffic environments use the following patterns to compare a candidate model against the current production model with minimum risk.

### Shadow deployment

The candidate model receives a copy of every production request but its predictions are **logged, not acted on**. The active production model continues to serve users. Engineering teams compare the two prediction streams offline: agreement rate, distribution shift, latency, error rate.

When to use: any model whose decisions are consequential and irreversible (claim approvals, fraud blocks, pricing). Shadow deployment is the cheapest way to discover that the candidate produces unacceptable predictions on real traffic before any user is affected.

Implementation. Mirror the request to the candidate either in the API gateway (a fan-out at the load balancer level) or asynchronously by replaying logs. Compare on a daily batch.

### Champion / challenger pattern

Two (or more) models serve real traffic, split by user or session. The current best model is the **champion**; the candidate is the **challenger**. Traffic is split (e.g., 90/10 or 50/50) and the business metric is measured per arm.

Statistical decision rule: declare a winner when the per-arm metric difference is significant at a pre-specified $\alpha$ (commonly 0.05) with sufficient power, using sequential testing (mSPRT, group sequential designs) to allow early stopping without inflating false positives. If the challenger wins, promote it to champion. If it loses, roll back automatically.

The rollback trigger should be coded explicitly into the deployment system: a metric crosses a threshold (e.g., the 95% CI lower bound of the challenger's metric falls below the champion's by more than X), the challenger traffic share goes to zero, and an alert fires.

### Bandit deployment

When you have multiple candidate models or model configurations and want to learn which performs best while minimizing regret, frame the problem as a multi-armed bandit. **Thompson sampling** (sample a posterior over each arm's reward, pick the argmax sample, observe the reward, update) and **epsilon-greedy** (with probability $\epsilon$ try a random arm, otherwise pick the best so far) are the standard online algorithms.

Use cases: ranking model selection, prompt-template selection for an LLM, hyperparameter selection in an automated retraining loop. Trade-offs vs champion/challenger: bandits adapt traffic in real time and minimize regret; champion/challenger gives cleaner statistical guarantees and is easier to explain to a regulator. For regulated decisions prefer champion/challenger; for non-consequential UI optimization prefer bandits. See `techniques/multi-armed-bandits` for the algorithms.

## Data lineage vs model lineage

Two distinct lineage graphs that operations teams conflate at their peril.

- **Data lineage.** What data, transformations, and joins produced this dataset, feature, or training table? Tools: OpenLineage, Marquez, dbt's exposure graph, Atlan, Great Expectations data docs. Used to answer "what does this column actually mean and where did it come from" and to assess blast radius when an upstream source breaks.
- **Model lineage.** What training run, code version, dataset version, and configuration produced this model artifact? Tools: MLflow run IDs, model registry metadata, DVC. Used to answer "if this model misbehaves in production, what specific commit and data slice produced it" and to enable exact reproduction for audit.

Both must point to immutable artifacts: a Git SHA for code, a content-hashed dataset, a configuration file checked into the model registry. A common gap is logging the dataset name without its version; six months later the dataset has been rewritten and the model can no longer be reproduced. Make every lineage edge point to a content-addressed artifact.

## LLMOps specifics

Operating LLM-based applications introduces concerns the classical ML stack does not address.

- **Token counters per request.** Log input tokens, output tokens, and cost per request, broken out by user, feature, and model. Cost is the primary operational metric; without this telemetry teams discover the bill at end of month.
- **Prompt versioning.** Store every prompt template under a content hash and a semantic tag (e.g., `summarize_claim_v3`). Bind every production response to the prompt hash that produced it. Without this, A/B-testing prompt changes is impossible and regressions are silent.
- **Evaluation pipelines.** Beyond classical accuracy, LLM evaluation uses LLM-as-judge for open-ended outputs. Calibrate the judge with a small human-labeled set (200 to 500 examples), measure the judge-vs-human agreement (Cohen's kappa), and re-run as the judge model updates. RAGAS-style metrics (faithfulness, answer relevance, context precision) for RAG applications.
- **KV-cache accounting.** Inference engines that reuse the KV cache across requests (vLLM, TGI) deliver dramatic cost reduction when the prompt prefix is shared. Monitor cache hit rate; a sudden drop indicates a prompt-template change broke prefix reuse.
- **Latency SLA tracking.** Token-streaming LLMs have two latency metrics: **time-to-first-token** (perceived responsiveness) and **inter-token latency** (perceived speed). Track both at P50, P95, P99. SLAs commonly target sub-second TTFT and 30 to 80 ms ITL for chat applications.
- **Safety and abuse monitoring.** Log inputs that trigger safety filters; track jailbreak attempt rates; fingerprint and rate-limit abusive callers.

## CI/CD for ML

CI for code: linting, unit tests, integration tests.

CD for ML: additional stages specific to models.

- **Data validation.** Schema checks, distribution checks on new data.
- **Model training.** Run the pipeline; record metrics.
- **Model validation.** Compare new model to current production on a holdout. If worse, fail the pipeline.
- **Staging deployment.** Smoke tests, canary test traffic.
- **Production deployment.** Blue-green or canary rollout with automatic rollback.

Tools: GitHub Actions, GitLab CI, Argo Workflows, Kubeflow Pipelines, Airflow.

## Monitoring

Three axes of monitoring:

### Operational

- Request count, latency P50/P95/P99.
- Error rate.
- Resource utilization (CPU, memory, GPU).

Standard SRE tools: Prometheus, Grafana, CloudWatch.

### Data quality

- Schema validation on incoming features.
- Null rates, out-of-range values.
- Distribution shift on each feature (PSI, KL divergence).

Tools: Evidently, Great Expectations, WhyLogs, Arize.

### Model performance

- Online metrics (CTR, conversion, revenue).
- Offline metrics replayed (compare prediction vs actual once labels arrive).
- Prediction distribution shift.
- Confidence calibration over time.

For models where labels arrive with delay (fraud confirmed weeks later), monitoring uses proxies (score distribution, override rate from downstream humans) until ground truth lands.

## Retraining

Automate retraining when:

- Performance drops below a threshold.
- Data drift exceeds a threshold.
- On a fixed schedule (weekly, monthly).

Pattern: pipeline triggers from a monitoring alert, runs training, passes through validation gates, promotes to production via standard flow.

## Governance and compliance

- **Model cards** (Mitchell et al. 2019). Public documentation.
- **Audit logs.** Every prediction, every override, who made changes.
- **Access control.** Not every engineer can push a model to production.
- **Rollback plan.** Every deployment has a one-click rollback path.
- **Incident response.** When a model causes a problem, a defined process kicks in.

Regulated industries (banking, insurance, healthcare) have more formal requirements: model validation committees, periodic re-certification, regulatory reporting.

## Cost management

- Track inference cost per 1K predictions.
- Track training cost per model run.
- Profile. Often 80 percent of serving cost comes from one underoptimized endpoint.
- Cache predictions where appropriate.
- Quantize and batch.

## Common mistakes

1. Deploying a notebook. Notebooks are not production code.
2. No monitoring. Models decay silently.
3. One-time data preprocessing. The pipeline works for training data; fails for production data with slightly different schema.
4. No rollback plan. First failure causes panic.
5. Retraining without revalidation. A worse model ships because nobody checks.
6. Feature skew. Training features and production features diverge over time.
7. Ignoring cost until the bill arrives.

## Linking to the portfolio

- `mlops-model-service`: the full reference implementation of these patterns (FastAPI, Docker, GitHub Actions, Cloud Run, monitoring).
- `technique/mlops-fundamentals`: the companion technique reference.
- `technique/model-monitoring-drift`: deep dive on drift detection.

## References

- Google Cloud, "MLOps: Continuous delivery and automation pipelines in machine learning".
- Huyen, *Designing Machine Learning Systems*, 2022.
- Kreuzberger, Kuhl, Hirschl, "Machine Learning Operations (MLOps): Overview, Definition, and Architecture", 2022.
- Mitchell et al., "Model Cards for Model Reporting", 2019.
- Sculley et al., "Hidden Technical Debt in Machine Learning Systems", NIPS 2015.
- Feast documentation, feast.dev.
- MLflow documentation, mlflow.org.

## Exercises

1. What is training-serving skew and how does a feature store prevent it?
2. Describe the champion / challenger deployment pattern. Which statistical test do you use to declare a winner?
3. You want to deploy a new model without risk to 100 percent of users. Describe three progressive rollout strategies in order of increasing risk exposure.
4. PSI $= 0.30$ is observed on a key input feature. What action do you take and why?
5. What is the difference between data drift and concept drift? Give one example of each in an insurance context.
- OpenLineage specification, openlineage.io.
- vLLM project, "Efficient Memory Management for Large Language Model Serving with PagedAttention", SOSP 2023.
