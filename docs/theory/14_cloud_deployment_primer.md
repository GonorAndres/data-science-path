# Cloud Deployment Primer for ML Services

## Why deployment literacy matters

A model in a notebook has zero business value. A model behind an API that serves real traffic has significant value. Deployment literacy is the bridge between data science and ML engineering, and it is one of the clearest signals to recruiters that you are production-ready.

## The three major clouds

In the Mexican market, GCP, AWS, and Azure all have substantial presence.

- **GCP.** Favored by newer startups, data-first companies. Tight BigQuery integration, Cloud Run is excellent. Popular at Nu, Bitso, Kavak.
- **AWS.** Largest market share. Banks, retail, multinationals. Ubiquitous at BBVA, Santander, Walmart MX.
- **Azure.** Strong in enterprise (Microsoft shops, insurance, manufacturing). Dominant at Deloitte Digital, Capgemini, many banks.

The portfolio defaults to GCP (Cloud Run for `mlops-model-service`) because the free tier plus simplicity make it the fastest way to ship a public demo. Skills transfer.

## Core cloud primitives

### Compute

- **Serverless functions.** GCP Cloud Functions, AWS Lambda, Azure Functions. Short-lived, event-driven, no infrastructure to manage.
- **Containerized serverless.** GCP Cloud Run, AWS App Runner, Azure Container Apps. Run a Docker container, auto-scale to zero.
- **Managed VMs / containers.** GCP GKE, AWS ECS/EKS, Azure AKS. More control, more complexity.
- **ML-specific managed.** Vertex AI endpoints, SageMaker endpoints, Azure ML endpoints. Purpose-built for model serving.

### Storage

- **Object storage.** GCS, S3, Azure Blob Storage. Durable, cheap, not filesystems.
- **Relational databases.** Cloud SQL, RDS, Azure SQL. Managed PostgreSQL, MySQL.
- **Data warehouses.** BigQuery, Redshift, Snowflake, Azure Synapse. Columnar, query-optimized.
- **Key-value.** Firestore, DynamoDB, Cosmos DB. Low-latency.

### Networking

- **Load balancers.** Route traffic across replicas.
- **CDN.** Cache static responses at edges.
- **VPC.** Private networks for secure backend traffic.

## Docker basics

Containers package code plus environment. Benefits: reproducible, portable, isolated.

### A minimal ML Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8080
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8080"]
```

Build:
```
docker build -t my-model:latest .
```

Run:
```
docker run -p 8080:8080 my-model:latest
```

### Size matters

- Start from slim or alpine base images.
- Use multi-stage builds: install build tools in stage 1, copy only what you need to stage 2.
- Avoid committing model weights to the image if they are large; load from object storage at startup.

## Deploying to Cloud Run (the fastest path)

Prerequisites: a GCP project, `gcloud` CLI installed, authenticated.

```
gcloud builds submit --tag gcr.io/PROJECT/my-model
gcloud run deploy my-model \
  --image gcr.io/PROJECT/my-model \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

Cloud Run returns a URL. Test:
```
curl https://my-model-XXX.run.app/predict -d '{"input": "..."}'
```

### Scaling

- Cloud Run autoscales from 0 to N instances based on traffic.
- Concurrency per instance (default 80) controls how many parallel requests one container handles.
- Set min-instances > 0 if cold starts matter for latency.

### Cost

- Pay per request (100 ms granularity) plus memory.
- First 2 million requests per month are free on GCP.
- A small demo costs pennies per month.

## Deploying to AWS (two common paths)

### Lambda + API Gateway

For small models (under 250 MB compressed), Lambda runs your code.

```
sam init
sam build
sam deploy --guided
```

For larger models, use container image support for Lambda (10 GB limit).

### ECS Fargate + Application Load Balancer

Run containers without managing servers. More flexible than Lambda, more configuration.

### SageMaker Endpoints

The opinionated path. Wrap the model in a SageMaker-specific format; endpoint is auto-managed, with autoscaling and monitoring.

## Deploying to Azure

### Azure Container Apps

Equivalent to Cloud Run. Serverless containers with auto-scaling.

### Azure ML Endpoints

Enterprise-targeted managed serving.

## CI/CD pipeline for a cloud-deployed model

Typical GitHub Actions or GitLab CI flow for `mlops-model-service`:

1. Push to branch.
2. Lint (ruff, black).
3. Unit tests (pytest).
4. Docker build.
5. Integration tests (spin up container, send test requests).
6. Push image to registry.
7. Deploy to staging.
8. Smoke test staging.
9. On approval, deploy to production (canary rollout).
10. Monitor; rollback on alert.

```yaml
# .github/workflows/deploy.yml (illustrative, abbreviated)
name: deploy
on: [push]
jobs:
  test-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -r requirements.txt
      - run: pytest
      - run: docker build -t $IMAGE .
      - run: gcloud auth configure-docker
      - run: docker push $IMAGE
      - run: gcloud run deploy my-model --image $IMAGE
```

## Secrets management

Never commit secrets to git. Cloud-native options:

- **GCP Secret Manager.** Access via service account.
- **AWS Secrets Manager or Parameter Store.** Similar.
- **Azure Key Vault.** Similar.

In CI, inject secrets as environment variables from a CI secret store (GitHub Actions secrets, GitLab protected variables).

## Monitoring in the cloud

- **Logs.** Stackdriver (GCP), CloudWatch (AWS), Azure Monitor. Every cloud logs container stdout automatically.
- **Metrics.** Request count, latency, error rate. Built in.
- **Alerts.** On any metric threshold.
- **Custom metrics.** Emit from the app (Prometheus format); scrape into the cloud monitoring system.

For model-specific metrics (drift, prediction distribution), add a service that computes these and emits them.

## Cost control

- **Set budgets.** Every cloud has a budget feature with email alerts.
- **Tag resources.** By project, by environment. Filter billing by tag.
- **Autoscaling.** Scale down when idle. Cloud Run scales to zero by default; other services often do not.
- **Reserved instances / committed use.** For predictable workloads, commit to 1 or 3 years for discounts.
- **Spot / preemptible.** For training, not for serving.

## Free-tier tips (for portfolio projects)

- **GCP.** $300 free credit for 90 days; always-free tier includes 2M Cloud Run requests, 1 GB storage.
- **AWS.** 12 months of free tier (750 hours EC2 t2.micro, 1M Lambda requests).
- **Azure.** $200 credit for 30 days.
- **Streamlit Cloud.** Free tier for small public apps.
- **HuggingFace Spaces.** Free Gradio and Streamlit hosting with GPU for some tiers.
- **Render.** Free web services (cold-starts).

For the portfolio, a zero-dollar monthly cost is achievable.

## Security essentials

- **IAM principle of least privilege.** Service accounts have only the permissions needed.
- **TLS.** Cloud services provide HTTPS by default; do not disable.
- **Authentication.** For private APIs, require authentication (API keys, OAuth, IAP).
- **DDoS protection.** Cloud Load Balancers have it built in.
- **Dependency scanning.** Snyk, Dependabot, GitHub's native.

## Common mistakes

1. Committing secrets to git. Once pushed to a public repo, they are compromised.
2. Open object storage buckets. S3 and GCS buckets should be private unless they serve public data.
3. Deploying with root container user. Use a non-root user for defense in depth.
4. No rate limiting. An unthrottled public endpoint is a billing bomb waiting to happen.
5. No idle-scaling. Always-on VMs cost more than serverless for bursty workloads.

## Linking to the portfolio

- `mlops-model-service`: deploys to Cloud Run with all these patterns.
- `llm-rag-qa`: local-first with Ollama; optional cloud deployment documented.
- `streamlit-exec-dashboard`: Streamlit Cloud deployment.

## GCP free-tier note (currency caveat)

The "first 2 million Cloud Run requests per month free" claim above is accurate as of the last verification (Q1 2026) but Google has changed Cloud Run free-tier quotas multiple times. Before relying on the always-free tier for a portfolio demo or production workload, check the live pricing page at cloud.google.com/run/pricing. Free-tier limits apply per billing account, not per project; deploying many demos under one account can exhaust the quota. The same caveat applies to BigQuery (1 TB queries / month free), GCS (5 GB standard storage), and Cloud Functions (2M invocations / month). Treat any free-tier number in this document as a starting estimate, not a contract.

## GPU serving

CPU inference is sufficient for tabular models, classical NLP, and small transformers (under roughly 500M parameters with int8). GPU serving becomes worthwhile when one of these conditions holds:

- Model is over 7B parameters (LLM territory).
- Latency SLA is under 200 ms for a model that does not fit in CPU SIMD throughput.
- Throughput requires batched parallelism that CPU cannot deliver (recommender ranking, real-time CV).

Three production-grade options:

- **Cloud Run with GPU.** As of late 2024 GCP added L4 GPU support to Cloud Run (us-central1, europe-west1, asia-southeast1 initially). Scales to zero, pay per 100 ms of GPU time, attach 1 L4 per instance with up to 24 GB of GDDR6. Suitable for 7B-class quantized LLMs and most CV workloads. Cold start with a 7B model is 30 to 90 seconds because the model must load into VRAM, so plan for either min-instances or lazy-loading patterns.
- **Vertex AI custom containers with GPU.** Endpoints support T4, L4, A100, H100. More configuration, true autoscaling on QPS or GPU utilization, integrated with Vertex Model Registry, Feature Store, and Pipelines. Right choice when you need per-endpoint authentication, traffic splitting between model versions, or A100 / H100 capacity.
- **GKE with GPU node pools.** Full control. Use when you need custom batching servers (Triton Inference Server, vLLM, TGI), multi-GPU tensor parallelism, or specific CUDA driver versions.

Cost framing: an L4 on Cloud Run runs roughly $0.50 to $0.65 per hour active (2026 pricing, verify). At 1 request/second for an LLM, GPU saves roughly 5x to 20x on latency vs CPU; whether that justifies cost depends on the SLA and traffic shape. Below 100 requests/hour, CPU + quantization usually wins on total cost.

## Cold-start mitigation in depth

Setting `min-instances > 0` is the blunt fix. When that is too expensive (idle GPU billing, $200 to $500 / month for one always-on L4), use these patterns instead or in combination:

- **Lazy model loading.** Load the model into memory at first request, not at container startup. The first user pays the cold-start tax; subsequent users on the same instance hit a warm cache. Combine with a `/warmup` endpoint that the deployment pipeline calls right after rollout to pre-load before real traffic arrives.
- **Container startup probes.** Cloud Run and Kubernetes both expose startup-probe configuration. By default the platform considers a container ready as soon as the port is open, which can route traffic before the model is loaded. Use a startup probe that hits a `/ready` endpoint returning 200 only after the model is in memory. This delays "ready" but prevents the first 5 to 30 requests from timing out.
- **Provisioned concurrency on Vertex AI.** Vertex AI Endpoints support a minimum-replica setting plus `automatic_resources.min_replica_count`. Setting this to 1 keeps one warm replica without paying for autoscaler-burst capacity. Cheaper than Cloud Run min-instances for GPU workloads because Vertex bills at endpoint granularity, not per request.
- **Model server warmup.** vLLM, TGI, and Triton all support warmup configurations that run a synthetic forward pass at startup. This compiles fused kernels, warms KV-cache allocators, and surfaces VRAM-OOM errors before traffic arrives.
- **Two-tier deployment.** Keep one tiny CPU instance always warm to handle health checks and trivial routing; cold-start the GPU container only when a real prediction request arrives. Common pattern for LLM demos with bursty traffic.

## OpenTelemetry and distributed tracing

Logs answer "what happened?" Metrics answer "how often?" Traces answer "where did the time go?" For a multi-service ML pipeline (ingest, feature lookup, model, post-processing, downstream service), traces are the only way to debug latency without guesswork.

OpenTelemetry (OTel) is the vendor-neutral standard. The flow:

1. Instrument the FastAPI service with the OTel SDK; spans are created per request and per outbound call.
2. Propagate trace context (W3C `traceparent` header) across service boundaries so all spans for one user request share a trace ID.
3. Export spans to Cloud Trace (GCP), AWS X-Ray, or Azure Monitor via the OTel Collector.

Minimal FastAPI instrumentation:

```python
from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(CloudTraceSpanExporter())
)

app = FastAPI()
FastAPIInstrumentor.instrument_app(app)
RequestsInstrumentor().instrument()

tracer = trace.get_tracer(__name__)

@app.post("/predict")
def predict(payload: dict):
    with tracer.start_as_current_span("feature_lookup"):
        features = lookup_features(payload["user_id"])
    with tracer.start_as_current_span("model_inference"):
        result = model.predict(features)
    return result
```

Structured logging with trace IDs ties logs to traces. Use `python-json-logger` or `structlog` and inject the current span's trace ID into every log record:

```python
span = trace.get_current_span()
ctx = span.get_span_context()
log.info("prediction.completed", trace_id=format(ctx.trace_id, "032x"), latency_ms=elapsed)
```

In Cloud Logging, the `logging.googleapis.com/trace` field auto-links each log line to the corresponding trace, and the Cloud Trace UI shows logs inline with spans.

Why this matters: a multi-service request that takes 2.5 seconds may spend 200 ms in your model and 2.3 seconds in a feature lookup that calls a misconfigured BigQuery view. Without traces you blame the model; with traces you fix the actual bottleneck in five minutes.

## Load testing for ML services

Load testing answers two production questions: can the service handle expected peak QPS, and what autoscaling thresholds keep tail latency within SLA?

Two open-source tools dominate:

- **locust** (Python). Write user behavior as Python classes, run a swarm of simulated users, get real-time UI. Easiest for ML teams already in Python.
- **k6** (JavaScript, Go binary). Higher throughput per generator node, better suited for high-QPS scenarios. Cloud-friendly (k6 Cloud).

Minimal locust file for a model endpoint:

```python
from locust import HttpUser, task, between

class ModelUser(HttpUser):
    wait_time = between(0.5, 2.0)

    @task
    def predict(self):
        self.client.post("/predict", json={"feature_a": 1.2, "feature_b": "x"})
```

Run with `locust -f locustfile.py --host https://my-model.run.app`, ramp users from 1 to target QPS, watch the live dashboard.

Metrics to track during the load test:

- **p50, p95, p99 latency.** p50 tells you the typical user; p99 tells you the worst-case experience and is what the SLA is usually written against. p99 / p50 ratio under 3 is healthy; over 5 indicates queueing or GC pauses.
- **Error rate.** Should be under 0.1 percent. A jump above 1 percent under load points to OOM kills, container restarts, or upstream throttling.
- **GPU utilization.** From `nvidia-smi` or Cloud Monitoring agent. Below 30 percent suggests batching is undersized; above 95 percent sustained suggests you are memory-bound or compute-saturated and need horizontal scaling.
- **CPU and memory utilization** per instance. Sets the autoscaling target.
- **Concurrency per instance.** Cloud Run default is 80; for a heavy model 4 to 16 may be the right ceiling.

How to convert load-test results into autoscaling settings:

1. Find the "knee" of the latency curve: the QPS at which p95 starts climbing past your SLA. Call this `qps_safe_per_instance`.
2. Set Cloud Run concurrency such that one instance running at that concurrency does not exceed `qps_safe_per_instance`. For a model with 100 ms p95 and concurrency 4, one instance handles roughly 40 QPS.
3. Set autoscaling target to scale up at 60 to 70 percent of capacity (keep 30 to 40 percent headroom for traffic spikes).
4. Estimate min-instances from the rate of cold-start-tolerable requests at off-peak. If even one cold start violates SLA, min-instances >= 1.
5. Set max-instances to roughly 2x peak QPS / `qps_safe_per_instance`, so the autoscaler can absorb a doubled spike before throttling.

Repeat the load test after every model-version change, every infrastructure change (instance size, GPU swap), and quarterly otherwise.

## References

- Google Cloud Architecture Center, cloud.google.com/architecture.
- AWS Well-Architected Framework, aws.amazon.com/architecture/well-architected.
- Microsoft Cloud Adoption Framework, learn.microsoft.com/azure/cloud-adoption-framework.
- Hightower, "Kubernetes the Hard Way", github.com/kelseyhightower/kubernetes-the-hard-way.
- Cloud Run documentation, cloud.google.com/run/docs.
- OpenTelemetry documentation, opentelemetry.io/docs.
- Sridharan, *Distributed Systems Observability*, O'Reilly 2018.
- Beyer et al., *Site Reliability Engineering*, Google / O'Reilly 2016 (latency, SLO chapters).
- locust documentation, docs.locust.io.
- k6 documentation, k6.io/docs.

## Exercises

1. Your Cloud Run service has p99 latency of 8 seconds on the first request and 120 ms on subsequent requests. What is likely causing this and how do you fix it?
2. Compare Cloud Run and Vertex AI Endpoints for serving a 7B-parameter LLM. Which do you choose and why?
3. What is the difference between a blue-green deployment and a canary deployment for an ML model?
4. Describe how you would instrument a FastAPI ML service to emit structured logs and traces to Google Cloud Observability.
5. Your model receives 50,000 requests per day with peaks of 500 req/min. Configure autoscaling settings for Cloud Run and justify each parameter.
