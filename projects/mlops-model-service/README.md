# mlops-model-service

## Resumen (ES)

**Problema:** empaquetar el modelo de `nlp-reviews-multilingual` como un servicio de inferencia en produccion con todo el stack MLOps de un equipo real: FastAPI, Docker, GitHub Actions CI/CD, despliegue a GCP Cloud Run, monitoreo basico (metrics + drift), y manejo de secrets.

**Metodo:** FastAPI sirve el modelo fine-tuneado, contenedor Docker multi-stage, pipeline CI en GitHub Actions que corre linters + tests + build + deploy, endpoint `/metrics` para Prometheus, log estructurado con trace IDs.

**Resultado objetivo:** URL publica funcional en Cloud Run, badge de CI verde en el README principal, latencia P95 menor a 200 ms en CPU, costo menor a $5 USD/mes bajo trafico de demo.

## Summary (EN)

**Problem:** wrap the `nlp-reviews-multilingual` model in a production-style inference service with the full MLOps stack a real team would use: FastAPI, Docker, GitHub Actions CI/CD, deploy to GCP Cloud Run, basic monitoring (metrics + drift), and secrets management.

**Method:** FastAPI serves the fine-tuned model; multi-stage Docker build; GitHub Actions pipeline runs linters + tests + build + deploy; `/metrics` endpoint for Prometheus; structured logging with trace IDs.

**Target result:** functional public URL on Cloud Run, green CI badge on root README, P95 latency under 200 ms on CPU, cost under $5 USD/month at demo traffic.

---

## Business framing

A data science team trains a model. Shipping it is a different skill. This project demonstrates that skill end-to-end with no hand-waving: every step from source code to live URL is automated and reproducible.

The service wraps `nlp-reviews-multilingual`. Any review text posted to `/predict` returns a sentiment label, confidence, and per-class probability. The service also exposes `/health`, `/metrics`, and request tracing.

## Why this project

Deployment literacy is the single clearest signal to ML-engineer-leaning recruiters. A portfolio with a trained model but no deployment reads as incomplete. Adding deploy infrastructure:

- Proves you can ship.
- Demonstrates Docker, FastAPI, GitHub Actions, and Cloud Run in one project.
- Creates a live URL recruiters can click.
- Pairs naturally with a separately-built model (showcases separation of concerns).

## Architecture

```
+-------------------+   HTTPS   +------------------+
|   Client          +---------->+   Cloud Run     |
| (curl, web app)   |           |   (container)   |
+-------------------+           +--------+---------+
                                         |
                                         | loads
                                         v
                                +------------------+
                                |  HuggingFace     |
                                |  model weights   |
                                |  (bundled in img)|
                                +------------------+

Observability:
   /health       simple readiness probe
   /metrics      Prometheus-format metrics
   structured logs -> Cloud Logging
   request IDs   propagated via X-Request-ID
```

## Repository structure

```
mlops-model-service/
├── README.md
├── theory/
│   ├── 00_problem_framing.md
│   ├── 01_serving_patterns.md
│   ├── 02_ci_cd_for_ml.md
│   ├── 03_monitoring_and_drift.md
│   └── 04_cost_and_latency.md
├── app/
│   ├── __init__.py
│   ├── main.py                # FastAPI app
│   ├── schema.py              # Pydantic request/response
│   ├── model.py               # model loader wrapper
│   ├── metrics.py             # Prometheus counters, histograms
│   └── logging_setup.py       # structured JSON logging
├── tests/
│   ├── __init__.py
│   ├── test_api.py            # black-box tests with TestClient
│   └── test_schema.py
├── Dockerfile                 # multi-stage build
├── .dockerignore
├── requirements.txt
├── Makefile
├── cloudbuild.yaml            # optional: GCP Cloud Build alternative
└── scripts/
    ├── load_test.sh           # simple k6 or hey script
    └── deploy_cloud_run.sh    # one-shot manual deploy
```

The CI workflow lives at the repo root (`/.github/workflows/mlops-model-service.yml`) so GitHub Actions picks it up.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | /predict | Classify one review |
| POST | /predict/batch | Classify up to 100 reviews |
| GET | /health | Readiness probe |
| GET | /metrics | Prometheus metrics |
| GET | /docs | OpenAPI autogen docs |

Example:

```bash
curl -X POST https://SERVICE-URL/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Me encanto el producto!"}'
```

Response:

```json
{
  "label": "positive",
  "confidence": 0.94,
  "probs": {"negative": 0.02, "neutral": 0.04, "positive": 0.94},
  "model_version": "1.0.0",
  "request_id": "c1a4..."
}
```

## Quickstart (local)

```bash
cd projects/mlops-model-service
pip install -r requirements.txt

# Copy the trained model from the NLP project
make copy-model

# Run locally
uvicorn app.main:app --reload --port 8080

# Test
curl http://localhost:8080/health
curl -X POST http://localhost:8080/predict -H 'Content-Type: application/json' -d '{"text": "El producto es excelente"}'
```

## Quickstart (Docker)

```bash
docker build -t nlp-reviews-service .
docker run -p 8080:8080 nlp-reviews-service
```

## Deploy to Cloud Run

```bash
export PROJECT=your-gcp-project
export SERVICE=nlp-reviews
export REGION=us-central1
bash scripts/deploy_cloud_run.sh
```

The script does:

```bash
gcloud builds submit --tag gcr.io/$PROJECT/$SERVICE
gcloud run deploy $SERVICE \
  --image gcr.io/$PROJECT/$SERVICE \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --min-instances 0 \
  --max-instances 10 \
  --timeout 60s
```

Cloud Run returns a URL. Put it in the main repo README once live.

## CI/CD

GitHub Actions pipeline (see `.github/workflows/mlops-model-service.yml`):

1. Lint with ruff.
2. Type-check with mypy (optional).
3. Run pytest.
4. Build Docker image.
5. On push to `main`, push to Container Registry.
6. On push to `main`, deploy to Cloud Run.

Secrets required (set in GitHub repo settings):

- `GCP_SA_KEY`: service account JSON with Cloud Run Admin + Storage Admin.
- `GCP_PROJECT`: GCP project ID.

## Monitoring

- **Latency histogram** (`http_request_duration_seconds` by endpoint).
- **Request count** (`http_requests_total` by endpoint and status).
- **Prediction counts by class** (`predictions_total` labeled by predicted class).
- **Input length histogram** (tokens in the request).
- **Structured logs** with request ID for tracing.

Drift monitoring (stubbed): on every request, emit the input length and predicted class. A downstream script (not implemented here, sketched in `theory/03_monitoring_and_drift.md`) reads these daily and compares to a reference distribution.

## Cost

Cloud Run:

- 2 million requests per month free.
- Estimate at 10,000 predictions/month with ~2s warm-up and 150 ms per request: under 3 USD/month.
- If min-instances = 0, cold-start is ~5 seconds. If you need steady warm performance, set min-instances = 1 (~30 USD/month).

## Decisions and trade-offs

| Decision | Alternative | Why we chose this |
|----------|-------------|-------------------|
| Cloud Run | Vertex AI Endpoint | Cloud Run is cheaper, simpler, no vendor-managed overhead. Vertex adds value at higher scale. |
| FastAPI | Flask, TorchServe, Triton | FastAPI auto-generates OpenAPI docs, type-validates requests via Pydantic, first-class async. |
| Multi-stage Docker | Single-stage slim | Image size matters for cold-start. Multi-stage drops build tools. |
| Prometheus text exposition | OpenTelemetry | Prometheus is simpler; OTel works too but more config. |
| Manual deploy script + GH Actions | Fully automatic via Terraform | Terraform adds complexity disproportionate to project scale. |
| Model bundled in image | Model in GCS, loaded at startup | Simpler. GCS path avoids re-building image when model updates, but adds a step. Easy to switch later. |

## Retrospective (to fill after deploy)

- **What went well:**
- **What to improve:**
- **Skills developed:**

## Presentation notes (non-technical)

- Lead: "I wrapped the sentiment model in a public API. Anyone can hit this URL and get a classification back."
- Show the live URL in a browser at `/docs` (Swagger UI looks impressive).
- Business angle: "If a product team wants to embed sentiment scoring in their app, they call this endpoint. No model expertise required on their side."
- Reliability: "CI runs on every push. If a change breaks a test, deploy does not happen. Rollback is one click."
- Cost: "Under $5 a month at demo traffic because the service scales to zero when idle."

## References

- `docs/theory/13_mlops_architecture_patterns.md`
- `docs/theory/14_cloud_deployment_primer.md`
- FastAPI docs: fastapi.tiangolo.com
- GitHub Actions docs: docs.github.com/en/actions
- Cloud Run docs: cloud.google.com/run/docs
