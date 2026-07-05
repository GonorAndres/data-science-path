# Problem Framing: Wrapping a Model for Production

## The problem

A trained model is not a product. It needs a surface that other systems can call without knowing anything about machine learning internals. That surface is an HTTP API.

Without it:

- The model lives in a notebook and is only usable by its author.
- Any downstream system must install HuggingFace, PyTorch, and the model weights locally.
- Any retraining requires downstream teams to copy new weights manually.
- There is no monitoring, no versioning, no rollback.

This project provides the surface: a FastAPI service that takes review text and returns sentiment, plus the infrastructure around it.

## Stakeholders

- **Upstream data scientist** (the person who trained the model). Wants to ship new versions without rebuilding the service from scratch.
- **Downstream developers** (app engineers, data engineers). Want a stable HTTP contract.
- **SRE or ML platform engineer.** Wants observability and rollback.
- **Cost owner.** Wants predictable, low infrastructure cost.
- **Security.** Wants no secrets in source and no unauthenticated admin endpoints.

## Non-goals

- Not a training pipeline (that lives in `nlp-reviews-multilingual`).
- Not a fully-fledged model registry (MLflow tracks training; this service pulls a specific artifact).
- Not a multi-model hosting platform (one model per deployment; multi-tenant comes later with Triton).

## Why a separate project?

Separation of concerns.

- `nlp-reviews-multilingual` owns the model: data, training, evaluation, notebooks.
- `mlops-model-service` owns the delivery: API, Docker, CI/CD, monitoring, deployment.

A real production team splits these. The model team iterates weekly; the platform team iterates on infrastructure. Coupling them in one repo slows both.

For the portfolio, showing this split is a real ML-engineer signal. Recruiters who look at this structure read "this person has worked on a team that knows how to draw the line."

## Success metrics

| Metric | Target |
|--------|--------|
| Live URL (Cloud Run) reachable | Yes |
| CI pipeline green | Yes, badge on root README |
| P95 latency at 10 RPS | under 250 ms |
| Cost per month at demo traffic | under $5 USD |
| Predictions logged with request IDs | 100 percent |
| Drift monitoring stub functional | Yes |
| Rollback time from bad deploy | under 5 min (via `gcloud run services update-traffic`) |

## Risks

- **Cold start.** Cloud Run scales to zero; first request after idle can take 5+ seconds. Mitigate: min-instances = 1 if required; otherwise accept and document.
- **Model size.** The DistilBERT model is ~500 MB. Container image becomes large; cold start slower. Mitigate: quantize model, or load from GCS at startup (trades image size for startup time).
- **Cost overrun.** An abusive client can hammer the endpoint; Cloud Run scales and bills accordingly. Mitigate: rate limiting at the gateway, budget alerts, max-instances cap.
- **Model drift.** The model was trained on 2020 reviews; 2026 language evolves. Mitigate: periodic retraining, drift monitoring, canary rollouts.
- **Dependency vulnerabilities.** Python deps have CVEs over time. Mitigate: Dependabot, pinned versions, quarterly review.

## What good looks like

A reviewer lands on the root README, clicks the Cloud Run URL, sees OpenAPI docs at `/docs`, tries an example, gets a sensible classification with latency under 200 ms, sees the CI badge green, reads the README and understands the architecture within 5 minutes, opens the `theory/` folder for depth.

That is the bar for "production-ready portfolio project" in 2026.

## Cross-links

- `docs/theory/13_mlops_architecture_patterns.md`: the broader framing.
- `docs/theory/14_cloud_deployment_primer.md`: cloud-specific details.
- `nlp-reviews-multilingual/README.md`: the model this service wraps.
