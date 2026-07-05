# Serving Patterns for ML Models

## The choice of pattern

Four dominant patterns for serving an ML model:

1. **Batch.** Run the model on a schedule; write predictions to a table. Latency measured in hours.
2. **Online synchronous.** HTTP API; one request, one response. Latency in milliseconds.
3. **Streaming.** Consume events from a queue (Kafka, PubSub), publish predictions to another. Latency in seconds.
4. **Edge / on-device.** Run on mobile or IoT hardware. Latency bound by device.

This project uses pattern 2 (online synchronous) because:

- The review classifier's natural use is integrated into other apps (dashboards, review widgets) that want a response now.
- Latency budget is permissive (200 ms is easy).
- Volume is moderate (thousands of requests per day at demo scale).

## FastAPI as the framework

FastAPI was chosen because:

- **First-class async.** Uvicorn + FastAPI handles hundreds of concurrent requests on one worker.
- **Pydantic validation.** Request bodies are typed; invalid input is rejected automatically with helpful error messages.
- **OpenAPI for free.** Swagger UI at `/docs`, ReDoc at `/redoc`. Enormous value for reviewers and downstream devs.
- **Standard library-like.** Minimal boilerplate; easy to read.

Alternatives considered:

- **Flask.** More mature, more plugins, but synchronous and no automatic schema.
- **TorchServe.** Purpose-built for PyTorch; adds complexity for a single-model service.
- **Triton.** Nvidia, multi-backend; overkill here.
- **BentoML.** Nice abstraction but adds another layer; FastAPI is enough.

## Request pattern

```
Client -> Load balancer (Cloud Run router) -> Container (FastAPI + Uvicorn worker)
           [TLS, gRPC, HTTP/2]                [middleware, endpoint, model]
```

Per request:

1. Middleware assigns a request ID (from header or generated) and starts a timer.
2. Pydantic validates the request body against the schema.
3. Endpoint handler extracts the text, calls `get_predictor().predict(text)`.
4. Response is serialized and returned.
5. Middleware observes latency and increments metrics counters.
6. Structured log line emitted.

## Model loading: lazy vs eager

Two options:

- **Eager** (at container startup). First request is fast; startup is slow.
- **Lazy** (on first request). Startup is fast; first request is slow.

This service uses eager via the FastAPI `lifespan` context. Cloud Run startup probes wait for the health endpoint to return ok, and since the model is loaded at startup, `/health` only reports ok once the model is ready.

Tradeoff: if the container has min-instances = 0 and must cold-start for the first request of the day, the user waits for model load (~2-3 seconds for DistilBERT). Acceptable for demo; not acceptable for a consumer product, where you set min-instances = 1 or use always-warm patterns.

## Async vs sync endpoint handlers

The predict endpoints are declared as `async def`. The model inference itself is synchronous (PyTorch call). For a lightweight model and CPU inference, wrapping in a thread pool is overkill; the GIL is not really the bottleneck at the RPS scale this project targets.

For higher throughput (thousands of RPS), options include:

- Run the model in a worker pool (`concurrent.futures.ThreadPoolExecutor`).
- Batch requests dynamically (collect requests for 10 ms, run as one batch).
- Use ONNX Runtime or torch.compile for faster inference.
- Scale horizontally (more Cloud Run instances).

## Batch endpoint

`/predict/batch` accepts up to 100 texts and processes them sequentially. For a real load pattern, batch inference is 3 to 5x faster per text than sequential single-text calls because of fixed per-call overhead.

The implementation here is sequential for simplicity. A production upgrade: pass the list through the tokenizer in one call (padding), then through the model in one forward pass, then split the outputs. Much faster.

## Input validation

Pydantic does the first pass:

- `text` must be a non-empty string.
- `text` max 10k characters (rejects pathological inputs).
- `texts` in batch: 1 to 100 items.

Domain validation (are the texts sensible, not just huge random bytes) is not enforced. Could be added:

- Detect and reject binary / control-character inputs.
- Detect and reject URLs-only or email-only inputs (probably not reviews).
- Language detection; warn if outside ES/EN.

For this demo, Pydantic's schema validation is enough.

## Response contract

Fields:

- `label`: one of `{"negative", "neutral", "positive"}`.
- `confidence`: float [0, 1], the probability of the chosen class.
- `probs`: dict with all three classes' probabilities.
- `model_version`: semver string.
- `request_id`: string for tracing.

Stability: semver on `model_version` and a documented deprecation window before breaking changes.

## Error handling

- 422: Pydantic validation failed. Automatic.
- 503: model not loaded (e.g., model path missing at startup). Handled in `/predict`.
- 500: unexpected exception. Middleware logs and propagates.
- 4xx: client errors return as raised by the framework.

Downstream callers should implement retries with exponential backoff and circuit breakers.

## Testing strategy

Two levels:

1. **Schema tests** (`tests/test_schema.py`). Pydantic validation.
2. **API tests** (`tests/test_api.py`). FastAPI `TestClient`; mock the predictor so tests do not need weights.

Skipped for the integration-level test (real model end-to-end). That runs in a separate CI job that pulls the model artifact; see `.github/workflows/`.

## Cross-links

- `02_ci_cd_for_ml.md`: how these tests run in CI.
- `03_monitoring_and_drift.md`: observability story.
- `04_cost_and_latency.md`: perf numbers.
- `docs/theory/13_mlops_architecture_patterns.md`: broader MLOps framing.
