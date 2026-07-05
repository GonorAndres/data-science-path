# Cost and Latency

## Latency budget

Target for this service: P95 under 200 ms on Cloud Run with 2 vCPU, 2 GB RAM, CPU inference.

Budget breakdown per request:

| Step | Budget |
|------|--------|
| Network (client to Cloud Run) | 20 ms |
| Load balancer routing | 5 ms |
| Container request handling + middleware | 5 ms |
| Tokenization | 10 ms |
| Model forward pass (DistilBERT, 128 tokens, CPU) | 100 ms |
| Response serialization | 5 ms |
| Network (Cloud Run to client) | 20 ms |
| **Total P50** | ~165 ms |
| **Total P95 (with tail)** | ~250 ms |

Measured via `scripts/load_test.sh` (hey) and Cloud Run metrics.

## Where the time goes

Model forward pass dominates. Options to reduce:

1. **Dynamic quantization (int8).** PyTorch's `torch.quantization.quantize_dynamic`. 2-3x speedup on CPU, ~1 percent F1 drop.
2. **ONNX Runtime.** Export to ONNX, serve via `onnxruntime`. Typically 2x faster than PyTorch for transformer inference.
3. **Distillation.** Train a smaller student model (e.g., TinyBERT). Work-intensive.
4. **Batch inference.** If requests arrive in bursts, batch them. Latency per batch item drops dramatically.

For this project, ONNX export is documented but not implemented; adds complexity for a demo-scale service.

## Cold start

Cloud Run with min-instances = 0 cold-starts on the first request after idle. For this service:

- Container pull: ~2 seconds (500 MB image).
- Python interpreter + FastAPI import: ~500 ms.
- Model load (500 MB weights from local disk): ~2 seconds.
- Total cold start: ~4 to 5 seconds.

Subsequent requests are ~150 ms.

Mitigations:

- `min-instances = 1`: no cold start, but ~30 USD/month.
- Smaller image: reduce dependencies, use distroless or Alpine base.
- Smaller model: quantize and save int8 weights, cutting both image size and load time.
- Async warmup: precompute common inputs at startup.

## Throughput

One Cloud Run instance at 2 vCPU serves approximately:

- 5 to 10 RPS with 150 ms P50 latency (CPU-bound).
- Horizontal scale up to `max-instances` (default 10, configurable).
- At max 10 instances, ~50 to 100 RPS total.

For higher RPS, increase CPU allocation (up to 8 per instance on Cloud Run) or scale instances higher.

## Cost model

Cloud Run pricing is per-request plus per-CPU-second plus per-GB-second.

### Assumptions

- 10,000 requests per month (small demo).
- Average request: 150 ms wall clock.
- 2 vCPU, 2 GB RAM allocated.
- min-instances = 0 (scale to zero).

### Variable cost

- CPU time: 10,000 * 0.15 s * 2 vCPU = 3,000 vCPU-seconds = 0.83 vCPU-hours.
- Memory: 10,000 * 0.15 s * 2 GB = 3,000 GB-seconds = 0.83 GB-hours.

At published rates:

- CPU: $0.000024 per vCPU-second = ~$0.07.
- Memory: $0.0000025 per GB-second = ~$0.01.
- Requests: first 2M/month free, so $0.

Total compute: under $0.10 at 10,000 requests.

### Fixed cost

- None when scaled to zero.
- If min-instances = 1: 720 hours x (2 vCPU + 2 GB) at idle rate ~ $20-30 USD/month.

### Storage

Container image in Container Registry: 500 MB. ~$0.025 per GB-month = $0.01.

### Egress

Response body is small (~500 bytes). Egress at ~10,000 * 0.5 KB = 5 MB/month. Negligible.

### Total

Under $1 USD/month at 10,000 requests with scale-to-zero. Spikes to ~$10 if 1M requests/month.

## Scaling limits

At 1M requests/month (~25 RPS average, spikes higher):

- Still under 2M free requests per month.
- CPU cost: ~$7.
- Total: ~$10/month.

At 100M requests/month:

- 98M billable requests: ~$39.
- CPU: ~$700.
- Consider alternatives: reserved Compute Engine, dedicated inference service.

## Cost controls

- **Budget alerts.** Set at $10, $50, $100 in GCP Billing. Email on threshold crossing.
- **Max instances cap.** Default 10; raises possible DDoS-driven cost bombs.
- **Request size limit.** Already enforced at 10k chars per text.
- **Rate limit.** Could add Cloud Endpoints or a reverse proxy for per-IP limits (not implemented; documented).

## Latency vs cost tradeoffs

| Scenario | Cost | P95 latency |
|----------|------|-------------|
| min-instances=0, CPU=2 | $1/month | 200 ms warm, 5 s cold |
| min-instances=1, CPU=2 | $30/month | 200 ms always |
| min-instances=1, CPU=4 | $50/month | 120 ms always |
| min-instances=1, CPU=8, ONNX int8 | $80/month | 50 ms always |
| GPU (Cloud Run with GPU, 2026 beta) | $400+/month | 20 ms always |

Portfolio default: min-instances=0, CPU=2. Cheap. P95 of 5 seconds cold is acceptable for reviewer demo.

## Benchmarking workflow

1. Start the service locally or against Cloud Run URL.
2. Run `scripts/load_test.sh` with `URL` set.
3. Read the hey output: RPS, mean, median, 90th, 95th, 99th percentiles.
4. Compare to the table in `README.md`.
5. If P95 exceeds target, investigate via tracing.

## Cross-links

- `01_serving_patterns.md`: architecture context.
- `03_monitoring_and_drift.md`: continuous observability.
- `docs/theory/14_cloud_deployment_primer.md`: broader cloud cost framing.
