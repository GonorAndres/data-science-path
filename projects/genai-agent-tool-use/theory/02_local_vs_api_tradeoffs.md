# Local vs API Tradeoffs

## The axes

| Dimension | Local (Ollama, Llama 3.1 8B) | API (Claude, GPT-4o) |
|-----------|-------------------------------|----------------------|
| Cost per query | $0 (owned hardware) | ~$0.001 to $0.05 |
| Latency | 2 to 10 seconds per turn | 1 to 5 seconds per turn |
| Capability | Good for simple tasks | Much better for complex reasoning |
| Privacy | Data never leaves your machine | Data sent to vendor |
| Determinism | temperature=0 works well | Some inherent stochasticity |
| Setup | 30 minutes | 5 minutes (API key) |
| Operational | Self-hosted; you manage uptime | Vendor-hosted |
| Context length | 8k to 128k depending on model | Up to 200k or 1M+ |
| Multilingual | Depends on model | Generally strong |

## When local wins

- Privacy-sensitive data (health, legal, finance) where the data cannot leave your infrastructure.
- High-volume workloads where API cost at scale exceeds hardware cost.
- Fine-grained control over the generation process.
- Demo and portfolio projects where showing "I can run this locally" is a signal.
- Offline or air-gapped environments.

## When API wins

- Complex tasks needing frontier capability.
- Small projects where setup cost outweighs compute savings.
- Rapid prototyping.
- Multi-language tasks in low-resource languages (Claude and GPT-4o outperform 8B local on Spanish subtle reasoning).
- When you need the best model available.

## Hybrid patterns

Production systems often route:

- Easy tasks -> local model.
- Hard tasks -> API.
- Cost ceiling per task -> local first, API fallback.

For this portfolio, a pure local approach is simpler and more differentiated.

## Model selection for local

- **Llama 3.1 8B Instruct.** Strong default. Multilingual. Good instruction following.
- **Qwen 2.5 7B Instruct.** Stronger in some benchmarks, especially non-English.
- **Mistral 7B Instruct.** Older but very efficient.
- **Phi-3.5 Mini.** Smaller (3.8B); lightweight.
- **Llama 3.1 70B.** Much stronger but requires server-class GPU.

Quantization (Q4, Q5, Q8) trades quality for memory. Q4_K_M is a good default; Q5_K_M if memory permits.

## Hardware

- **CPU-only.** Works for 7B Q4 models. 10 to 30 tokens/sec on 8-core modern CPU.
- **Mac M-series.** Apple silicon runs Ollama very well via Metal.
- **Consumer GPU (12-16 GB VRAM).** 8B Q4 runs at 100+ tokens/sec. Ideal setup.
- **A100 / H100.** Overkill for 8B; makes sense for 70B.

## Production deployment of local

- Single-host deployment: Docker container with the model bundled or pulled on startup.
- Multi-host: a dedicated inference service with a pool of GPU instances; the app calls this internal service.
- On-device: smaller models (3-4B) can run on mobile via llama.cpp with aggressive quantization.

## Cross-links

- `01_llm_tool_use.md`
- `03_agent_evaluation.md`
- `docs/theory/09_llm_foundations_2026.md`
