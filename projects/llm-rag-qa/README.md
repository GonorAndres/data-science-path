# llm-rag-qa

## Resumen (ES)

**Problema:** construir un sistema de preguntas y respuestas con RAG sobre un corpus publico en espanol, usando un LLM local via Ollama. Sin costos de API, con despliegue reproducible en cualquier maquina con 16GB RAM.

**Metodo:** chunking + embeddings multilingues (sentence-transformers) + Chroma como vector store + Llama 3.1 8B Instruct o Qwen 2.5 7B via Ollama para generacion. Evaluacion con faithfulness, context recall y answer relevance.

**Resultado objetivo:** pipeline completo reproducible, evaluacion documentada en 50 preguntas de prueba, demo Gradio local, reporte de costo y latencia.

## Summary (EN)

**Problem:** build a RAG QA system over a public Spanish corpus using a local LLM via Ollama. Zero API cost, reproducible deploy on any 16GB-RAM machine.

**Method:** chunking + multilingual embeddings (sentence-transformers) + Chroma as vector store + Llama 3.1 8B Instruct or Qwen 2.5 7B via Ollama for generation. Evaluation with faithfulness, context recall, answer relevance.

**Target result:** full reproducible pipeline, evaluation documented on 50 test questions, local Gradio demo, cost and latency report.

---

## Prerequisites

1. Install Ollama: `curl -fsSL https://ollama.com/install.sh | sh`
2. Pull a model: `ollama pull llama3.1:8b-instruct-q4_K_M` (or `qwen2.5:7b-instruct`)
3. Verify: `ollama list`

Minimum hardware: 16 GB RAM. Apple silicon or a GPU helps but is not required.

## Dataset

Wikipedia articles in Spanish, filtered by topic (initial version: constitutional law). Downloaded via the `wikipedia-api` or from public dumps. ~5 to 10 MB of clean text.

## Structure

```
llm-rag-qa/
├── README.md
├── theory/
│   ├── 00_problem_framing.md
│   ├── 01_rag_architecture.md
│   ├── 02_embeddings_and_vector_stores.md
│   ├── 03_retrieval_evaluation.md
│   └── 04_prompt_patterns.md
├── data_download.py
├── src/
│   ├── __init__.py
│   ├── chunk.py              # text chunking
│   ├── embed.py              # sentence-transformers encoder
│   ├── store.py              # Chroma wrapper
│   ├── generate.py           # Ollama client
│   ├── rag.py                # end-to-end pipeline
│   └── evaluate.py           # faithfulness, context recall
├── eval/
│   └── questions.jsonl       # 50 eval questions with gold answers
├── tests/
├── notebooks/
├── app.py                    # Gradio demo
├── requirements.txt
└── Makefile
```

## Quickstart

```bash
pip install -r requirements.txt
python data_download.py         # fetch & clean corpus
python -m src.chunk             # chunk the corpus
python -m src.embed             # embed chunks, populate Chroma
python -m src.rag "que es la separacion de poderes?"   # one-shot CLI query
python -m src.evaluate          # run eval set
python app.py                   # Gradio demo
```

## Expected results

| Metric | Target | Note |
|--------|--------|------|
| Context recall @5 | > 0.8 | Of the gold context, how much is retrieved |
| Faithfulness | > 0.85 | Share of claims grounded in context |
| Answer relevance | > 0.8 | Does the answer address the question |
| Median latency | < 5 s | On CPU-only; GPU < 1 s |

## Decisions and trade-offs

| Decision | Alternative | Why |
|----------|-------------|-----|
| Ollama local | Claude or GPT API | Zero API cost; showcases privacy-preserving deploy. |
| sentence-transformers multilingual | OpenAI embeddings | Open, free, runs locally. |
| Chroma | FAISS, pgvector | Simple Python API, SQLite-backed, fits the scope. |
| Llama 3.1 8B Instruct | 70B | 8B runs on consumer hardware; 70B needs server-class GPU. |
| Spanish Wikipedia constitutional subset | Mexican government PDFs | Smaller, cleaner, faster to iterate. |

## References

- `docs/theory/09_llm_foundations_2026.md`
- Lewis et al., "Retrieval-Augmented Generation", 2020.
- Ollama docs: ollama.com
- Chroma docs: docs.trychroma.com
