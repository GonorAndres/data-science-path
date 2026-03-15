# Generative AI & LLM Applications for Insurance

## Purpose

Build production-grade skills in retrieval-augmented generation (RAG), prompt engineering, fine-tuning, and LLM-powered applications -- all applied to insurance domain problems. This is the single highest-demand specialization in ML Engineering for 2025-2026, and insurance is massively underserved by quality LLM tooling.

---

## What It Unlocks

- **Career paths**: InsurTech AI Engineer, Innovation Lab ML Engineer, LLM Applications Engineer, AI Product Engineer
- **Skills**: RAG architecture design, embedding strategies, prompt optimization, evaluation of generative outputs, fine-tuning workflows, agent design patterns
- **Domain leverage**: Insurance documents (policies, endorsements, regulations) are dense, domain-specific text -- exactly where RAG shines over raw LLM knowledge. Your actuarial background lets you judge output quality in ways a generic ML engineer cannot.

---

## Key Technologies

| Technology | What It Does |
|---|---|
| **LangChain / LlamaIndex** | Orchestration frameworks for building LLM-powered pipelines (retrieval, chaining, agents) |
| **OpenAI / Anthropic APIs** | Foundation model access for generation, embeddings, and structured outputs |
| **Pinecone / ChromaDB** | Vector databases for storing and querying document embeddings at scale |
| **RAG Pipelines** | Architecture pattern: retrieve relevant context from a knowledge base, then generate grounded answers |
| **LoRA Fine-Tuning** | Parameter-efficient fine-tuning that adapts a base model to your domain without retraining billions of parameters |
| **RAGAS** | Evaluation framework for RAG pipelines -- measures faithfulness, relevance, context precision |
| **Guardrails / NeMo Guardrails** | Safety and output validation layers for production LLM apps |

---

## Project Ideas

### 1. Policy Document Q&A System

**Scope**: Build a RAG pipeline that answers underwriting questions by retrieving and reasoning over policy wordings, endorsements, and coverage definitions.

**Dataset**: Use publicly available sample policy wordings from state insurance department filings (NAIC SERFF database has thousands of filed policy forms). Supplement with ISO commercial lines forms.

**Deliverables**:
- Document ingestion pipeline (PDF parsing, chunking strategies, metadata extraction)
- Embedding pipeline with comparison of chunking approaches (fixed-size vs. semantic vs. section-aware)
- Vector store with hybrid search (dense + sparse retrieval)
- Evaluation suite using RAGAS metrics on a curated test set of 50+ underwriting questions
- Streamlit UI for interactive Q&A with source attribution
- Write-up comparing retrieval strategies and their impact on answer quality

**Complexity**: Medium-high. The hard part is chunking insurance documents intelligently -- policy sections reference each other extensively.

### 2. Claims Summarization Engine

**Scope**: Fine-tune an LLM to produce structured summaries from free-text adjuster notes. Output should extract: incident type, liability assessment, injury severity, reserve recommendation rationale.

**Dataset**: Synthetic claims notes generated from templates based on real claim patterns (you cannot use real claims data, but you can create realistic synthetic data using publicly available NHTSA crash narratives and workers' comp case summaries).

**Deliverables**:
- Synthetic dataset generation pipeline (500+ claim narratives with gold-standard structured summaries)
- LoRA fine-tuning pipeline on Llama 3 or Mistral base model
- Comparison: fine-tuned model vs. few-shot prompting on GPT-4 vs. zero-shot
- Latency and cost analysis for each approach
- Evaluation metrics: ROUGE, BERTScore, plus domain-specific accuracy on extracted fields
- Deployment-ready API with FastAPI

**Complexity**: High. Fine-tuning requires GPU access (use Lambda Labs, RunPod, or Colab Pro).

### 3. Regulatory Compliance Checker

**Scope**: Build an LLM agent that checks policy language against state-specific regulatory requirements, flagging non-compliant clauses and suggesting corrections.

**Dataset**: State insurance regulations from NAIC model laws and state DOI websites. Policy templates from ISO or sample filings.

**Deliverables**:
- Regulation knowledge base with structured extraction of requirements by state and line of business
- Agent architecture: parse policy clause -> retrieve applicable regulations -> assess compliance -> generate finding
- Multi-step reasoning chain with tool use (calculator for rate checks, date logic for filing deadlines)
- Confidence scoring and human-in-the-loop escalation design
- Test suite with known compliant and non-compliant policy excerpts
- Architecture documentation showing how this integrates into a carrier's filing workflow

**Complexity**: High. Requires careful agent design to avoid hallucinated compliance judgments.

---

## Career Impact & Salary Context

- **Demand**: Generative AI / LLM engineering is the highest-demand ML skill in 2025-2026. Job postings with "RAG" or "LLM" have grown 400%+ since 2023.
- **Where to target**: InsurTech startups (Lemonade, Hippo, Root, Clearcover), innovation/labs teams at large carriers (State Farm, Travelers, Zurich), consulting firms building AI solutions for insurance clients.
- **Salary range (US)**: $140K-$220K for LLM/AI Engineers with domain expertise. InsurTech startups often include equity.
- **Mexico/LATAM remote**: $60K-$120K USD for remote roles with US-based InsurTechs. Growing rapidly.

---

## How Your Actuarial Background Provides an Edge

- You can evaluate whether an LLM's output about policy coverage, exclusions, or risk assessment is actually correct -- most ML engineers cannot.
- Insurance language is highly specialized. Your training means you understand terms of art (occurrence vs. claims-made, aggregate limits, subrogation) that would confuse a generic engineer.
- Actuarial reserving and pricing knowledge lets you build RAG systems that answer quantitative questions, not just text retrieval.
- Regulatory knowledge from actuarial exams (especially on rate filings) directly applies to the compliance checker project.

---

## Prerequisites

- **Phase 1** (Python ML Foundations): Must be complete -- you need solid Python and ML fundamentals.
- **Phase 3** (ML System Design): Should be complete -- RAG systems are production systems that need proper architecture.
- **Phase 4** (Deployment): Should be complete -- you will deploy these as APIs and web apps.
- Basic comfort with API calls and JSON/REST patterns.

---

## Estimated Timeline

- **Weeks 1-2**: LLM fundamentals, prompt engineering, API usage patterns
- **Weeks 3-5**: RAG deep-dive (chunking, embeddings, retrieval, evaluation)
- **Weeks 6-7**: Project 1 (Policy Q&A System)
- **Weeks 8-10**: Fine-tuning fundamentals and Project 2 (Claims Summarization)
- **Weeks 11-12**: Agent patterns and Project 3 (Compliance Checker)
- **Total: 10-12 weeks** at 15-20 hours/week

---

## Resources

### Courses
- [LLM Bootcamp by Full Stack Deep Learning](https://fullstackdeeplearning.com/) -- best practical LLM engineering course
- [DeepLearning.AI: LangChain for LLM Application Development](https://www.deeplearning.ai/short-courses/) -- short and focused
- [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course) -- foundational NLP and transformers

### Papers
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020) -- the RAG paper
- "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021) -- fine-tuning method
- "RAGAS: Automated Evaluation of Retrieval Augmented Generation" (Es et al., 2023) -- evaluation framework

### Repos & Tools
- [LangChain](https://github.com/langchain-ai/langchain) -- orchestration framework
- [LlamaIndex](https://github.com/run-llama/llama_index) -- data framework for LLM apps
- [RAGAS](https://github.com/explodinggradients/ragas) -- RAG evaluation
- [Instructor](https://github.com/jxnl/instructor) -- structured outputs from LLMs
- [Unstructured](https://github.com/Unstructured-IO/unstructured) -- document parsing (essential for insurance PDFs)
