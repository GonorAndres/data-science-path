# MLOps Platform Engineering

## Purpose

Build the ML platform itself, not just use it. This specialization focuses on designing and implementing the infrastructure that enables an entire organization to train, validate, deploy, monitor, and retrain models at scale. You become the engineer who builds the systems that other data scientists and ML engineers depend on -- feature stores, model registries, automated retraining pipelines, A/B testing infrastructure, and self-service deployment tooling.

---

## What It Unlocks

- **Career paths**: ML Platform Engineer, MLOps Engineer, ML Infrastructure Engineer, Staff ML Engineer (Platform)
- **Skills**: Infrastructure-as-code, Kubernetes orchestration, feature store design, model registry and governance, CI/CD for ML, data quality monitoring, automated model validation, canary deployments
- **Domain leverage**: Insurance ML has strict regulatory requirements for model governance (model risk management, fair lending/pricing compliance, audit trails). Your actuarial background gives you a deep understanding of these requirements that pure DevOps engineers lack.

---

## Key Technologies

| Technology | What It Does |
|---|---|
| **Kubernetes (EKS/GKE)** | Container orchestration for deploying and scaling ML workloads. The industry standard for production ML infrastructure. |
| **Feast** | Open-source feature store providing consistent feature serving for both offline training and online inference. |
| **Seldon Core / KServe** | Kubernetes-native model serving frameworks. Handle model deployment, scaling, A/B testing, and canary rollouts. |
| **Terraform** | Infrastructure-as-code tool for provisioning cloud resources (compute, networking, storage) reproducibly. |
| **Argo Workflows** | Kubernetes-native workflow orchestration for ML pipelines (training, validation, deployment DAGs). |
| **MLflow (at scale)** | Experiment tracking, model registry, and model versioning. At scale means multi-team, governed, with access controls. |
| **Great Expectations** | Data quality validation framework. Catches data issues before they corrupt model training or predictions. |
| **Prometheus + Grafana** | Monitoring and alerting stack for ML system health, model performance metrics, and data drift detection. |

---

## Project Ideas

### 1. Self-Service ML Platform

**Scope**: Build a Kubernetes-based platform that allows data scientists to train models, validate them against governance checks, deploy to production, and monitor performance -- all through a standardized interface without needing infrastructure knowledge.

**Dataset**: This is an infrastructure project. Use any 2-3 existing ML models from your portfolio (e.g., a claims severity model, a churn classifier, a fraud scorer) as the models that run on the platform.

**Deliverables**:
- Kubernetes cluster setup with Terraform (use a local Kind/Minikube cluster for development, document how to deploy to EKS/GKE)
- Training pipeline: submit a training job via CLI or API, runs on K8s, logs experiments to MLflow
- Model registry: MLflow-based with model versioning, metadata (dataset version, training config, performance metrics), approval workflow
- Deployment pipeline: promote model from registry to serving endpoint (Seldon Core or KServe), with canary rollout strategy
- Monitoring: Prometheus metrics for latency, throughput, error rate; custom metrics for model prediction distribution drift
- Data quality gate: Great Expectations suite runs before training, blocks pipeline if data checks fail
- User documentation: step-by-step guide for a data scientist to go from notebook to production
- Architecture diagram showing all components and data flows

**Complexity**: Very high. This is a staff-level engineering project. Build incrementally -- start with training + registry, then add serving, then monitoring.

### 2. Insurance Feature Store

**Scope**: Build a Feast-based feature store serving insurance-specific features with both offline (for training) and online (for real-time inference) serving. Solve the training-serving skew problem that plagues most insurance ML systems.

**Dataset**: Create a realistic insurance feature set using public data:
- Policyholder features: demographics, tenure, coverage history, payment behavior
- Claims features: frequency, severity, development patterns, provider network usage
- External features: weather, economic indicators, geographic risk factors
- Use the French Motor TPL dataset and augment with synthetic temporal features

**Deliverables**:
- Feature definitions in Feast with proper entity keys (policy_id, claimant_id, provider_id), timestamps, and feature descriptions
- Offline store: materialized features in Parquet/Delta for training dataset generation with point-in-time correctness
- Online store: Redis-backed low-latency feature serving for real-time inference
- Feature pipelines: Airflow/Argo DAGs computing daily/hourly feature updates from raw data sources
- Data quality monitoring: statistical tests on feature distributions, alerting on drift
- Feature catalog: searchable documentation of all available features with lineage and ownership
- Demonstration: train a model using offline features, serve it using online features, verify consistency
- Write-up on how this solves the training-serving skew problem and why point-in-time correctness matters for insurance

**Complexity**: High. Feature store design is nuanced. Point-in-time correctness is critical and tricky for insurance (policies have effective dates, claims develop over time).

### 3. Automated Retraining Pipeline

**Scope**: Build a system that monitors deployed model performance, detects drift or degradation, automatically triggers retraining on fresh data, validates the new model against the incumbent, and performs a canary deployment -- all without human intervention (except approval gates for regulated models).

**Dataset**: Use a model from your portfolio (claims severity or churn prediction) and simulate data drift by progressively shifting the data distribution over time.

**Deliverables**:
- Drift detection module: statistical tests (PSI, KS test, KL divergence) on input features and model predictions, running on a schedule
- Automated retraining trigger: when drift exceeds threshold, kick off training pipeline with latest data
- Model validation suite: performance benchmarks (new model must beat incumbent on holdout set), fairness checks (no protected class performance degradation), calibration tests
- Champion/challenger framework: new model serves 5% of traffic initially, full rollout after performance confirmation
- Rollback mechanism: automatic revert if canary model underperforms within first 24 hours
- Audit log: complete record of every retraining event, validation result, deployment decision, and rollback -- essential for insurance regulatory compliance
- Alert routing: drift detection alerts to model owners, validation failures to ML platform team
- End-to-end simulation: demonstrate the full cycle with simulated drift over 30 days

**Complexity**: High. The automated validation and governance checks are what differentiate this from a simple retrain-on-cron setup.

---

## Career Impact & Salary Context

- **Demand**: ML Platform Engineering is consistently the highest-paid ML specialization. Every company running ML at scale needs platform engineers, and the supply is limited.
- **Where to target**: Tech companies (Google, Meta, Amazon, Stripe, Uber), large financial institutions building internal ML platforms (JP Morgan, Goldman Sachs, Capital One), InsurTech companies at scale (Root, Lemonade), ML platform vendors (Databricks, Weights & Biases, Tecton).
- **Salary range (US)**: $180K-$280K total compensation for ML Platform Engineers. Senior/Staff roles at top-tier companies reach $300K+. This is the ceiling of ML engineering compensation.
- **Mexico/LATAM remote**: $80K-$150K USD for remote roles. Platform engineering skills are highly valued globally because they scale impact across entire ML teams.

---

## How Your Actuarial Background Provides an Edge

- Insurance model governance is strict. Regulators require model documentation, validation reports, and audit trails (OCC SR 11-7, NAIC model governance). You understand these requirements from actuarial training and can build platforms that satisfy them by design.
- Actuarial model validation practices (back-testing, stress testing, sensitivity analysis) translate directly into automated model validation suites.
- Your understanding of insurance data complexities (policy effective dates, claim development, censored observations, regulatory reporting periods) helps you design feature stores with correct temporal semantics.
- Pricing and reserving cycles in insurance create natural retraining schedules. Your domain knowledge helps you design retraining pipelines that align with business rhythms rather than arbitrary cron schedules.

---

## Prerequisites

- **Phase 2** (Data Engineering): Must be complete -- data pipelines are foundational to platform engineering.
- **Phase 3** (ML System Design): Must be complete -- this is an extension of system design to platform-level thinking.
- **Phase 4** (Deployment): Must be complete -- Docker, CI/CD, and basic Kubernetes are prerequisites.
- Comfort with cloud services (AWS or GCP -- at least one).
- Basic Kubernetes knowledge (pods, deployments, services, persistent volumes).
- Familiarity with infrastructure-as-code concepts.

---

## Estimated Timeline

- **Weeks 1-2**: Kubernetes deep-dive (beyond basics), Terraform fundamentals, Argo Workflows
- **Weeks 3-4**: Feature store concepts, Feast setup, offline/online serving patterns
- **Weeks 5-7**: Project 2 (Insurance Feature Store) -- build this first as it feeds into the platform
- **Weeks 8-10**: Project 1 (Self-Service ML Platform) -- training, registry, deployment
- **Weeks 11-13**: Project 3 (Automated Retraining Pipeline) + monitoring integration
- **Week 14**: Integration testing, documentation, architecture diagrams
- **Total: 13-14 weeks** at 15-20 hours/week

---

## Resources

### Courses
- [Chip Huyen - Designing Machine Learning Systems](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) -- essential reading for ML platform design
- [Kubernetes for ML (Made With ML)](https://madewithml.com/) -- practical K8s for ML workloads
- [Terraform Associate Certification](https://developer.hashicorp.com/terraform/tutorials) -- infrastructure-as-code fundamentals
- [Full Stack Deep Learning](https://fullstackdeeplearning.com/) -- end-to-end ML infrastructure

### Papers & Blog Posts
- "Hidden Technical Debt in Machine Learning Systems" (Sculley et al., Google, 2015) -- the foundational paper on ML infrastructure
- "Uber Michelangelo" -- Uber Engineering blog on their ML platform
- "Feast: Feature Store for Machine Learning" -- Tecton/Feast design paper
- "Continuous Delivery for Machine Learning" (CD4ML by ThoughtWorks) -- CI/CD patterns for ML

### Repos & Tools
- [Feast](https://github.com/feast-dev/feast) -- open-source feature store
- [Seldon Core](https://github.com/SeldonIO/seldon-core) -- model serving on K8s
- [KServe](https://github.com/kserve/kserve) -- Kubernetes model inference platform
- [Argo Workflows](https://github.com/argoproj/argo-workflows) -- workflow orchestration
- [Great Expectations](https://github.com/great-expectations/great_expectations) -- data quality
- [Evidently AI](https://github.com/evidentlyai/evidently) -- ML model monitoring and drift detection
