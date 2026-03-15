# Real-Time ML Systems

## Purpose

Move from batch prediction pipelines to streaming inference systems capable of sub-second predictions. Insurance has critical real-time use cases -- fraud detection at point of claim, telematics risk scoring, dynamic pricing at quote time -- where latency matters as much as accuracy. This specialization teaches you to build the infrastructure that makes real-time ML possible.

---

## What It Unlocks

- **Career paths**: ML Platform Engineer, ML Infrastructure Engineer, Streaming ML Engineer, Real-Time Systems Engineer
- **Skills**: Stream processing architecture, online feature computation, low-latency model serving, online learning, distributed systems fundamentals
- **Domain leverage**: Insurance is transitioning from batch-overnight processing to real-time decisioning. Engineers who can build these systems for insurance-specific workflows are rare and valuable.

---

## Key Technologies

| Technology | What It Does |
|---|---|
| **Apache Kafka** | Distributed event streaming platform. The backbone for real-time data pipelines. Handles millions of events/second with durability guarantees. |
| **Apache Flink** | Stream processing framework for stateful computations over event streams. Handles windowed aggregations, complex event processing. |
| **Spark Structured Streaming** | Spark's streaming API -- good if you already know Spark. Micro-batch or continuous processing. |
| **Redis** | In-memory data store for ultra-low-latency feature serving. Sub-millisecond reads for pre-computed features. |
| **Kubernetes** | Container orchestration for deploying and scaling model serving infrastructure. Essential for production ML systems. |
| **gRPC** | High-performance RPC framework. Faster than REST for model serving (binary serialization via Protocol Buffers). |
| **Online Learning** | Models that update incrementally with each new observation (River library). Critical when data distribution shifts rapidly. |
| **Feature Stores (Feast)** | Unified feature serving for both batch training and real-time inference. Ensures training-serving consistency. |

---

## Project Ideas

### 1. Real-Time Fraud Detection Pipeline

**Scope**: Build an end-to-end streaming system that ingests claim events, computes features in real-time, scores a fraud model, and triggers alerts -- all within 100ms of the event arriving.

**Dataset**: Use the IEEE-CIS Fraud Detection dataset from Kaggle as a starting point, then augment with synthetic claim events to create a realistic stream. Alternatively, generate synthetic insurance claim events with known fraud patterns embedded.

**Deliverables**:
- Kafka producer simulating incoming claim events at configurable throughput (100-10,000 events/second)
- Flink job computing streaming features: velocity checks (claims per policyholder in last 24h), geographic anomalies, provider network patterns
- Redis feature store with pre-computed policyholder history features (claim count, average severity, time since last claim)
- Model serving via gRPC endpoint (XGBoost or LightGBM model trained offline, served with sub-10ms latency)
- Alert routing: high-confidence fraud -> immediate SIU alert, medium-confidence -> queue for review, low -> auto-approve
- Monitoring dashboard (Grafana) showing throughput, latency percentiles (p50, p95, p99), model score distribution
- Load testing results demonstrating system behavior under 10x normal volume

**Complexity**: High. This is a distributed systems project as much as an ML project. Use Docker Compose locally, then deploy to a small K8s cluster.

### 2. Telematics Driving Score (Streaming Sensor Data)

**Scope**: Process streaming telematics data (accelerometer, GPS, gyroscope) from connected vehicles, compute driving behavior features in real-time, and produce a continuously-updated risk score per driver.

**Dataset**: Use the UAH-DriveSet (University of Alcala) or generate synthetic telematics data following realistic driving patterns (harsh braking events, speeding, cornering g-forces, time-of-day patterns).

**Deliverables**:
- Kafka topic ingesting raw sensor readings at 10Hz per vehicle
- Flink streaming job computing trip-level features: harsh braking count, max speed, cornering severity, phone distraction proxy
- Sliding window aggregations: 7-day rolling risk score, trend detection (improving vs. deteriorating driving)
- Risk score model updating with each completed trip (online learning with River or incremental batch)
- REST API serving current driver risk score with feature breakdown
- Visualization dashboard showing real-time trip tracking and score evolution
- Architecture document explaining how this integrates with a usage-based insurance (UBI) pricing model

**Complexity**: Medium-high. The streaming computation is the core challenge. Signal processing on sensor data adds domain complexity.

### 3. Dynamic Pricing Engine

**Scope**: Build a system that adjusts insurance quote prices in real-time based on demand signals, competitive intelligence, capacity constraints, and risk appetite -- serving updated prices within 200ms of a quote request.

**Dataset**: Synthetic quote flow data with demand patterns (seasonal, day-of-week, marketing campaign effects). Use the French Motor TPL dataset from OpenML as a base risk model, then layer dynamic adjustments on top.

**Deliverables**:
- Quote request ingestion via Kafka with enrichment from multiple sources (risk model, demand signal, competitor proxy, capacity tracker)
- Pricing rules engine combining base actuarial price with dynamic adjustments (demand elasticity, competitive positioning, portfolio steering)
- Redis-cached price modifiers updated every 5 minutes from batch analytics
- A/B testing infrastructure: randomly assign quotes to pricing strategies, track conversion and profitability
- Real-time monitoring: conversion rate, average premium, loss ratio forecast by segment
- Guardrails: rate adequacy checks, regulatory min/max bounds, anti-discrimination constraints
- Technical write-up on the tension between actuarial pricing (long-term adequacy) and dynamic pricing (short-term optimization)

**Complexity**: High. Requires understanding of both the technical streaming infrastructure and the actuarial pricing domain.

---

## Career Impact & Salary Context

- **Demand**: Real-time ML infrastructure is a core competency for ML Platform teams at any company running ML at scale. Insurance companies are actively building these capabilities.
- **Where to target**: ML Platform teams at tech companies (Stripe, Uber, Airbnb), InsurTech companies with real-time products (Root, Metromile/Lemonade for telematics), large carriers building internal ML platforms.
- **Salary range (US)**: $170K-$250K for ML Infrastructure / Platform Engineers. This is consistently one of the highest-paid ML specializations because it requires both ML knowledge and distributed systems expertise.
- **Mexico/LATAM remote**: $70K-$130K USD for remote roles. Streaming infrastructure skills are highly portable across industries.

---

## How Your Actuarial Background Provides an Edge

- You understand insurance data flows (policy admin -> claims -> billing) and can design streaming pipelines that respect the actual business process, not just generic event streams.
- Actuarial pricing knowledge is essential for the Dynamic Pricing Engine project. You can validate that real-time adjustments maintain rate adequacy -- something a pure infrastructure engineer cannot do.
- Your understanding of loss development patterns and reserving helps you design fraud detection features that a generic data scientist would miss (e.g., claim development velocity as a fraud signal).
- Regulatory awareness from actuarial training helps you build compliant real-time systems (rate filing constraints, unfair discrimination rules).

---

## Prerequisites

- **Phase 2** (Data Engineering): Must be complete -- you need strong SQL, pipeline design, and data infrastructure fundamentals.
- **Phase 3** (ML System Design): Must be complete -- real-time systems are advanced ML system design.
- **Phase 4** (Deployment): Must be complete -- Docker, Kubernetes, and API design are foundational.
- Basic understanding of distributed systems concepts (partitioning, replication, consistency).
- Comfort with Linux command line and networking basics.

---

## Estimated Timeline

- **Weeks 1-2**: Kafka fundamentals, producer/consumer patterns, Docker-based local setup
- **Weeks 3-4**: Stream processing with Flink (windowing, state management, exactly-once semantics)
- **Weeks 5-7**: Project 1 (Real-Time Fraud Detection Pipeline)
- **Weeks 8-9**: Online learning, feature stores for real-time serving
- **Weeks 10-11**: Project 2 (Telematics Driving Score)
- **Weeks 12-14**: Project 3 (Dynamic Pricing Engine) + integration testing + load testing
- **Total: 12-14 weeks** at 15-20 hours/week

---

## Resources

### Courses
- [Confluent Kafka Fundamentals](https://developer.confluent.io/learn-kafka/) -- official Kafka learning path, free
- [Apache Flink Training](https://nightlies.apache.org/flink/flink-docs-stable/docs/learn-flink/overview/) -- official Flink tutorials
- [Chip Huyen - Designing Machine Learning Systems](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) -- Chapter on real-time ML (book)
- [Made With ML - MLOps](https://madewithml.com/) -- practical MLOps including serving patterns

### Papers
- "Machine Learning in Real-Time Applications" (survey) -- overview of real-time ML architecture patterns
- "Online Learning for Real-Time Fraud Detection" -- applicable patterns for streaming fraud
- "Uber's Michelangelo" -- real-world ML platform with real-time serving (Uber Engineering blog)

### Repos & Tools
- [River](https://github.com/online-ml/river) -- online machine learning library for Python
- [Feast](https://github.com/feast-dev/feast) -- open-source feature store
- [Seldon Core](https://github.com/SeldonIO/seldon-core) -- ML model serving on Kubernetes
- [Redpanda](https://github.com/redpanda-data/redpanda) -- Kafka-compatible streaming (simpler for local dev)
- [BentoML](https://github.com/bentoml/BentoML) -- model serving framework with good latency characteristics
