# Graph Neural Networks for Fraud Detection

## Purpose

Learn to detect fraud networks and collusion rings that tabular models miss entirely. Traditional ML treats each claim or policyholder as an independent row in a table. Graph-based methods model the relationships between entities -- claimants, providers, repair shops, attorneys, witnesses -- and find suspicious patterns in the structure of those connections. Insurance fraud costs over $80 billion per year in the US alone. GNNs are the most promising approach for catching organized fraud rings.

---

## What It Unlocks

- **Career paths**: Fraud Analytics ML Engineer, Graph ML Engineer, Insurance SIU (Special Investigation Unit) Data Scientist, Anti-Fraud Platform Engineer
- **Skills**: Graph construction from relational data, graph neural network architectures, node/edge classification, community detection, link prediction, graph databases
- **Domain leverage**: Insurance fraud is inherently a network problem. Organized rings share providers, addresses, attorneys, and witnesses. Your actuarial understanding of claims patterns lets you construct more meaningful graphs than a generic engineer would.

---

## Key Technologies

| Technology | What It Does |
|---|---|
| **PyTorch Geometric (PyG)** | Leading GNN library for PyTorch. Implements GCN, GraphSAGE, GAT, and dozens of other architectures. Production-ready. |
| **DGL (Deep Graph Library)** | Alternative GNN framework supporting both PyTorch and TensorFlow backends. Strong on heterogeneous graphs. |
| **NetworkX** | Python library for graph creation, analysis, and visualization. Good for prototyping and classical graph algorithms. |
| **Neo4j** | Graph database for storing and querying large-scale relationship data. Cypher query language. |
| **Node Classification** | GNN task: predict a label (fraudulent/legitimate) for each node based on its features and neighborhood structure. |
| **Link Prediction** | GNN task: predict missing or future edges. Useful for identifying hidden relationships between entities. |
| **Community Detection** | Algorithmic approach to finding clusters in graphs. Louvain, label propagation, spectral methods. |
| **Graph Attention Networks (GAT)** | GNN variant using attention mechanisms to weight neighbor contributions. Often outperforms GCN on heterogeneous data. |

---

## Project Ideas

### 1. Claims Fraud Ring Detection

**Scope**: Build a heterogeneous graph of insurance claims connecting claimants, medical providers, repair shops, attorneys, and witnesses. Train a GNN to flag suspicious clusters and individual nodes that are likely part of organized fraud rings.

**Dataset**: Start with the synthetic claims data from the Insurance Fraud Benchmark Dataset (if available) or construct a synthetic dataset with the following structure:
- 50,000+ claims with associated entities
- Embedded fraud rings (5-20 entities connected through shared providers, addresses, phone numbers)
- Entity features: claim frequency, average severity, geographic patterns, temporal patterns
- Use public datasets like the Medicare Provider Utilization data to create realistic provider-level features

**Deliverables**:
- Data pipeline transforming relational claims data into a heterogeneous graph (multiple node types, multiple edge types)
- Exploratory graph analysis: degree distribution, connected components, clustering coefficients, centrality measures
- Baseline: classical graph algorithms (PageRank, community detection) for fraud scoring
- GNN model: GraphSAGE or GAT for node classification (fraud/legitimate)
- Comparison against tabular baseline (XGBoost on flattened features) -- demonstrate what the graph captures that tabular misses
- Visualization of detected fraud rings using NetworkX or Pyvis (interactive HTML graphs)
- False positive analysis: what patterns cause false alarms, how to tune precision/recall for SIU workload capacity
- Write-up framing results for a non-technical SIU audience (this is a portfolio differentiator)

**Complexity**: High. Graph construction from relational data is the hardest part. Getting the node/edge feature engineering right requires domain knowledge.

### 2. Social Network Risk Scoring for Underwriting

**Scope**: Build a policyholder relationship graph and use it to improve underwriting risk assessment. The insight: people connected to high-risk policyholders are themselves higher risk (adverse selection through social networks, shared risk behaviors, potential for collusion).

**Dataset**: Synthetic policyholder data with relationship edges derived from:
- Shared addresses (current and historical)
- Shared vehicles or policies
- Referral chains (who referred whom)
- Co-involvement in claims (witness relationships)
- Use the French Motor TPL dataset from OpenML as a base, then layer synthetic relationship data on top

**Deliverables**:
- Graph construction pipeline with multiple relationship types (household, referral, co-claimant, shared provider)
- Node features from policy and claims history (loss ratio, claim frequency, tenure, coverage type)
- GNN-based risk score that incorporates neighborhood risk information
- Lift analysis: how much does graph-based risk scoring improve over traditional GLM pricing factors
- Fairness analysis: check whether graph-based scoring introduces proxy discrimination through network homophily
- Integration design: how this score feeds into an underwriting decision system alongside traditional factors
- Discussion of regulatory implications (using social connections in underwriting is legally sensitive)

**Complexity**: Medium-high. The fairness and regulatory analysis is what makes this project stand out in a portfolio.

---

## Career Impact & Salary Context

- **Demand**: Niche but extremely high-value. Insurance fraud detection is a permanent, well-funded problem. Companies like Shift Technology (valued at $1B+) and FRISS are entirely built on fraud detection.
- **Where to target**: Shift Technology, FRISS, Verisk, SAS (fraud division), large carrier SIU analytics teams (Travelers, AIG, Zurich), reinsurance analytics (Swiss Re, Munich Re).
- **Salary range (US)**: $150K-$210K for GNN/Graph ML specialists. Premium for insurance domain expertise. Fraud analytics roles often include performance bonuses tied to fraud savings.
- **Mexico/LATAM remote**: $55K-$100K USD. Fraud detection is global -- many US/European insurers have analytics teams in Mexico and LATAM.
- **Differentiation**: Very few ML engineers combine GNN skills with insurance domain knowledge. This makes you memorable in interviews and highly recruitable by fraud-focused InsurTechs.

---

## How Your Actuarial Background Provides an Edge

- You understand claims data structures intimately -- what fields exist, how claims develop, what patterns indicate reserving manipulation vs. legitimate development. This is critical for feature engineering on claim nodes.
- Actuarial training covers loss distribution analysis. You can distinguish between fraud patterns and legitimate heavy-tailed loss behavior that might fool a naive model.
- Your understanding of insurance operations (underwriting, claims handling, SIU workflows) means you can design graph features that reflect actual business processes, not just abstract network metrics.
- Pricing and reserving knowledge helps you quantify the financial impact of detected fraud -- essential for getting executive buy-in and prioritizing SIU resources.
- You can navigate the regulatory landscape (unfair claims practices acts, privacy regulations) that constrains how graph-based scoring can be used in practice.

---

## Prerequisites

- **Phase 1** (Python ML Foundations): Must be complete -- you need strong Python, PyTorch basics, and classification model evaluation.
- **Phase 3** (ML System Design): Recommended -- graph ML systems have unique design considerations (graph size, sampling strategies, inference patterns).
- Basic understanding of neural networks (backpropagation, gradient descent, embeddings).
- Comfort with PyTorch fundamentals (tensors, autograd, training loops).

---

## Estimated Timeline

- **Weeks 1-2**: Graph theory fundamentals, NetworkX, classical graph algorithms (PageRank, community detection, centrality)
- **Weeks 3-4**: GNN theory (message passing, GCN, GraphSAGE, GAT), PyTorch Geometric basics
- **Weeks 5-7**: Project 1 (Claims Fraud Ring Detection)
- **Weeks 8-10**: Project 2 (Social Network Risk Scoring) + fairness analysis
- **Week 11**: Neo4j integration, graph database patterns for production deployment
- **Total: 10-11 weeks** at 15-20 hours/week

---

## Resources

### Courses
- [Stanford CS224W: Machine Learning with Graphs](http://web.stanford.edu/class/cs224w/) -- the definitive GNN course (free lecture videos and slides)
- [PyTorch Geometric Tutorials](https://pytorch-geometric.readthedocs.io/en/latest/get_started/colabs.html) -- official hands-on tutorials
- [Graph Neural Networks: Foundations, Frontiers, and Applications](https://graph-neural-networks.github.io/) -- comprehensive textbook (free online)

### Papers
- "Inductive Representation Learning on Large Graphs" (Hamilton et al., 2017) -- GraphSAGE paper
- "Graph Attention Networks" (Velickovic et al., 2018) -- GAT paper
- "Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks" (Weber et al., 2019) -- financial fraud detection with GNNs
- "Heterogeneous Graph Neural Networks for Fraud Detection" -- directly applicable architecture pattern

### Repos & Tools
- [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) -- primary GNN library
- [DGL](https://github.com/dmlc/dgl) -- Deep Graph Library
- [Pyvis](https://github.com/WestHealth/pyvis) -- interactive graph visualization in Jupyter
- [Neo4j Python Driver](https://github.com/neo4j/neo4j-python-driver) -- graph database integration
- [OGB (Open Graph Benchmark)](https://ogb.stanford.edu/) -- standardized graph ML benchmarks for evaluating your models
