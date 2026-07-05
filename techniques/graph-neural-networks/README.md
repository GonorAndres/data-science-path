---
tags: [gnn, deep-learning, graphs]
status: complete
created: 2026-04-23
updated: 2026-04-25
---

# Graph Neural Networks

## What it is

Neural networks that operate directly on graph-structured data: social networks, molecular structures, knowledge graphs, user-item interactions, transaction networks. The core primitive is **message passing**: each node aggregates information from its neighbors, transforms it, and updates its own representation. Stack $L$ message-passing layers and a node's final representation incorporates information from its $L$-hop neighborhood.

## Why it matters

Many real datasets are naturally graphs and treating them as tables loses the structural signal. GNNs solve three classes of task:

- **Node classification.** Predict a label for each node (fraud / not fraud, customer segment, account-takeover risk).
- **Link prediction.** Predict whether an edge exists or will form (recommendation, knowledge-graph completion, drug-drug interaction).
- **Graph classification.** Predict a label for an entire graph (molecular property, malware family on a control-flow graph).

For an actuarial reader the canonical use case is fraud rings in claims data: collusion is invisible to a per-claim model but obvious as dense subgraphs in the claimant-witness-vehicle graph.

## The message-passing framework

A general message-passing layer at step $\ell$ updates each node's hidden state $h_v^{(\ell)}$ from the previous step:

$$
m_v^{(\ell+1)} = \text{AGGREGATE}^{(\ell)}\!\left( \{\, h_u^{(\ell)} : u \in \mathcal{N}(v) \,\} \right)
$$

$$
h_v^{(\ell+1)} = \text{UPDATE}^{(\ell)}\!\left( h_v^{(\ell)},\ m_v^{(\ell+1)} \right)
$$

Three components:

- **MESSAGE** (sometimes folded into AGGREGATE). For each neighbor $u$, compute a message $\phi(h_v, h_u, e_{uv})$. Often just $h_u$.
- **AGGREGATE.** Reduce the set of messages into a single vector. Must be permutation-invariant: sum, mean, max, attention-weighted mean.
- **UPDATE.** Combine the aggregated message with the node's own state. Often a small MLP.

GCN, GraphSAGE, and GAT all instantiate this framework with different choices.

## GCN: Graph Convolutional Network

Kipf and Welling (2017) derived the GCN layer as a first-order approximation of a spectral graph convolution. The matrix-form layer is

$$
H^{(\ell+1)} = \sigma\!\left( \tilde{D}^{-1/2}\, \tilde{A}\, \tilde{D}^{-1/2}\, H^{(\ell)}\, W^{(\ell)} \right)
$$

where $\tilde{A} = A + I$ is the adjacency matrix with self-loops, $\tilde{D}$ is the corresponding degree matrix, $H^{(\ell)} \in \mathbb{R}^{n \times d}$ stacks node features, $W^{(\ell)}$ is the layer's trainable weight matrix, and $\sigma$ is a nonlinearity (ReLU).

Equivalently, in per-node form, GCN aggregates neighbors with degree-normalized weights:

$$
h_v^{(\ell+1)} = \sigma\!\left( \sum_{u \in \mathcal{N}(v) \cup \{v\}} \frac{1}{\sqrt{\tilde{d}_v \tilde{d}_u}} W^{(\ell)} h_u^{(\ell)} \right)
$$

The symmetric normalization prevents high-degree nodes from dominating the aggregation.

Limitations of GCN:

- **Transductive.** The full adjacency matrix is part of the model; predictions on unseen nodes require recomputing on the extended graph.
- **Fixed graph.** No native handling of graph evolution between training and inference.
- **Uniform neighbor weights.** Every neighbor of equal degree contributes equally; GAT relaxes this.

## GraphSAGE: inductive sampling

Hamilton, Ying, and Leskovec (2017) addressed the transductive limitation by sampling a fixed number of neighbors per node and learning aggregator functions that generalize to unseen nodes.

Per-layer update:

$$
h_v^{(\ell+1)} = \sigma\!\left( W^{(\ell)} \cdot \big[\, h_v^{(\ell)} \,\|\, \text{AGG}\!\left(\{\, h_u^{(\ell)} : u \in \mathcal{N}_S(v) \,\}\right) \,\big] \right)
$$

where $\|\,$ denotes concatenation and $\mathcal{N}_S(v)$ is a uniformly sampled subset of $v$'s neighbors of fixed size.

Aggregators in the original paper:

- **Mean.** Element-wise mean of neighbor features. Cheapest, often best on sparse graphs.
- **LSTM.** Treat the sampled neighbors as a sequence (random order); pass through an LSTM. Higher capacity, sensitive to ordering despite the random shuffle.
- **Pooling.** Pass each neighbor's features through an MLP, then element-wise max-pool.

GraphSAGE is the practical default for production: inductive, scalable via sampling, and the embeddings are reusable for downstream classifiers.

## Graph Attention Network (GAT)

Velickovic et al. (2018) replace the fixed normalization with learned attention over neighbors. For each pair $(v, u)$ with $u \in \mathcal{N}(v)$, compute an unnormalized attention coefficient

$$
e_{vu} = \text{LeakyReLU}\!\left( a^\top [\, W h_v \,\|\, W h_u \,] \right)
$$

then softmax-normalize over $v$'s neighborhood:

$$
\alpha_{vu} = \frac{\exp(e_{vu})}{\sum_{u' \in \mathcal{N}(v)} \exp(e_{vu'})}
$$

The aggregated message is the attention-weighted sum:

$$
h_v^{(\ell+1)} = \sigma\!\left( \sum_{u \in \mathcal{N}(v)} \alpha_{vu} W h_u \right)
$$

**Multi-head attention** runs $K$ independent attention heads in parallel and concatenates (or averages) their outputs, in direct analogy to transformer attention. GAT typically wins on heterophilic graphs (where connected nodes are not similar) where uniform GCN aggregation underperforms.

## Neighbor sampling at scale

Full-graph computation requires loading the entire $n \times n$ adjacency matrix and all node features at once. For a 100-million-node graph this is infeasible. The standard fix is mini-batch sampling.

In PyTorch Geometric the `NeighborLoader` materializes a $L$-hop subgraph around a target batch of nodes:

```python
from torch_geometric.loader import NeighborLoader

train_loader = NeighborLoader(
    data,
    num_neighbors=[25, 10],   # 25 sampled at hop 1, 10 at hop 2
    batch_size=512,
    input_nodes=data.train_mask,
)

for batch in train_loader:
    out = model(batch.x, batch.edge_index)
    loss = F.cross_entropy(out[:batch.batch_size], batch.y[:batch.batch_size])
```

The `[25, 10]` sampling fanout means the L=2 GNN sees at most $1 + 25 + 25 \cdot 10 = 276$ nodes per target. Training cost becomes constant in the size of the full graph and linear in batch size.

## A worked example: fraud detection

Consider a synthetic claim-fraud graph where nodes are claimants, witnesses, addresses, and bank accounts; edges encode "shares-address-with", "is-witness-for", "is-paid-into". The supervised label is on claimant nodes (fraud / not fraud) and is sparse: under 1 percent of claimants.

```python
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader

class FraudGNN(torch.nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden, aggr="mean")
        self.conv2 = SAGEConv(hidden, hidden, aggr="mean")
        self.head  = torch.nn.Linear(hidden, out_dim)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=0.3, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        return self.head(h)

# data is a torch_geometric.data.Data with x, edge_index, y, train_mask
model = FraudGNN(in_dim=data.num_features, hidden=64, out_dim=2)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

# class imbalance: weight the rare class
weight = torch.tensor([1.0, 50.0])

train_loader = NeighborLoader(
    data, num_neighbors=[20, 10], batch_size=256,
    input_nodes=data.train_mask, shuffle=True,
)

for epoch in range(20):
    model.train()
    for batch in train_loader:
        opt.zero_grad()
        out = model(batch.x, batch.edge_index)
        loss = F.cross_entropy(out[:batch.batch_size],
                               batch.y[:batch.batch_size],
                               weight=weight)
        loss.backward()
        opt.step()
```

The weighted loss handles the 1 percent fraud rate; without it the model collapses to predicting "not fraud" for everything. Evaluate by precision-recall AUC, not accuracy.

## Common gotchas

1. **Over-smoothing.** Stacking many GNN layers makes all node embeddings converge to similar vectors. The remedy is to keep depth at 2 to 4 layers and to use residual connections, JK-Net jumping connections, or PairNorm.
2. **Over-squashing.** Information from a distant node has to pass through a bottleneck of intermediate aggregations and gets compressed away. Diagnosed by sensitivity analysis; remedied with attention, virtual nodes, or graph rewiring.
3. **Heterophily.** Standard GNNs assume connected nodes are similar (homophily). Fraud rings often violate this: fraudsters connect to non-fraudsters as cover. Use heterogeneous GNNs (HGT, RGCN) that learn distinct aggregators per edge type.
4. **Train/eval graph leakage.** When sampling subgraphs for training, ensure validation and test nodes are not used as message-passing neighbors during training. The default split in some benchmarks does not enforce this.
5. **Cold-start nodes.** A new node with zero edges produces only a self-loop message; the GNN cannot leverage neighbors. Fall back to a content-only model (MLP on node features) for cold-start.
6. **Memory.** Full-graph PyG tensors balloon in memory. For graphs above 10M nodes use sampled training plus mini-batch inference; do not call `model(data.x, data.edge_index)` on the full graph.

## Applications

- **Fraud detection.** Heterogeneous claim-claimant-address-account graphs; node classification with heavy class weighting.
- **Recommendation.** PinSage at Pinterest, GraphSAGE at Uber Eats; user-item bipartite graphs with optional content edges.
- **Knowledge graph completion.** Predict missing relations in Wikidata-style graphs; RGCN, CompGCN.
- **Molecular property prediction.** Atoms as nodes, bonds as edges; predict solubility, toxicity, binding affinity.
- **Traffic forecasting.** Roads as edges; ST-GCN combines GCN with temporal convolutions.

## Resources

- `docs/theory/12_recommender_system_foundations.md` for the recsys connection.
- `techniques/transformers-nlp/README.md` for the attention background that GAT extends.
- PyG docs at pytorch-geometric.readthedocs.io.

## References

- Kipf, Welling, "Semi-Supervised Classification with Graph Convolutional Networks", ICLR 2017.
- Hamilton, Ying, Leskovec, "Inductive Representation Learning on Large Graphs" (GraphSAGE), NeurIPS 2017.
- Velickovic et al., "Graph Attention Networks", ICLR 2018.
- Ying et al., "Graph Convolutional Neural Networks for Web-Scale Recommender Systems" (PinSage), KDD 2018.
- Hu et al., "Heterogeneous Graph Transformer", WWW 2020.
- Hamilton, *Graph Representation Learning*, Morgan & Claypool 2020.
- Schlichtkrull et al., "Modeling Relational Data with Graph Convolutional Networks" (RGCN), ESWC 2018.
