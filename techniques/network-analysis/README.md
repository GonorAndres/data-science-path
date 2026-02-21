# Network Analysis

## What it is

Network analysis (also called graph analysis) studies relationships between entities by representing them as a graph: entities are **nodes** (also called vertices), and the connections between them are **edges** (also called links). If you have data about who interacts with whom, what is connected to what, or how things flow between entities, network analysis gives you the tools to measure, visualize, and model that structure. The core Python library for this is **NetworkX**, which provides data structures for graphs and a rich set of algorithms for computing network metrics, detecting communities, and analyzing connectivity.

## Why it matters

Insurance and actuarial work involves complex webs of relationships that traditional tabular analysis misses:

- **Fraud ring detection**: Fraudulent claims often involve networks of collaborating actors -- a claimant, a doctor, a lawyer, and a body shop that repeatedly work together. Network analysis can reveal these hidden rings by identifying unusually dense connections among entities that should be independent.
- **Referral networks**: Understanding how policyholders are connected through referral programs, family relationships, or employer groups helps predict retention and acquisition patterns.
- **Reinsurance networks**: The web of cession and retrocession between insurers and reinsurers creates systemic risk. Network analysis quantifies this interconnectedness.
- **Provider networks**: In health insurance, analyzing the network of doctors, hospitals, and specialists that patients use reveals utilization patterns and potential waste.
- **Agent/broker networks**: Understanding the relationships between agents, brokers, and policyholders helps identify key distribution partners and their influence.
- **Claims supply chain**: The flow from incident to adjuster to appraiser to repair shop to payment creates a process network that can be analyzed for bottlenecks and anomalies.

Network analysis lets you see structure that is invisible in spreadsheets.

## When to use it

- You have relational data: who called whom, who referred whom, who shares an address/phone/bank account with whom, which providers treated which claimants.
- You suspect **coordinated fraud** and want to find clusters of entities with suspiciously dense interconnections.
- You want to identify **influential nodes**: which agents drive the most business? Which providers are central to the healthcare network?
- You need to analyze **information flow or contagion**: how quickly does a new product spread through a referral network? How does a reinsurer's failure cascade?
- You want to **segment entities** based on their network position rather than (or in addition to) their individual attributes.
- You have a graph that can be extracted from transactional data and want to derive features for a downstream ML model (graph feature engineering).

## How it works (intuitive)

### Graph basics

**Nodes**: The entities in your network. People, companies, policies, claims, addresses, phone numbers -- anything.

**Edges**: The relationships between nodes. Can be:
- **Undirected**: A and B are connected (co-occurrence, shared attribute). Example: two claimants share the same address.
- **Directed**: A connects to B (but not necessarily the reverse). Example: agent A referred customer B.
- **Weighted**: Edges have a strength/weight. Example: two providers collaborated on 15 claims (weight=15) vs. 2 claims (weight=2).

**Analogy**: Think of a social network. People are nodes. Friendships are undirected edges ("we are friends"). A follow on Twitter is a directed edge ("I follow you, you don't follow me"). The number of messages exchanged is the edge weight.

### Key graph types

| Type | Description | Example |
|------|-------------|---------|
| **Undirected** | Edges have no direction | Shared address network |
| **Directed** | Edges point from source to target | Referral network, claims flow |
| **Bipartite** | Two types of nodes, edges only between types | Claimant-Provider network |
| **Weighted** | Edges have numerical values | Transaction amount network |
| **Multigraph** | Multiple edges between same pair of nodes | Multiple shared attributes |

### Building a fraud detection network

A practical example of building and analyzing a network for insurance fraud:

1. **Define nodes**: Claimants, providers (doctors, lawyers, body shops), addresses, phone numbers, bank accounts.

2. **Define edges**: Two nodes are connected if they share an attribute or transaction:
   - Same address
   - Same phone number
   - Same bank account for payment
   - Provider treated claimant
   - Claims filed on same day at same location

3. **Build the graph**: Create nodes and edges from your data.

4. **Compute metrics**: For each node and subgraph, compute centrality, clustering, and density metrics.

5. **Detect communities**: Find groups of densely connected nodes (potential fraud rings).

6. **Score and investigate**: Flag communities with anomalous patterns (high density, shared attributes, unusual claim patterns) for investigation.

### Network metrics explained

**Degree**: The number of connections a node has. In a fraud network, a high-degree node (connected to many claimants, providers, and addresses) is a potential hub of coordinated activity.

**Degree centrality**: Degree normalized by the maximum possible connections: $C_D(v) = \frac{\deg(v)}{n-1}$

**Betweenness centrality**: How often a node sits on the shortest path between other pairs of nodes. High betweenness means the node is a "bridge" or "broker" in the network -- removing it would disconnect groups.

$$C_B(v) = \sum_{s \neq v \neq t} \frac{\sigma_{st}(v)}{\sigma_{st}}$$

Where $\sigma_{st}$ is the total number of shortest paths between $s$ and $t$, and $\sigma_{st}(v)$ is the number that pass through $v$.

**Analogy**: In a criminal network, the person with high betweenness is the "middleman" who connects different parts of the operation. Removing them disrupts communication between groups.

**Closeness centrality**: How close a node is to all other nodes (inverse of average shortest path length). High closeness means a node can quickly reach everyone else.

$$C_C(v) = \frac{n-1}{\sum_{u \neq v} d(v, u)}$$

**Eigenvector centrality**: A node is important if it is connected to other important nodes. Google's PageRank is a variant of eigenvector centrality. In a referral network, an agent with high eigenvector centrality is connected to other well-connected agents.

**Clustering coefficient**: The fraction of a node's neighbors that are also connected to each other. High clustering means the node's contacts form a tight-knit group.

$$C_i = \frac{2 \cdot |\text{edges among neighbors of } i|}{k_i (k_i - 1)}$$

Where $k_i$ is the degree of node $i$.

**Analogy**: If your friends are also friends with each other, your clustering coefficient is high. In fraud detection, a claimant whose providers all know each other (high clustering) is more suspicious than one whose providers are independent.

**Graph density**: The ratio of actual edges to possible edges. A dense subgraph (many connections relative to nodes) might indicate coordinated activity.

$$\text{Density} = \frac{2|E|}{|V|(|V|-1)}$$

### Community detection

Community detection algorithms find groups of nodes that are more densely connected to each other than to the rest of the network. This is the network equivalent of clustering.

**Louvain algorithm**: The most popular method. It optimizes modularity -- a measure of how much the network departs from random connections. Fast and scalable.

**Girvan-Newman**: Iteratively removes edges with the highest betweenness centrality, breaking the network into communities. Slower but provides a hierarchical decomposition.

**Label propagation**: Each node adopts the most common label among its neighbors, iteratively. Fast and simple but non-deterministic.

**Modularity**: The quality metric for community detection:

$$Q = \frac{1}{2m} \sum_{ij} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)$$

Where $A_{ij}$ is the adjacency matrix, $k_i$ is the degree of node $i$, $m$ is the total number of edges, and $\delta(c_i, c_j) = 1$ if nodes $i$ and $j$ are in the same community. Higher modularity = better community structure.

## Key concepts

### Adjacency matrix

A graph can be represented as an adjacency matrix $A$ where $A_{ij} = 1$ if nodes $i$ and $j$ are connected (or $A_{ij} = w_{ij}$ for weighted graphs). For undirected graphs, $A$ is symmetric.

### Graph traversal algorithms

- **Breadth-First Search (BFS)**: Explore all neighbors at distance 1, then distance 2, etc. Useful for finding shortest paths and connected components.
- **Depth-First Search (DFS)**: Explore as far as possible along each branch before backtracking. Useful for finding all connected components and cycles.

### Connected components

A connected component is a maximal set of nodes where every node can reach every other node via some path. In fraud analysis, each connected component is an independent "cluster" of related entities. Very large components might indicate a fraud network; very small components (isolated pairs) are usually normal.

### Bipartite networks

Many insurance networks are naturally bipartite: claimants on one side, providers on the other. Edges connect claimants to the providers they used. Analyzing the "projection" (connecting two claimants if they share a provider) reveals indirect relationships.

### Graph feature engineering

Network metrics can be used as features in a traditional ML model:

| Feature | Description | Fraud signal |
|---------|-------------|-------------|
| Degree | Number of connections | Hub of activity |
| Betweenness centrality | Bridge between groups | Middleman/coordinator |
| Clustering coefficient | Cliquishness of neighborhood | Tight-knit group |
| Component size | Size of connected component | Large ring |
| PageRank | Importance based on connections | Central player |
| Average neighbor degree | How connected neighbors are | Part of dense cluster |
| Number of triangles | How many closed triads | Coordinated relationships |

Add these as features alongside traditional tabular features (claim amount, policy tenure, etc.) for a richer fraud prediction model.

## Python implementation

### Building and exploring graphs with NetworkX

```python
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

# Create a graph from edge list
G = nx.Graph()

# Add edges from claims-provider data
# Each row: (claimant_id, provider_id, claim_amount)
edges = [
    ('C001', 'P01', {'weight': 5000}),
    ('C001', 'P02', {'weight': 3000}),
    ('C002', 'P01', {'weight': 8000}),
    ('C002', 'P03', {'weight': 2000}),
    ('C003', 'P01', {'weight': 4500}),
    ('C003', 'P02', {'weight': 6000}),
]
G.add_edges_from(edges)

# From a pandas DataFrame
df_edges = pd.DataFrame({
    'source': ['C001', 'C002', 'C003', 'C001'],
    'target': ['P01', 'P01', 'P01', 'P02'],
    'weight': [5000, 8000, 4500, 3000]
})
G = nx.from_pandas_edgelist(df_edges, 'source', 'target',
                             edge_attr='weight')

# Basic graph info
print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
print(f"Density: {nx.density(G):.4f}")
print(f"Connected components: {nx.number_connected_components(G)}")
```

### Computing centrality metrics

```python
# Degree centrality
degree_cent = nx.degree_centrality(G)
print("Top 5 by degree:", sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:5])

# Betweenness centrality
between_cent = nx.betweenness_centrality(G, weight='weight')
print("Top 5 by betweenness:", sorted(between_cent.items(), key=lambda x: x[1], reverse=True)[:5])

# Eigenvector centrality
eigen_cent = nx.eigenvector_centrality(G, max_iter=1000)
print("Top 5 by eigenvector:", sorted(eigen_cent.items(), key=lambda x: x[1], reverse=True)[:5])

# Clustering coefficient
clustering = nx.clustering(G)
print("Average clustering:", nx.average_clustering(G))

# PageRank
pagerank = nx.pagerank(G, weight='weight')
print("Top 5 by PageRank:", sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:5])

# Compile into a DataFrame for ML feature engineering
node_features = pd.DataFrame({
    'node': list(G.nodes()),
    'degree': [G.degree(n) for n in G.nodes()],
    'degree_centrality': [degree_cent[n] for n in G.nodes()],
    'betweenness': [between_cent[n] for n in G.nodes()],
    'eigenvector': [eigen_cent[n] for n in G.nodes()],
    'clustering': [clustering[n] for n in G.nodes()],
    'pagerank': [pagerank[n] for n in G.nodes()]
})
```

### Community detection

```python
# Louvain community detection (requires python-louvain or community package)
import community as community_louvain

# Best partition
partition = community_louvain.best_partition(G, weight='weight')
modularity = community_louvain.modularity(partition, G, weight='weight')
print(f"Modularity: {modularity:.4f}")
print(f"Number of communities: {len(set(partition.values()))}")

# Add community labels to nodes
nx.set_node_attributes(G, partition, 'community')

# Analyze each community
communities = {}
for node, comm_id in partition.items():
    communities.setdefault(comm_id, []).append(node)

for comm_id, members in communities.items():
    subgraph = G.subgraph(members)
    print(f"Community {comm_id}: {len(members)} nodes, "
          f"density={nx.density(subgraph):.3f}, "
          f"edges={subgraph.number_of_edges()}")

# Girvan-Newman (hierarchical)
from networkx.algorithms.community import girvan_newman
communities_gn = girvan_newman(G)
first_level = next(communities_gn)
print(f"Girvan-Newman first split: {[len(c) for c in first_level]} nodes per community")

# Label propagation
from networkx.algorithms.community import label_propagation_communities
communities_lp = list(label_propagation_communities(G))
print(f"Label propagation found {len(communities_lp)} communities")
```

### Fraud ring detection example

```python
def build_claims_network(claims_df):
    """
    Build a network where claimants are connected if they share
    a provider, address, phone, or bank account.
    """
    G = nx.Graph()

    # Add all claimants as nodes
    for _, row in claims_df.iterrows():
        G.add_node(row['claimant_id'], node_type='claimant')

    # Connect claimants sharing the same provider
    for provider, group in claims_df.groupby('provider_id'):
        claimants = group['claimant_id'].unique()
        for i in range(len(claimants)):
            for j in range(i+1, len(claimants)):
                if G.has_edge(claimants[i], claimants[j]):
                    G[claimants[i]][claimants[j]]['weight'] += 1
                    G[claimants[i]][claimants[j]]['shared_providers'].append(provider)
                else:
                    G.add_edge(claimants[i], claimants[j],
                             weight=1,
                             shared_providers=[provider])

    # Similarly for shared addresses, phones, bank accounts...
    return G

def flag_suspicious_communities(G, min_density=0.5, min_size=3):
    """Find densely connected communities that may indicate fraud rings."""
    partition = community_louvain.best_partition(G)
    communities = {}
    for node, comm_id in partition.items():
        communities.setdefault(comm_id, []).append(node)

    suspicious = []
    for comm_id, members in communities.items():
        if len(members) < min_size:
            continue
        subgraph = G.subgraph(members)
        density = nx.density(subgraph)
        avg_clustering = nx.average_clustering(subgraph)

        if density >= min_density:
            suspicious.append({
                'community_id': comm_id,
                'members': members,
                'size': len(members),
                'density': density,
                'avg_clustering': avg_clustering,
                'total_edges': subgraph.number_of_edges()
            })

    return sorted(suspicious, key=lambda x: x['density'], reverse=True)
```

### Visualization

```python
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Basic network visualization
plt.figure(figsize=(12, 8))

# Layout algorithms
pos = nx.spring_layout(G, seed=42)        # Force-directed (good default)
# pos = nx.kamada_kawai_layout(G)          # Energy-based (cleaner for small graphs)
# pos = nx.circular_layout(G)             # Circular

# Color by community
colors = [partition[node] for node in G.nodes()]
node_sizes = [300 * degree_cent[node] + 100 for node in G.nodes()]

nx.draw(G, pos,
        node_color=colors,
        node_size=node_sizes,
        cmap=cm.Set3,
        with_labels=True,
        font_size=8,
        edge_color='gray',
        alpha=0.8)
plt.title('Claims Network -- Communities and Node Importance')
plt.tight_layout()
plt.show()

# For large networks, use interactive visualization
# pyvis, Gephi (export with nx.write_gexf), or plotly
from pyvis.network import Network

nt = Network(height='600px', width='100%', notebook=True)
nt.from_nx(G)
nt.show('network.html')
```

### Bipartite network projection

```python
from networkx.algorithms import bipartite

# Create a bipartite graph: claimants <-> providers
B = nx.Graph()
B.add_nodes_from(claimant_ids, bipartite=0)
B.add_nodes_from(provider_ids, bipartite=1)
B.add_edges_from(claimant_provider_pairs)

# Project to claimant-claimant network (connected if they share a provider)
claimant_network = bipartite.weighted_projected_graph(B, claimant_ids)

# Project to provider-provider network (connected if they share a claimant)
provider_network = bipartite.weighted_projected_graph(B, provider_ids)
```

## Gotchas & lessons learned

### Network construction is the hardest part
Defining what counts as a "connection" is a critical design decision. Too liberal (any shared attribute), and your network is a giant hairball where everything connects to everything. Too strict, and you miss real relationships. Start conservative, add edge types iteratively, and validate with known fraud cases.

### Large networks require approximate algorithms
NetworkX is great for learning but stores everything in memory and uses Python loops. For networks with millions of nodes, use:
- **graph-tool** (C++ backend, much faster)
- **igraph** (C backend, available in Python and R)
- **Neo4j** (graph database for persistent storage and querying)
- **DGL or PyG** (GPU-accelerated, for graph neural networks)

### Visualization breaks down at scale
You cannot meaningfully visualize a network with more than ~1,000 nodes. For large networks:
- Visualize only the subgraph around suspicious nodes
- Use aggregation (show communities as super-nodes)
- Use interactive tools (Gephi, pyvis) with filtering

### Spurious connections from common attributes
Common addresses (apartment buildings), shared phone numbers (business phones), and popular providers (large hospitals) create connections that are coincidental, not suspicious. Filter out or down-weight connections through very high-degree "hub" nodes, or remove known business entities.

### Temporal dimension matters
A network built from 10 years of data may contain many connections that are not contemporaneous. Two claimants sharing a provider 5 years apart is less suspicious than two sharing a provider on the same day. Add temporal constraints to edge creation: connections only count if they occurred within a reasonable time window.

### Degree distribution follows a power law
Most real networks have a power-law degree distribution: most nodes have few connections, a few nodes have many. These high-degree nodes ("hubs") dominate many metrics. Be aware of this when interpreting centrality scores -- a high-degree node might just be a large hospital, not a fraud ring leader.

### Community detection is non-deterministic
The Louvain algorithm (and others) may produce different results on different runs. Run the algorithm multiple times and check stability. If community assignments change drastically, the community structure may be weak.

### Feature engineering is where the value lies
The most practical value of network analysis in insurance is not the pretty visualizations but the features you extract (degree, centrality, community membership, clustering coefficient) and feed into a traditional ML model (XGBoost, logistic regression). Networks enrich your feature space, not replace it.

### Privacy and ethical considerations
Building networks from personal data (shared addresses, phone numbers, social connections) raises privacy concerns. Ensure compliance with data protection regulations (GDPR, CCPA). Network analysis can also inadvertently profile communities in ways that raise fairness concerns.

## Resources

- [NetworkX documentation](https://networkx.org/documentation/stable/) -- The standard Python library for network analysis
- [python-louvain (community detection)](https://github.com/taynaud/python-louvain) -- Louvain community detection for NetworkX
- [pyvis](https://pyvis.readthedocs.io/) -- Interactive network visualization
- [igraph Python](https://python.igraph.org/) -- Faster alternative to NetworkX for large graphs
- [Barabasi (2016) - Network Science](http://networksciencebook.com/) -- Free online textbook, excellent introduction
- [Newman (2018) - Networks: An Introduction](https://global.oup.com/academic/product/networks-9780198805090) -- Comprehensive academic textbook
- [Graph Analytics for Fraud Detection (Neo4j)](https://neo4j.com/use-cases/fraud-detection/) -- Industry applications
- [Social Network Analysis: Methods and Applications (Wasserman & Faust)](https://www.cambridge.org/core/books/social-network-analysis/90030086891EB3491D096034684EFFB8) -- Classic reference
- [CAS Fraud Analytics Working Party](https://www.casact.org/) -- Insurance-specific fraud analytics resources
- [Gephi](https://gephi.org/) -- Open-source network visualization and exploration tool
