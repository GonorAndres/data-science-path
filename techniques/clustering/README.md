# Clustering (K-Means, DBSCAN, Hierarchical Clustering)

## What it is

Clustering is an **unsupervised learning** technique that groups similar observations together without predefined labels. Unlike classification (where you know the categories in advance), clustering discovers the structure in your data by finding natural groupings. The three most important methods are: **K-Means** (partition data into k spherical clusters around centroids), **DBSCAN** (find clusters of arbitrary shape based on density), and **Hierarchical Clustering** (build a tree-like hierarchy of clusters that can be cut at any level). Each method makes different assumptions about what a "cluster" looks like, so choosing the right one depends on your data and your business question.

## Why it matters

Clustering is the backbone of **customer segmentation**, which is one of the most common and high-value analytics tasks in business. Insurers use clustering to identify policyholder segments for targeted marketing, differentiated pricing, and retention strategies. Beyond segmentation, clustering is used for anomaly detection (observations that do not belong to any cluster), data exploration (understanding the structure of a new dataset), and feature engineering (cluster membership as a new feature for supervised models). As an actuarial data scientist, you will frequently need to segment policyholders, claims, or risks into meaningful groups.

## When to use it

- **Customer segmentation**: Group customers by behavior (purchase patterns, claim history, engagement) to inform marketing and retention strategies.
- **Risk segmentation**: Identify natural risk groups in a portfolio that might not correspond to traditional rating factors.
- **Anomaly detection**: DBSCAN naturally identifies outliers as points that do not belong to any cluster.
- **Data exploration**: Understand the structure of a new dataset before building supervised models.
- **Feature engineering**: Use cluster labels as features in downstream models.
- **Image compression**: K-Means can reduce the number of colors in an image (classic demonstration).
- **Document grouping**: Cluster text documents by topic (combined with TF-IDF or embeddings).

## How it works (intuitive)

### K-Means

K-Means is the simplest and most widely used clustering algorithm. It partitions data into exactly $k$ clusters, where each observation belongs to the cluster with the nearest centroid (center point).

**Step by step:**

1. **Initialize**: Randomly place $k$ centroids in the feature space.
2. **Assign**: Assign each data point to the nearest centroid (using Euclidean distance).
3. **Update**: Recalculate each centroid as the mean of all points assigned to it.
4. **Repeat**: Steps 2-3 until centroids stop moving (convergence).

**Analogy**: Imagine scattering $k$ flags on a field full of people. Each person walks to the nearest flag. Then each flag moves to the center of its crowd. People reassign, flags move again. After a few rounds, the crowds stabilize.

**Properties**:
- Always converges (but possibly to a local minimum).
- Produces spherical (globular) clusters of roughly equal size.
- Requires specifying $k$ in advance.
- Sensitive to initialization -- run multiple times with different random starts (scikit-learn does this by default with `n_init=10`).

**K-Means++ initialization**: Instead of purely random initialization, K-Means++ places initial centroids far apart from each other, dramatically improving convergence and cluster quality.

### DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

DBSCAN defines clusters as **dense regions of points** separated by sparse regions. It does not require specifying the number of clusters and can find clusters of arbitrary shape.

**Two key parameters**:
- $\varepsilon$ (eps): The radius of the neighborhood around each point.
- `min_samples`: The minimum number of points required within the $\varepsilon$-neighborhood for a point to be considered a **core point**.

**Point types**:
- **Core point**: Has at least `min_samples` neighbors within distance $\varepsilon$.
- **Border point**: Within $\varepsilon$ of a core point but does not have enough neighbors to be a core point itself.
- **Noise point**: Neither core nor border -- it is an outlier.

**Step by step:**

1. Pick an unvisited point.
2. If it is a core point, start a new cluster. Add all points in its $\varepsilon$-neighborhood to the cluster.
3. For each new core point added, expand the cluster by adding its neighbors recursively.
4. If the starting point is not a core point, label it as noise (for now -- it might later be claimed as a border point).
5. Repeat until all points are visited.

**Analogy**: Imagine a city at night, viewed from above. Dense neighborhoods (with many buildings close together) form distinct clusters. Buildings in the suburbs that are close to a neighborhood get included as border points. Isolated farmhouses in the countryside are labeled as noise.

**Properties**:
- No need to specify the number of clusters.
- Can find clusters of arbitrary shape (not just spheres).
- Naturally identifies outliers.
- Struggles with clusters of varying densities.
- Sensitive to the choice of $\varepsilon$ and `min_samples`.

### Hierarchical Clustering

Hierarchical clustering builds a tree-like structure (a **dendrogram**) that shows how observations merge into clusters at different levels of similarity.

**Agglomerative (bottom-up) approach** (most common):

1. Start with each observation as its own cluster ($n$ clusters of size 1).
2. Find the two closest clusters and merge them.
3. Repeat until all observations are in a single cluster.
4. Cut the dendrogram at the desired level to get $k$ clusters.

**The key question**: How do you define "distance" between clusters (not just individual points)?

| Linkage Method | Distance Between Clusters | Behavior |
|---------------|--------------------------|----------|
| Single | Min distance between any pair of points | Finds elongated, chain-like clusters. Sensitive to noise. |
| Complete | Max distance between any pair of points | Finds compact, spherical clusters. |
| Average (UPGMA) | Average distance between all pairs | Balanced between single and complete. |
| Ward | Minimizes within-cluster variance | Similar to K-Means. Produces compact, equal-sized clusters. |

**Analogy**: Imagine building a family tree in reverse. Start with individual people. The two most similar people merge into a pair. Then the most similar pair/person merges next. The dendrogram shows the entire history of merging. You can "cut" the tree at any height to get as many or as few groups as you want.

**Properties**:
- Produces a dendrogram that provides a rich, multi-level view of cluster structure.
- No need to specify $k$ in advance (you choose after seeing the dendrogram).
- Computationally expensive: $O(n^2)$ memory and $O(n^3)$ time for naive implementations. Impractical for very large datasets.
- Once a merge is made, it cannot be undone (greedy algorithm).

## Key concepts

### Distance Metrics

The choice of distance metric profoundly affects clustering results:

| Metric | Formula | When to use |
|--------|---------|-------------|
| Euclidean | $\sqrt{\sum(x_i - y_i)^2}$ | Default for most applications. Assumes equal importance of all features. |
| Manhattan | $\sum\|x_i - y_i\|$ | When features represent different "costs" or movements along axes. |
| Cosine | $1 - \frac{x \cdot y}{\|x\| \|y\|}$ | Text data, when direction matters more than magnitude. |
| Gower | Mixed distances | Mixed data types (numeric + categorical). Available in R's `cluster` package. |

**Critical**: Always **standardize features** before clustering (unless you intentionally want certain features to dominate). If age ranges from 20-80 and income from 20,000-200,000, income will dominate Euclidean distance.

### Choosing K (Number of Clusters)

For K-Means and hierarchical clustering, you must choose $k$. Common approaches:

**1. Elbow Method**: Plot within-cluster sum of squares (inertia) vs. $k$. Look for the "elbow" where the rate of decrease sharply changes. Subjective but widely used.

**2. Silhouette Score**: For each point, measure:
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

Where:
- $a(i)$ = average distance to points in the same cluster (compactness).
- $b(i)$ = average distance to points in the nearest neighboring cluster (separation).
- $s(i) \in [-1, 1]$: 1 = well-clustered, 0 = on the border, -1 = probably in the wrong cluster.

Average the silhouette score across all points. Higher is better. Also plot the silhouette diagram (per-cluster silhouette widths) to check for uneven or misshapen clusters.

**3. Gap Statistic**: Compares within-cluster dispersion to what you would expect from a uniform random distribution. More principled than the elbow method.

**4. Dendrogram (for hierarchical)**: Visually inspect the dendrogram for natural cut points -- large vertical gaps suggest meaningful cluster separations.

### Choosing Epsilon for DBSCAN

Use the **k-distance plot**:

1. For each point, compute the distance to its $k$-th nearest neighbor (where $k$ = `min_samples`).
2. Sort these distances and plot them.
3. Look for the "knee" -- the point where the curve bends sharply. This is a good candidate for $\varepsilon$.

### Cluster Validation

| Metric | Type | What it measures |
|--------|------|-----------------|
| Silhouette Score | Internal | Compactness vs. separation |
| Calinski-Harabasz Index | Internal | Ratio of between-cluster to within-cluster variance |
| Davies-Bouldin Index | Internal | Average similarity between each cluster and its most similar one (lower is better) |
| Adjusted Rand Index | External | Agreement with ground-truth labels (when available) |
| Normalized Mutual Information | External | Information shared between clustering and ground truth |

**"Internal" metrics** do not require ground truth. **"External" metrics** require known labels and are used to compare clustering methods or validate against expert judgment.

## Python implementation

```python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import numpy as np

# Always scale first
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- K-Means ---
# Elbow method
inertias = []
sil_scores = []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, km.labels_))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(K_range, inertias, 'bo-')
ax1.set_xlabel('k'); ax1.set_ylabel('Inertia'); ax1.set_title('Elbow Method')
ax2.plot(K_range, sil_scores, 'ro-')
ax2.set_xlabel('k'); ax2.set_ylabel('Silhouette Score'); ax2.set_title('Silhouette Method')

# Fit best k
best_k = 4
km_final = KMeans(n_clusters=best_k, n_init=10, random_state=42)
labels_km = km_final.fit_predict(X_scaled)

# --- DBSCAN ---
# K-distance plot to find eps
nn = NearestNeighbors(n_neighbors=5)
nn.fit(X_scaled)
distances, _ = nn.kneighbors(X_scaled)
distances = np.sort(distances[:, -1])
plt.plot(distances)
plt.xlabel('Points (sorted)'); plt.ylabel('5-NN Distance')
plt.title('K-Distance Plot')

dbscan = DBSCAN(eps=0.5, min_samples=5)
labels_db = dbscan.fit_predict(X_scaled)
n_clusters_db = len(set(labels_db)) - (1 if -1 in labels_db else 0)
n_noise = list(labels_db).count(-1)
print(f"DBSCAN: {n_clusters_db} clusters, {n_noise} noise points")

# --- Hierarchical Clustering ---
Z = linkage(X_scaled, method='ward')
plt.figure(figsize=(12, 6))
dendrogram(Z, truncate_mode='lastp', p=30)
plt.title('Dendrogram (Ward Linkage)')

hc = AgglomerativeClustering(n_clusters=4, linkage='ward')
labels_hc = hc.fit_predict(X_scaled)

# --- Evaluation ---
for name, labels in [('K-Means', labels_km), ('Hierarchical', labels_hc)]:
    sil = silhouette_score(X_scaled, labels)
    ch = calinski_harabasz_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)
    print(f"{name}: Silhouette={sil:.3f}, CH={ch:.1f}, DB={db:.3f}")
```

## R implementation (if applicable)

```r
library(cluster)
library(factoextra)
library(dbscan)

# Scale data
X_scaled <- scale(X)

# K-Means with elbow and silhouette
fviz_nbclust(X_scaled, kmeans, method = "wss")  # Elbow
fviz_nbclust(X_scaled, kmeans, method = "silhouette")  # Silhouette
fviz_nbclust(X_scaled, kmeans, method = "gap_stat")  # Gap statistic

km_result <- kmeans(X_scaled, centers = 4, nstart = 25)
fviz_cluster(km_result, data = X_scaled)

# Hierarchical
hc_result <- hclust(dist(X_scaled), method = "ward.D2")
plot(hc_result)
rect.hclust(hc_result, k = 4)

# DBSCAN
kNNdistplot(X_scaled, k = 5)  # K-distance plot
db_result <- dbscan(X_scaled, eps = 0.5, minPts = 5)
fviz_cluster(list(data = X_scaled, cluster = db_result$cluster))
```

## Gotchas & lessons learned

1. **Always scale your features.** This is the single most important preprocessing step for distance-based methods. Unscaled data produces clusters dominated by the highest-magnitude feature.

2. **K-Means assumes spherical, equally-sized clusters.** If your true clusters are elongated, crescent-shaped, or vary in density, K-Means will give misleading results. Use DBSCAN for non-spherical clusters.

3. **There is no "correct" number of clusters.** The elbow method, silhouette score, and gap statistic are guides, not answers. The best $k$ also depends on business utility -- 3 segments might be actionable, 15 might not be.

4. **Clustering is not deterministic (for K-Means).** Different random initializations can give different results. Always set `random_state` for reproducibility and use multiple initializations (`n_init >= 10`).

5. **DBSCAN does not work well with varying-density clusters.** If one cluster is dense and another is sparse, no single $\varepsilon$ works well. Consider HDBSCAN (hierarchical DBSCAN) which adapts to varying densities.

6. **High-dimensional data breaks distance-based clustering.** In high dimensions, distances between all pairs of points become similar (curse of dimensionality). Apply PCA or UMAP before clustering in high-dimensional spaces.

7. **Cluster labels are arbitrary.** K-Means cluster 0 in one run might correspond to cluster 3 in another run. Labels have no inherent meaning -- you must interpret clusters by examining their feature distributions.

8. **Validate clusters with domain knowledge.** Statistical metrics tell you about cluster quality but not about business relevance. Profile each cluster: compute mean/median of each feature per cluster. Do the resulting segments make intuitive sense?

9. **Beware of clustering artifacts from outliers.** A few extreme outliers can distort K-Means centroids or create singleton clusters. Consider removing or capping outliers before clustering, or use a method robust to outliers (DBSCAN, trimmed K-Means).

10. **In customer segmentation**, the goal is **actionable segments**, not statistically perfect clusters. A slightly sub-optimal clustering that maps to clear business strategies (e.g., "high-value loyalists," "price-sensitive switchers") is more useful than a statistically optimal one that defies interpretation.

## Resources

- **scikit-learn Clustering documentation**: [Clustering](https://scikit-learn.org/stable/modules/clustering.html) -- Comprehensive with comparison of all methods.
- **An Introduction to Statistical Learning (ISLR)**, Chapter 12 -- Covers K-Means and hierarchical clustering clearly.
- **scikit-learn "Selecting the number of clusters"**: [Example gallery](https://scikit-learn.org/stable/auto_examples/cluster/) -- Practical code examples.
- **HDBSCAN documentation**: [hdbscan.readthedocs.io](https://hdbscan.readthedocs.io/) -- Excellent for varying-density clusters. Arguably the best general-purpose clustering algorithm.
- **Kassambara, "Practical Guide to Cluster Analysis in R"**: [datanovia.com](https://www.datanovia.com/en/lessons/cluster-analysis-in-r/) -- Thorough R-focused guide.
- **StatQuest (Josh Starmer)**: YouTube videos on K-Means, hierarchical clustering, and DBSCAN.
