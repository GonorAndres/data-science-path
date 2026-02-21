# Dimensionality Reduction (PCA, t-SNE, UMAP)

## What it is

Dimensionality reduction transforms high-dimensional data into a lower-dimensional representation while preserving as much meaningful structure as possible. When your dataset has dozens or hundreds of features, these techniques help you **visualize** the data (project it to 2D or 3D), **remove noise** (discard dimensions that carry mostly noise), **speed up downstream models** (fewer features = faster training), and **mitigate the curse of dimensionality** (many algorithms degrade in high-dimensional spaces). This README covers three methods: **PCA** (Principal Component Analysis, a linear projection that maximizes variance), **t-SNE** (t-Distributed Stochastic Neighbor Embedding, a non-linear method optimized for 2D/3D visualization), and **UMAP** (Uniform Manifold Approximation and Projection, a newer non-linear method that balances local and global structure).

## Why it matters

Real-world actuarial and data science datasets often have many features: policyholder demographics, claim history, behavioral signals, geographic variables, and more. High dimensionality creates several problems: models overfit more easily, distances between points become less meaningful, and human intuition breaks down (you cannot visualize 50 dimensions). Dimensionality reduction addresses these problems. PCA is essential for feature engineering, noise reduction, and multicollinearity handling. t-SNE and UMAP are invaluable for exploratory data analysis -- they let you "see" the structure of your data in a way that no amount of summary statistics can match.

## When to use it

- **PCA for preprocessing**: Reduce features before feeding into a model (e.g., reduce 100 correlated features to 15 principal components before running logistic regression or clustering).
- **PCA for multicollinearity**: When your features are highly correlated and you want orthogonal (uncorrelated) inputs for regression.
- **PCA for noise reduction**: The first few principal components capture signal; later components capture noise. Truncating is a form of denoising.
- **t-SNE for visualization**: When you want to visualize high-dimensional data in 2D to look for clusters, outliers, or group separations. Best for presentation-quality exploratory plots.
- **UMAP for visualization + preprocessing**: When you want a faster alternative to t-SNE that also preserves global structure better. Can also be used to create reduced features for downstream models (unlike t-SNE).
- **General rule**: Use PCA when you need a linear, interpretable, invertible transformation. Use t-SNE/UMAP when you need non-linear visualization of complex structure.

## How it works (intuitive)

### PCA (Principal Component Analysis)

PCA finds the directions (axes) along which the data varies the most and projects the data onto those directions.

**Analogy**: Imagine you are photographing a 3D object. You want to take a single 2D photograph that captures as much of the object's shape as possible. You would rotate the object until the most informative angle faces the camera. PCA does this mathematically -- it finds the angles (principal components) that capture the most variance.

**Step by step:**

1. **Center the data**: Subtract the mean of each feature so the data is centered at the origin.
2. **Compute the covariance matrix**: This $p \times p$ matrix captures how each pair of features varies together.
3. **Find the eigenvectors and eigenvalues** of the covariance matrix:
   - Each **eigenvector** defines a direction (a principal component axis).
   - Each **eigenvalue** tells you how much variance is captured along that direction.
4. **Sort** eigenvectors by their eigenvalues (largest first).
5. **Project** the data onto the top $k$ eigenvectors to get the $k$-dimensional representation.

**Key formula**: The covariance matrix is:

$$C = \frac{1}{n-1} X^T X$$

Where $X$ is the centered data matrix. The eigendecomposition gives:

$$C = V \Lambda V^T$$

Where $V$ contains the eigenvectors (principal component directions) and $\Lambda$ is diagonal with eigenvalues (variances along each PC).

**Variance explained**: The proportion of variance captured by the $i$-th component is:

$$\text{Explained variance ratio}_i = \frac{\lambda_i}{\sum_{j=1}^{p} \lambda_j}$$

A **scree plot** (eigenvalues vs. component number) or a **cumulative explained variance plot** helps you choose how many components to keep. Common rule: keep enough components to explain 80-95% of total variance.

**Properties of PCA**:
- **Linear**: PCA only captures linear relationships. Non-linear structure is missed.
- **Orthogonal**: Principal components are uncorrelated with each other.
- **Invertible**: You can approximately reconstruct the original data from the reduced representation.
- **Deterministic**: Same data always gives the same result (no random initialization).
- **Sensitive to scale**: Features with larger variance dominate. Always standardize first (unless features are already on the same scale).

### t-SNE (t-Distributed Stochastic Neighbor Embedding)

t-SNE is a non-linear dimensionality reduction method specifically designed for **visualization** of high-dimensional data in 2D or 3D. It preserves **local structure** -- points that are close together in high dimensions remain close in the low-dimensional plot.

**How it works (simplified):**

1. **In high-dimensional space**: For each pair of points, compute a similarity based on Gaussian distances. Nearby points get high similarity; distant points get low similarity. The bandwidth of the Gaussian is controlled by the **perplexity** parameter (roughly: "how many effective neighbors should each point consider").

2. **In low-dimensional space**: Place points randomly in 2D. Compute similarities using a **Student's t-distribution** (heavier tails than Gaussian).

3. **Minimize the mismatch**: Move the 2D points around until the low-dimensional similarities match the high-dimensional similarities as closely as possible (minimizing KL divergence).

**Why the t-distribution?** In high dimensions, moderate distances and large distances are hard to distinguish. The Gaussian would crowd moderate-distance points together. The heavy-tailed t-distribution gives more room for moderately distant points to spread out, creating cleaner visual separations.

**Analogy**: Imagine you have a social network with 1000 people. t-SNE creates a 2D seating chart where people who are close friends sit near each other. It does not try to preserve the exact distances between people who barely know each other -- it focuses on keeping friend groups intact.

**Key parameter -- Perplexity**: Roughly the number of effective neighbors. Typical range: 5-50. Low perplexity focuses on very local structure; high perplexity considers more global structure. Always try multiple perplexity values and compare.

### UMAP (Uniform Manifold Approximation and Projection)

UMAP is a newer non-linear method that often produces better visualizations than t-SNE, runs faster, and better preserves **global structure** (the relative positions of clusters).

**How it works (simplified):**

1. **Build a neighborhood graph**: For each point, find its $k$ nearest neighbors. Weight edges by distance (using a fuzzy set formulation).
2. **Construct a low-dimensional layout**: Initialize points in low dimensions and optimize the layout so that the neighborhood graph structure is preserved.
3. **The key difference from t-SNE**: UMAP uses a cross-entropy loss (instead of KL divergence) and a different graph-construction approach that better preserves both local and global relationships.

**Key parameters**:
- `n_neighbors`: Similar to perplexity. Controls how local vs. global the embedding is. Low values = tight clusters, high values = broader view.
- `min_dist`: Controls how tightly points are packed. Low values = tighter clusters. High values = more spread out.
- `metric`: The distance metric in the original space (Euclidean, cosine, etc.).

**Properties**:
- Faster than t-SNE, especially for large datasets.
- Better preserves global structure (cluster positions relative to each other are more meaningful).
- Can be used for dimensionality reduction beyond just visualization (embed into higher than 2D).
- Has a well-defined `transform` method for new data (t-SNE does not).
- Still non-deterministic (random initialization).

### Comparison Table

| Property | PCA | t-SNE | UMAP |
|----------|-----|-------|------|
| Type | Linear | Non-linear | Non-linear |
| Preserves | Global variance | Local structure | Local + some global |
| Deterministic | Yes | No | No |
| Speed (large n) | Fast | Slow ($O(n^2)$) | Moderate |
| Invertible | Yes | No | Approximately |
| Transform new data | Yes | No (must refit) | Yes |
| Good for preprocessing | Yes | No (visualization only) | Yes (cautiously) |
| Interpretable axes | Yes (loadings) | No | No |

## Key concepts

### Eigenvalues and Eigenvectors

An eigenvector of a matrix $A$ is a vector $v$ that, when multiplied by $A$, only changes in scale (not direction):

$$Av = \lambda v$$

Where $\lambda$ is the eigenvalue (the scaling factor).

In PCA:
- **Eigenvectors** = the directions of the principal components.
- **Eigenvalues** = the amount of variance along each direction.
- The eigenvector with the largest eigenvalue is the first principal component (the direction of maximum variance).

### Loadings

PCA **loadings** tell you how much each original feature contributes to each principal component:

$$\text{Loading}_{ij} = v_{ij} \cdot \sqrt{\lambda_j}$$

Where $v_{ij}$ is the $i$-th element of the $j$-th eigenvector and $\lambda_j$ is its eigenvalue.

Examining loadings is how you interpret what a principal component "means." For example, if PC1 has high loadings on income, assets, and credit score, you might call it a "financial strength" component.

### The Manifold Assumption

t-SNE and UMAP are based on the **manifold hypothesis**: high-dimensional data often lies on or near a low-dimensional surface (manifold) embedded in the high-dimensional space.

**Analogy**: The surface of the Earth is a 2D manifold embedded in 3D space. Even though we live in 3D, most of our movement is constrained to this 2D surface. Similarly, data with 100 features might actually vary along only 5-10 meaningful dimensions.

### Choosing the Number of Components (PCA)

1. **Scree plot**: Plot eigenvalues vs. component number. Look for an "elbow."
2. **Cumulative explained variance**: Keep enough components to explain 80-95% of variance.
3. **Kaiser criterion**: Keep components with eigenvalues > 1 (when using standardized data).
4. **Cross-validation**: In supervised settings, use CV to find the number of PCA components that optimizes downstream model performance.

## Python implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap  # pip install umap-learn

# Always standardize for PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- PCA ---
# Full PCA to examine variance explained
pca_full = PCA()
pca_full.fit(X_scaled)

# Scree plot and cumulative variance
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.bar(range(1, len(pca_full.explained_variance_ratio_) + 1),
        pca_full.explained_variance_ratio_)
ax1.set_xlabel('Component'); ax1.set_ylabel('Variance Explained')
ax1.set_title('Scree Plot')

ax2.plot(range(1, len(pca_full.explained_variance_ratio_) + 1),
         np.cumsum(pca_full.explained_variance_ratio_), 'bo-')
ax2.axhline(y=0.9, color='r', linestyle='--', label='90% threshold')
ax2.set_xlabel('Components'); ax2.set_ylabel('Cumulative Variance')
ax2.set_title('Cumulative Explained Variance')
ax2.legend()

# Reduced PCA
n_components = 10  # or however many you need
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)
print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.3f}")

# Examine loadings
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(n_components)],
    index=feature_names
)
print(loadings.round(3))

# --- t-SNE ---
# Always try multiple perplexity values
for perp in [5, 15, 30, 50]:
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42,
                n_iter=1000, learning_rate='auto', init='pca')
    X_tsne = tsne.fit_transform(X_scaled)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='tab10', s=5)
    plt.title(f't-SNE (perplexity={perp})')

# --- UMAP ---
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2,
                     random_state=42, metric='euclidean')
X_umap = reducer.fit_transform(X_scaled)
plt.figure(figsize=(8, 6))
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=labels, cmap='tab10', s=5)
plt.title('UMAP')

# UMAP can transform new data (t-SNE cannot)
X_new_umap = reducer.transform(X_new_scaled)
```

## R implementation (if applicable)

```r
# PCA
pca_result <- prcomp(X, scale. = TRUE, center = TRUE)
summary(pca_result)
screeplot(pca_result, type = "lines")
biplot(pca_result)

# t-SNE
library(Rtsne)
tsne_result <- Rtsne(X_scaled, perplexity = 30, dims = 2, verbose = TRUE)
plot(tsne_result$Y, col = labels, pch = 16)

# UMAP
library(umap)
umap_result <- umap(X_scaled, n_neighbors = 15, min_dist = 0.1)
plot(umap_result$layout, col = labels, pch = 16)
```

## Gotchas & lessons learned

1. **Always standardize before PCA.** PCA maximizes variance. If one feature has variance 10,000 and another has variance 1, the first feature dominates PC1 regardless of importance. Standardization puts all features on equal footing.

2. **t-SNE distances between clusters are meaningless.** Two clusters that appear far apart in a t-SNE plot are not necessarily more different than two clusters that appear close. Only within-cluster structure is meaningful. UMAP is somewhat better at preserving inter-cluster distances.

3. **t-SNE is for visualization only.** Do not use t-SNE components as features for downstream models. The embedding is not stable (different runs give different results) and does not generalize to new data.

4. **Always try multiple perplexity values for t-SNE.** A single t-SNE plot with one perplexity can be misleading. Create a grid of plots with perplexity = 5, 15, 30, 50 and look for consistent patterns.

5. **PCA components are not always interpretable.** Sometimes the principal components correspond to meaningful concepts (e.g., PC1 = "size," PC2 = "shape"). Often they do not -- they are just the directions of maximum variance. Do not force interpretations.

6. **The curse of dimensionality is real.** In 100 dimensions, nearly all points are approximately equidistant from each other. Nearest-neighbor methods (KNN, DBSCAN) become unreliable. Dimensionality reduction before clustering is often essential.

7. **PCA assumes linear relationships.** If the structure in your data is inherently non-linear (e.g., a Swiss roll, concentric circles), PCA will miss it. Use kernel PCA, t-SNE, or UMAP instead.

8. **UMAP is sensitive to its hyperparameters.** `n_neighbors` and `min_dist` significantly affect the output. Explore different settings before drawing conclusions.

9. **Do not confuse dimensionality reduction with feature selection.** PCA creates new features (linear combinations of all original features). Feature selection (Lasso, mutual information) keeps a subset of original features. Both reduce dimensionality but in fundamentally different ways.

10. **For large datasets, use incremental PCA.** Scikit-learn's `IncrementalPCA` processes data in batches and is memory-efficient.

## Resources

- **scikit-learn Decomposition documentation**: [PCA and related methods](https://scikit-learn.org/stable/modules/decomposition.html)
- **UMAP documentation**: [umap-learn.readthedocs.io](https://umap-learn.readthedocs.io/) -- Includes excellent conceptual explanations.
- **Distill.pub, "How to Use t-SNE Effectively"** (Wattenberg et al., 2016) -- Essential reading on interpreting t-SNE. [distill.pub/2016/misread-tsne](https://distill.pub/2016/misread-tsne/)
- **An Introduction to Statistical Learning (ISLR)**, Chapter 12 -- PCA coverage.
- **3Blue1Brown, "Eigenvectors and Eigenvalues"**: YouTube video that builds geometric intuition for the math behind PCA.
- **McInnes et al., "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction" (2018)**: The original UMAP paper. Surprisingly readable.
- **StatQuest**: Visual explanations of PCA and t-SNE.
