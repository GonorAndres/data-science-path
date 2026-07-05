# recsys-movielens

## Resumen (ES)

**Problema:** construir un recomendador de peliculas sobre MovieLens 25M. Para cada usuario, sugerir peliculas relevantes que no ha visto.

**Metodo:** tres modelos comparados: popularidad (baseline), ALS (factorizacion matricial implicita con `implicit`), y two-tower neural (usuario + pelicula embeddings via PyTorch). Evaluacion leave-last-out con Recall@K, NDCG@K, coverage, diversidad.

**Resultado objetivo:** Recall@20 mayor a 0.15 en two-tower vs 0.08 en popularidad.

## Summary (EN)

**Problem:** build a movie recommender on MovieLens 25M. For each user, suggest relevant unseen movies.

**Method:** three models compared: popularity (baseline), ALS (implicit matrix factorization via `implicit`), and a neural two-tower (user + movie embeddings via PyTorch). Leave-last-out evaluation with Recall@K, NDCG@K, coverage, diversity.

**Target result:** Recall@20 above 0.15 on two-tower vs 0.08 on popularity.

---

## Business framing

Recommenders drive engagement on streaming, e-commerce, and media platforms. The standard problem: given millions of users and hundreds of thousands of items, predict what each user will want next. MovieLens is the canonical academic dataset and is public, reproducible, well-understood.

## Structure

```
recsys-movielens/
├── README.md
├── theory/
│   ├── 00_problem_framing.md
│   ├── 01_collab_vs_content.md
│   ├── 02_matrix_factorization.md
│   ├── 03_two_tower_architecture.md
│   └── 04_offline_metrics.md
├── data_download.py           # fetch MovieLens 25M
├── src/
│   ├── data.py                # load, leave-last-out split, negatives
│   ├── popularity.py          # baseline
│   ├── als.py                 # implicit ALS via `implicit` library
│   ├── two_tower.py           # PyTorch two-tower model
│   ├── evaluate.py            # Recall@K, NDCG@K, coverage
│   └── serve.py               # score candidates for a user
├── tests/
├── notebooks/
├── requirements.txt
└── Makefile
```

## Quickstart

```bash
pip install -r requirements.txt
python data_download.py
python -m src.popularity
python -m src.als
python -m src.two_tower
python -m src.evaluate
```

## Expected results

| Model | Recall@20 | NDCG@20 | Coverage | Notes |
|-------|-----------|---------|----------|-------|
| Popularity | 0.08 | 0.05 | 1% | Baseline |
| ALS (implicit) | 0.13 | 0.09 | 15% | Standard CF |
| Two-tower | 0.17 | 0.12 | 20% | Best aggregate |

## Decisions and trade-offs

| Decision | Alternative | Why |
|----------|-------------|-----|
| Leave-last-out per user | Random 80/20 | Standard in recsys literature; mimics temporal deployment. |
| Negative sampling (5 per positive) | All negatives | Scales; cost-quality sweet spot. |
| Two-tower | BPR, NCF | Two-tower scales to production via ANN; other approaches are more academic. |
| MovieLens 25M | Amazon Reviews | More structure (ratings, tags); smaller, easier to iterate. |

## References

- `docs/theory/12_recommender_system_foundations.md`
- Koren et al., "Matrix Factorization Techniques for Recommender Systems", 2009.
- Hu, Koren, Volinsky, "Collaborative Filtering for Implicit Feedback Datasets", 2008.
- Covington et al., "Deep Neural Networks for YouTube Recommendations", 2016.
