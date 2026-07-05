"""FAISS quickstart: build an index, query nearest neighbors."""

import faiss
import numpy as np


def main() -> None:
    d = 64
    n = 10_000
    rng = np.random.default_rng(0)
    xb = rng.standard_normal((n, d)).astype("float32")

    index = faiss.IndexFlatL2(d)
    index.add(xb)
    print(f"indexed {index.ntotal} vectors of dim {d}")

    xq = rng.standard_normal((5, d)).astype("float32")
    distances, ids = index.search(xq, k=3)
    for i, (d_row, id_row) in enumerate(zip(distances, ids)):
        print(f"query {i}: top-3 ids {id_row.tolist()}  distances {d_row.tolist()}")


if __name__ == "__main__":
    main()
