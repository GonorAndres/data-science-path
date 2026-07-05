"""PSI calculation on synthetic data."""

import numpy as np


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)

    eps = 1e-6
    return float(np.sum((cur_pct - ref_pct) * np.log((cur_pct + eps) / (ref_pct + eps))))


def main() -> None:
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, size=10_000)

    no_shift = rng.normal(0, 1, size=10_000)
    small_shift = rng.normal(0.1, 1, size=10_000)
    big_shift = rng.normal(1.0, 1.5, size=10_000)

    print(f"no shift PSI: {psi(reference, no_shift):.4f}")
    print(f"small shift PSI: {psi(reference, small_shift):.4f}")
    print(f"big shift PSI: {psi(reference, big_shift):.4f}")


if __name__ == "__main__":
    main()
