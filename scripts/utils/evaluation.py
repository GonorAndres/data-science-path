"""
Model evaluation utilities for insurance pricing.
All functions return JSON-serializable types (lists, dicts, floats) -- never numpy arrays.
"""

import numpy as np
import pandas as pd


def gini_coefficient(y_true, y_pred, sample_weight=None) -> float:
    """
    Calculate the Gini coefficient (2 * AUC - 1 style, for ranking quality).

    Uses the ordered Lorenz approach: sort by predicted, compute concentration.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=float)
    else:
        w = np.ones_like(y_true)

    # Sort by predicted ascending
    order = np.argsort(y_pred)
    y_sorted = y_true[order]
    w_sorted = w[order]

    # Weighted cumulative sums
    cum_w = np.cumsum(w_sorted)
    cum_yw = np.cumsum(y_sorted * w_sorted)

    total_w = cum_w[-1]
    total_yw = cum_yw[-1]

    if total_yw == 0 or total_w == 0:
        return 0.0

    # Gini = 1 - 2 * area under Lorenz curve
    # Area = sum of trapezoids
    lorenz_x = cum_w / total_w
    lorenz_y = cum_yw / total_yw

    # Prepend origin
    lorenz_x = np.concatenate([[0], lorenz_x])
    lorenz_y = np.concatenate([[0], lorenz_y])

    _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    area = _trapz(lorenz_y, lorenz_x)
    gini = 1.0 - 2.0 * area

    return round(float(gini), 6)


def normalized_gini(y_true, y_pred, sample_weight=None) -> float:
    """Normalized Gini = Gini(pred) / Gini(perfect)."""
    gini_pred = gini_coefficient(y_true, y_pred, sample_weight)
    gini_perfect = gini_coefficient(y_true, y_true, sample_weight)
    if gini_perfect == 0:
        return 0.0
    return round(float(gini_pred / gini_perfect), 6)


def lorenz_curve(y_true, y_pred, sample_weight=None, n_points: int = 100) -> dict:
    """
    Compute Lorenz curve data for plotting.

    Returns dict with keys: fraction_policies, fraction_claims, diagonal.
    Each value is a list of floats.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=float)
    else:
        w = np.ones_like(y_true)

    order = np.argsort(y_pred)
    y_sorted = y_true[order]
    w_sorted = w[order]

    cum_w = np.cumsum(w_sorted)
    cum_yw = np.cumsum(y_sorted * w_sorted)

    total_w = cum_w[-1]
    total_yw = cum_yw[-1]

    if total_yw == 0:
        total_yw = 1.0

    frac_policies = cum_w / total_w
    frac_claims = cum_yw / total_yw

    # Subsample to n_points
    indices = np.linspace(0, len(frac_policies) - 1, n_points, dtype=int)

    result_x = [0.0] + [round(float(frac_policies[i]), 6) for i in indices]
    result_y = [0.0] + [round(float(frac_claims[i]), 6) for i in indices]
    diagonal = [round(float(x), 6) for x in np.linspace(0, 1, len(result_x))]

    return {
        "fraction_policies": result_x,
        "fraction_claims": result_y,
        "diagonal": diagonal,
    }


def lift_chart(y_true, y_pred, sample_weight=None, n_bins: int = 10) -> dict:
    """
    Compute lift chart: sort by predicted, bin into deciles, compare actual vs predicted.

    Returns dict with: bins (list of dicts with bin_label, avg_predicted, avg_actual, count, lift).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=float)
    else:
        w = np.ones_like(y_true)

    order = np.argsort(y_pred)
    y_true_sorted = y_true[order]
    y_pred_sorted = y_pred[order]
    w_sorted = w[order]

    # Split into bins
    bin_edges = np.array_split(np.arange(len(y_true)), n_bins)

    bins = []
    overall_avg = float(np.average(y_true, weights=w))

    for i, idx in enumerate(bin_edges):
        yt = y_true_sorted[idx]
        yp = y_pred_sorted[idx]
        wt = w_sorted[idx]

        avg_actual = float(np.average(yt, weights=wt))
        avg_predicted = float(np.average(yp, weights=wt))
        lift = avg_actual / overall_avg if overall_avg > 0 else 0.0

        bins.append({
            "bin": i + 1,
            "bin_label": f"Decile {i + 1}",
            "avg_predicted": round(avg_predicted, 6),
            "avg_actual": round(avg_actual, 6),
            "count": int(len(idx)),
            "lift": round(lift, 4),
        })

    return {"bins": bins, "overall_avg": round(overall_avg, 6)}


def calibration_data(y_true, y_pred, sample_weight=None, n_bins: int = 10) -> dict:
    """
    Compute calibration data: group by predicted quantiles, compare avg predicted vs avg actual.

    Returns dict with: bins (list of dicts).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=float)
    else:
        w = np.ones_like(y_true)

    # Quantile-based bins
    try:
        bin_indices = pd.qcut(y_pred, n_bins, labels=False, duplicates="drop")
    except ValueError:
        bin_indices = pd.cut(y_pred, n_bins, labels=False)

    bins = []
    for b in sorted(set(bin_indices)):
        mask = bin_indices == b
        yt = y_true[mask]
        yp = y_pred[mask]
        wt = w[mask]

        bins.append({
            "bin": int(b + 1),
            "avg_predicted": round(float(np.average(yp, weights=wt)), 6),
            "avg_actual": round(float(np.average(yt, weights=wt)), 6),
            "count": int(mask.sum()),
        })

    return {"bins": bins}


def poisson_deviance(y_true, y_pred, sample_weight=None) -> float:
    """
    Compute mean Poisson deviance.
    D = 2 * mean(y_pred - y_true + y_true * log(y_true / y_pred))
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float).clip(min=1e-10)

    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=float)
    else:
        w = np.ones_like(y_true)

    # For y_true = 0: contribution = 2 * y_pred
    # For y_true > 0: contribution = 2 * (y_pred - y_true + y_true * log(y_true / y_pred))
    dev = np.where(
        y_true == 0,
        2.0 * y_pred,
        2.0 * (y_pred - y_true + y_true * np.log(y_true / y_pred)),
    )

    return round(float(np.average(dev, weights=w)), 6)


def gamma_deviance(y_true, y_pred, sample_weight=None) -> float:
    """
    Compute mean Gamma deviance (for severity, y_true > 0 only).
    D = 2 * mean(-log(y_true / y_pred) + (y_true / y_pred) - 1)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float).clip(min=1e-10)

    # Filter to positive values
    mask = y_true > 0
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=float)[mask]
    else:
        w = np.ones_like(y_true)

    if len(y_true) == 0:
        return 0.0

    ratio = y_true / y_pred
    dev = 2.0 * (-np.log(ratio) + ratio - 1.0)

    return round(float(np.average(dev, weights=w)), 6)


def d_squared(y_true, y_pred, sample_weight=None, family: str = "poisson") -> float:
    """
    D-squared (pseudo R-squared based on deviance).
    D^2 = 1 - D(model) / D(null)
    """
    if family == "poisson":
        dev_fn = poisson_deviance
    elif family == "gamma":
        dev_fn = gamma_deviance
    else:
        raise ValueError(f"Unknown family: {family}")

    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=float)
    else:
        w = np.ones(len(y_true), dtype=float)

    dev_model = dev_fn(y_true, y_pred, w)

    # Null model: weighted mean
    y_true_arr = np.asarray(y_true, dtype=float)
    if family == "gamma":
        mask = y_true_arr > 0
        null_pred = np.full_like(y_true_arr, np.average(y_true_arr[mask], weights=w[mask]))
    else:
        null_pred = np.full_like(y_true_arr, np.average(y_true_arr, weights=w))

    dev_null = dev_fn(y_true, null_pred, w)

    if dev_null == 0:
        return 0.0

    return round(float(1.0 - dev_model / dev_null), 6)


def compute_all_metrics(
    y_true, y_pred, sample_weight=None, family: str = "poisson",
) -> dict:
    """
    Compute all metrics for a single model. Returns a flat dict.
    """
    metrics = {
        "gini": gini_coefficient(y_true, y_pred, sample_weight),
        "normalized_gini": normalized_gini(y_true, y_pred, sample_weight),
        "d_squared": d_squared(y_true, y_pred, sample_weight, family),
    }

    if family == "poisson":
        metrics["poisson_deviance"] = poisson_deviance(y_true, y_pred, sample_weight)
    elif family == "gamma":
        metrics["gamma_deviance"] = gamma_deviance(y_true, y_pred, sample_weight)

    return metrics


def model_comparison_table(
    y_true,
    predictions: dict[str, np.ndarray],
    sample_weight=None,
    family: str = "poisson",
) -> list[dict]:
    """
    Compare multiple models. predictions = {model_name: predicted_values}.

    Returns list of dicts, one per model, sorted by Gini descending.
    """
    results = []
    for name, y_pred in predictions.items():
        metrics = compute_all_metrics(y_true, y_pred, sample_weight, family)
        metrics["model"] = name
        results.append(metrics)

    results.sort(key=lambda x: x["gini"], reverse=True)
    return results
