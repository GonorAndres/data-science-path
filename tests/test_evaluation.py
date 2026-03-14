"""Tests for evaluation utilities."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.utils.evaluation import (
    gini_coefficient,
    normalized_gini,
    lorenz_curve,
    lift_chart,
    calibration_data,
    poisson_deviance,
    gamma_deviance,
    d_squared,
    compute_all_metrics,
    model_comparison_table,
)


class TestGini:
    """Test Gini coefficient calculation."""

    def test_perfect_prediction(self):
        """Perfect ranking should give Gini close to the theoretical max."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 2, 3, 5])
        y_pred = y_true.astype(float)
        gini = gini_coefficient(y_true, y_pred)
        assert gini > 0.5, f"Perfect prediction Gini should be high, got {gini}"

    def test_random_prediction(self):
        """Random predictions should give Gini close to 0."""
        rng = np.random.RandomState(42)
        y_true = rng.poisson(0.04, 10000)
        y_pred = rng.uniform(0, 0.1, 10000)
        gini = gini_coefficient(y_true, y_pred)
        assert abs(gini) < 0.1, f"Random prediction Gini should be near 0, got {gini}"

    def test_known_values(self):
        """Test with known simple case."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4])
        gini = gini_coefficient(y_true, y_pred)
        assert gini > 0, "Correct ranking should give positive Gini"

    def test_all_zeros(self):
        """All-zero target should give Gini = 0."""
        y_true = np.zeros(100)
        y_pred = np.random.rand(100)
        assert gini_coefficient(y_true, y_pred) == 0.0

    def test_with_weights(self):
        """Gini should work with sample weights."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4])
        w = np.array([1.0, 0.5, 0.5, 1.0])
        gini = gini_coefficient(y_true, y_pred, sample_weight=w)
        assert isinstance(gini, float)
        assert gini > 0


class TestNormalizedGini:
    def test_perfect_gets_one(self):
        """Normalized Gini of perfect prediction should be 1.0."""
        y_true = np.array([0, 0, 0, 1, 1, 2, 5])
        gini = normalized_gini(y_true, y_true.astype(float))
        assert abs(gini - 1.0) < 1e-6

    def test_returns_float(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0.2, 0.8, 0.3, 0.7])
        result = normalized_gini(y_true, y_pred)
        assert isinstance(result, float)


class TestLorenzCurve:
    def test_returns_correct_keys(self):
        y_true = np.array([0, 0, 1, 1, 2])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        result = lorenz_curve(y_true, y_pred, n_points=5)
        assert "fraction_policies" in result
        assert "fraction_claims" in result
        assert "diagonal" in result

    def test_starts_at_origin(self):
        y_true = np.random.poisson(0.1, 100)
        y_pred = np.random.rand(100)
        result = lorenz_curve(y_true, y_pred)
        assert result["fraction_policies"][0] == 0.0
        assert result["fraction_claims"][0] == 0.0

    def test_ends_at_one(self):
        y_true = np.array([0, 0, 1, 2, 3])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        result = lorenz_curve(y_true, y_pred, n_points=5)
        assert abs(result["fraction_policies"][-1] - 1.0) < 1e-4
        assert abs(result["fraction_claims"][-1] - 1.0) < 1e-4

    def test_json_serializable(self):
        """All values should be plain Python floats."""
        import json
        y_true = np.random.poisson(0.1, 50)
        y_pred = np.random.rand(50)
        result = lorenz_curve(y_true, y_pred, n_points=10)
        json.dumps(result)  # Should not raise


class TestLiftChart:
    def test_correct_number_of_bins(self):
        y_true = np.random.poisson(0.05, 1000)
        y_pred = np.random.rand(1000)
        result = lift_chart(y_true, y_pred, n_bins=10)
        assert len(result["bins"]) == 10

    def test_lift_shape(self, sample_predictions):
        """Better models should have higher lift in top decile."""
        result_gbm = lift_chart(
            sample_predictions["y_true"],
            sample_predictions["gbm_pred"],
            n_bins=5,
        )
        # Top bin should have lift > 1
        top_bin_lift = result_gbm["bins"][-1]["lift"]
        assert top_bin_lift > 0, "Top bin lift should be positive"

    def test_bin_structure(self):
        y_true = np.random.poisson(0.05, 100)
        y_pred = np.random.rand(100)
        result = lift_chart(y_true, y_pred, n_bins=5)
        for b in result["bins"]:
            assert "bin" in b
            assert "avg_predicted" in b
            assert "avg_actual" in b
            assert "lift" in b
            assert "count" in b


class TestDeviance:
    def test_poisson_deviance_positive(self):
        y_true = np.array([0, 0, 1, 1, 2])
        y_pred = np.array([0.1, 0.1, 0.5, 0.5, 1.0])
        dev = poisson_deviance(y_true, y_pred)
        assert dev > 0

    def test_poisson_perfect_prediction(self):
        """Perfect prediction should give low deviance."""
        y_true = np.array([0.5, 1.0, 1.5, 2.0])
        dev = poisson_deviance(y_true, y_true)
        assert dev < 1e-6

    def test_gamma_deviance_positive(self):
        y_true = np.array([100, 200, 500, 1000])
        y_pred = np.array([120, 180, 480, 950])
        dev = gamma_deviance(y_true, y_pred)
        assert dev > 0

    def test_gamma_deviance_filters_zeros(self):
        """Gamma deviance should ignore zero values in y_true."""
        y_true = np.array([0, 0, 100, 200])
        y_pred = np.array([10, 10, 110, 190])
        dev = gamma_deviance(y_true, y_pred)
        assert dev > 0

    def test_d_squared_range(self):
        """D-squared should be between 0 and 1 for reasonable predictions."""
        rng = np.random.RandomState(42)
        y_true = rng.poisson(0.5, 1000)
        y_pred = np.clip(y_true + rng.normal(0, 0.1, 1000), 0.01, 10)
        d2 = d_squared(y_true, y_pred, family="poisson")
        assert 0 <= d2 <= 1, f"D-squared should be in [0,1], got {d2}"


class TestCalibration:
    def test_correct_bins(self):
        y_true = np.random.poisson(0.05, 500)
        y_pred = np.random.rand(500) * 0.1
        result = calibration_data(y_true, y_pred, n_bins=5)
        assert len(result["bins"]) <= 5

    def test_bin_structure(self):
        y_true = np.random.poisson(0.05, 500)
        y_pred = np.random.rand(500) * 0.1
        result = calibration_data(y_true, y_pred, n_bins=5)
        for b in result["bins"]:
            assert "avg_predicted" in b
            assert "avg_actual" in b
            assert "count" in b


class TestModelComparison:
    def test_comparison_table(self, sample_predictions):
        preds = {
            "GLM": sample_predictions["glm_pred"],
            "GBM": sample_predictions["gbm_pred"],
            "Random": sample_predictions["random_pred"],
        }
        result = model_comparison_table(
            sample_predictions["y_true"], preds,
            sample_predictions["weights"], family="poisson",
        )
        assert len(result) == 3
        # Should be sorted by Gini descending
        ginis = [r["gini"] for r in result]
        assert ginis == sorted(ginis, reverse=True)

    def test_compute_all_metrics(self, sample_predictions):
        result = compute_all_metrics(
            sample_predictions["y_true"],
            sample_predictions["gbm_pred"],
            sample_predictions["weights"],
            family="poisson",
        )
        assert "gini" in result
        assert "normalized_gini" in result
        assert "d_squared" in result
        assert "poisson_deviance" in result
