"""Tests for data loader utilities."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.utils.data_loader import (
    EXPECTED_FREQ_COLS,
    EXPECTED_SEV_COLS,
    NUMERIC_COLS,
    CATEGORICAL_COLS,
    _clean_freq,
    _clean_sev,
)


@pytest.fixture
def raw_freq_df():
    """Create a raw frequency DataFrame mimicking OpenML output."""
    rng = np.random.RandomState(42)
    n = 200
    return pd.DataFrame({
        "IDpol": [str(i) for i in range(1, n + 1)],  # string from OpenML
        "ClaimNb": [str(x) for x in rng.poisson(0.04, n)],  # string
        "Exposure": rng.uniform(0.01, 1.5, n),  # some > 1
        "VehPower": rng.randint(4, 15, n),
        "VehAge": rng.randint(0, 25, n),
        "DrivAge": rng.randint(18, 90, n),
        "BonusMalus": rng.randint(50, 200, n),
        "VehBrand": rng.choice(["B1", "B2", "B3"], n),
        "VehGas": rng.choice(["Diesel", "Regular"], n),
        "Area": rng.choice(["A", "B", "C", "D", "E", "F"], n),
        "Density": rng.lognormal(5, 2, n),
        "Region": rng.choice(["R11", "R24", "R53"], n),
    })


@pytest.fixture
def raw_sev_df():
    """Create a raw severity DataFrame mimicking OpenML output."""
    rng = np.random.RandomState(42)
    n = 50
    return pd.DataFrame({
        "IDpol": [str(rng.randint(1, 100)) for _ in range(n)],
        "ClaimAmount": rng.lognormal(7, 1.5, n),
    })


class TestCleanFreq:
    def test_idpol_is_int(self, raw_freq_df):
        result = _clean_freq(raw_freq_df)
        assert result["IDpol"].dtype == int or result["IDpol"].dtype == np.int64

    def test_claimnb_capped_at_4(self, raw_freq_df):
        # Inject a high claim count
        raw_freq_df.loc[0, "ClaimNb"] = "10"
        result = _clean_freq(raw_freq_df)
        assert result["ClaimNb"].max() <= 4

    def test_exposure_clipped(self, raw_freq_df):
        raw_freq_df.loc[0, "Exposure"] = 2.0  # over 1
        raw_freq_df.loc[1, "Exposure"] = -0.5  # negative
        result = _clean_freq(raw_freq_df)
        assert result["Exposure"].max() <= 1.0
        assert result["Exposure"].min() > 0

    def test_categorical_dtypes(self, raw_freq_df):
        result = _clean_freq(raw_freq_df)
        for col in CATEGORICAL_COLS:
            assert result[col].dtype.name == "category"

    def test_numeric_dtypes(self, raw_freq_df):
        result = _clean_freq(raw_freq_df)
        for col in NUMERIC_COLS:
            assert np.issubdtype(result[col].dtype, np.number)


class TestCleanSev:
    def test_idpol_is_int(self, raw_sev_df):
        result = _clean_sev(raw_sev_df)
        assert result["IDpol"].dtype == int or result["IDpol"].dtype == np.int64

    def test_positive_claims_only(self, raw_sev_df):
        raw_sev_df.loc[0, "ClaimAmount"] = 0
        raw_sev_df.loc[1, "ClaimAmount"] = -100
        result = _clean_sev(raw_sev_df)
        assert (result["ClaimAmount"] > 0).all()

    def test_claim_amount_numeric(self, raw_sev_df):
        result = _clean_sev(raw_sev_df)
        assert np.issubdtype(result["ClaimAmount"].dtype, np.number)


class TestSchema:
    def test_expected_freq_columns(self):
        """Verify expected column names are defined."""
        assert "IDpol" in EXPECTED_FREQ_COLS
        assert "ClaimNb" in EXPECTED_FREQ_COLS
        assert "Exposure" in EXPECTED_FREQ_COLS
        assert len(EXPECTED_FREQ_COLS) == 12

    def test_expected_sev_columns(self):
        assert "IDpol" in EXPECTED_SEV_COLS
        assert "ClaimAmount" in EXPECTED_SEV_COLS
        assert len(EXPECTED_SEV_COLS) == 2
