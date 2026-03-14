"""
Load freMTPL2 dataset from OpenML, merge frequency and severity,
cache as parquet, and provide train/val/test splits.
"""

from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
FREQ_CACHE = CACHE_DIR / "freMTPL2_freq.parquet"
SEV_CACHE = CACHE_DIR / "freMTPL2_sev.parquet"
MERGED_CACHE = CACHE_DIR / "freMTPL2_merged.parquet"

FREQ_OPENML_ID = 41214
SEV_OPENML_ID = 41215

EXPECTED_FREQ_COLS = [
    "IDpol", "ClaimNb", "Exposure", "VehPower", "VehAge", "DrivAge",
    "BonusMalus", "VehBrand", "VehGas", "Area", "Density", "Region",
]
EXPECTED_SEV_COLS = ["IDpol", "ClaimAmount"]

NUMERIC_COLS = ["ClaimNb", "Exposure", "VehPower", "VehAge", "DrivAge",
                "BonusMalus", "Density"]
CATEGORICAL_COLS = ["VehBrand", "VehGas", "Area", "Region"]


def _fetch_and_cache(openml_id: int, cache_path: Path, name: str) -> pd.DataFrame:
    """Fetch dataset from OpenML and cache locally as parquet."""
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    print(f"Downloading {name} from OpenML (id={openml_id})...")
    data = fetch_openml(data_id=openml_id, as_frame=True, parser="auto")
    df = data.frame  # type: ignore[union-attr]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f"Cached {name} to {cache_path} ({len(df):,} rows)")
    return df


def load_freq() -> pd.DataFrame:
    """Load frequency table (677,991 policies)."""
    df = _fetch_and_cache(FREQ_OPENML_ID, FREQ_CACHE, "freMTPL2freq")
    return _clean_freq(df)


def load_sev() -> pd.DataFrame:
    """Load severity table (26,444 individual claims)."""
    df = _fetch_and_cache(SEV_OPENML_ID, SEV_CACHE, "freMTPL2sev")
    return _clean_sev(df)


def _clean_freq(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure correct dtypes for frequency data."""
    df = df.copy()

    # IDpol to int
    df["IDpol"] = df["IDpol"].astype(int)

    # Numeric columns
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ClaimNb as int (cap at 4 -- standard actuarial practice)
    df["ClaimNb"] = df["ClaimNb"].astype(int).clip(upper=4)

    # Exposure in (0, 1]
    df["Exposure"] = df["Exposure"].clip(lower=1e-9, upper=1.0)

    # Categorical columns
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


def _clean_sev(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure correct dtypes for severity data."""
    df = df.copy()
    df["IDpol"] = df["IDpol"].astype(int)
    df["ClaimAmount"] = pd.to_numeric(df["ClaimAmount"], errors="coerce")
    # Remove zero/negative claims
    df = df[df["ClaimAmount"] > 0].copy()
    return df


def load_merged() -> pd.DataFrame:
    """
    Merge frequency and severity: aggregate claim amounts per policy,
    creating AvgClaimAmount and TotalClaimAmount columns.

    Policies with ClaimNb=0 get AvgClaimAmount=0 and TotalClaimAmount=0.
    """
    if MERGED_CACHE.exists():
        return pd.read_parquet(MERGED_CACHE)

    freq = load_freq()
    sev = load_sev()

    # Aggregate severity per policy
    sev_agg = sev.groupby("IDpol").agg(
        TotalClaimAmount=("ClaimAmount", "sum"),
        AvgClaimAmount=("ClaimAmount", "mean"),
        ClaimCount_sev=("ClaimAmount", "count"),
    ).reset_index()

    # Merge
    merged = freq.merge(sev_agg, on="IDpol", how="left")

    # Fill NaN for policies with no claims
    merged["TotalClaimAmount"] = merged["TotalClaimAmount"].fillna(0.0)
    merged["AvgClaimAmount"] = merged["AvgClaimAmount"].fillna(0.0)
    merged["ClaimCount_sev"] = merged["ClaimCount_sev"].fillna(0).astype(int)

    # HasClaim indicator
    merged["HasClaim"] = (merged["ClaimNb"] > 0).astype(int)

    # Pure premium = TotalClaimAmount / Exposure
    merged["PurePremium"] = merged["TotalClaimAmount"] / merged["Exposure"]

    MERGED_CACHE.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(MERGED_CACHE, index=False)
    print(f"Cached merged dataset to {MERGED_CACHE} ({len(merged):,} rows)")

    return merged


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split into train/val/test using stratified sampling on HasClaim.

    Returns (train, val, test) DataFrames.
    """
    if "HasClaim" not in df.columns:
        df = df.copy()
        df["HasClaim"] = (df["ClaimNb"] > 0).astype(int)

    # First split: train+val vs test
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=random_state,
        stratify=df["HasClaim"],
    )

    # Second split: train vs val
    val_fraction = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_fraction, random_state=random_state,
        stratify=train_val["HasClaim"],
    )

    print(f"Split sizes -- train: {len(train):,}, val: {len(val):,}, test: {len(test):,}")
    return train.copy(), val.copy(), test.copy()


if __name__ == "__main__":
    merged = load_merged()
    print(f"\nDataset shape: {merged.shape}")
    print(f"Columns: {list(merged.columns)}")
    print(f"Claim rate: {merged['HasClaim'].mean():.3%}")
    print(f"\nBasic stats:")
    print(merged[["ClaimNb", "Exposure", "TotalClaimAmount", "PurePremium"]].describe())

    train, val, test = split_data(merged)
    print(f"\nTrain claim rate: {train['HasClaim'].mean():.3%}")
    print(f"Test claim rate: {test['HasClaim'].mean():.3%}")
