"""Schema + IO for the carrier table `candidates.parquet`.

Base columns are guaranteed across the whole funnel; stages append score/rank
columns. Validation gives an early, explicit failure on schema drift.
"""

from pathlib import Path

import pandas as pd

__all__ = [
    "BASE_COLUMNS",
    "SchemaError",
    "empty_candidates",
    "validate_candidates",
    "read_candidates",
    "write_candidates",
]

# id: unique candidate id; sequence: AA string; source: generator@temp / slot type;
# parent: parent sequence id for sfGFP-derived designs (nullable).
BASE_COLUMNS: tuple[str, ...] = ("id", "sequence", "source", "parent")


class SchemaError(ValueError):
    """Raised when a candidate table violates the carrier schema."""


def empty_candidates() -> pd.DataFrame:
    """Return an empty candidate table with exactly the base columns."""
    return pd.DataFrame({c: pd.Series(dtype="object") for c in BASE_COLUMNS})


def validate_candidates(
    df: pd.DataFrame,
    required: tuple[str, ...] = BASE_COLUMNS,
) -> None:
    """Validate that `df` carries every required column with unique ids.

    Raises:
        SchemaError: missing required columns or duplicate ids.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(f"candidate table missing required columns: {missing}")
    if "id" in df.columns and df["id"].duplicated().any():
        dups = df.loc[df["id"].duplicated(), "id"].tolist()
        raise SchemaError(f"duplicate candidate ids: {dups}")


def read_candidates(path: str | Path) -> pd.DataFrame:
    """Read a candidate table from parquet and validate the base schema."""
    df = pd.read_parquet(path)
    validate_candidates(df)
    return df


def write_candidates(df: pd.DataFrame, path: str | Path) -> None:
    """Validate then write a candidate table to parquet."""
    validate_candidates(df)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
