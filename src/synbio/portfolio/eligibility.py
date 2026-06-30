from __future__ import annotations

from pathlib import Path

import pandas as pd

from synbio.io.constraints import validate_sequence

__all__ = ["load_exclusion_set", "annotate_eligibility"]


def load_exclusion_set(path: str | Path) -> set[str]:
    """Read the `Sequence` column of Exclusion_List.csv into a membership set."""
    col = pd.read_csv(path, usecols=["Sequence"])["Sequence"]
    return set(col.dropna().astype(str).str.strip())


def annotate_eligibility(
    df: pd.DataFrame,
    exclusion: set[str],
    min_brightness: float,
) -> pd.DataFrame:
    """Add `eligible` and `ineligible_reason` columns (does not drop rows)."""
    out = df.copy()
    eligible: list[bool] = []
    reasons: list[str] = []
    for _, row in out.iterrows():
        rs: list[str] = []
        if float(row["b_hat"]) < float(min_brightness):
            rs.append(f"brightness {float(row['b_hat']):.3f} < {min_brightness}")
        ok, violations = validate_sequence(str(row["sequence"]), exclusion, require_nterm=False)
        rs.extend(violations)
        eligible.append(not rs)
        reasons.append("; ".join(rs))
    out["eligible"] = eligible
    out["ineligible_reason"] = reasons
    return out
