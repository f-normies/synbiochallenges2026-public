"""Shared builder for candidates.parquet rows with generation provenance.

Base columns (id/sequence/source/parent) satisfy the carrier schema; provenance
columns carry per-design mutation accounting for the manifest / downstream stages.
"""

from collections.abc import Sequence

import pandas as pd

from synbio.io.artifacts import validate_candidates

from .apply import Mutation, hamming
from .recharge import net_charge_delta

__all__ = ["PROVENANCE_COLUMNS", "design_row", "rows_to_frame"]

PROVENANCE_COLUMNS: tuple[str, ...] = (
    "n_mut", "hamming", "mutations", "k_bright",
    "k_recharge", "k_consensus", "net_charge_delta", "bucket",
)


def design_row(
    cid: str,
    sequence: str,
    wt_seq: str,
    muts: Sequence[Mutation],
    source: str,
    parent: str,
    pool_counts: dict[str, int],
    bucket: str,
) -> dict:
    """Build one candidate row (base + provenance columns)."""
    ordered = sorted(muts, key=lambda m: m.pos)
    return {
        "id": cid,
        "sequence": sequence,
        "source": source,
        "parent": parent,
        "n_mut": len(ordered),
        "hamming": hamming(wt_seq, sequence),
        "mutations": ":".join(m.token for m in ordered),
        "k_bright": int(pool_counts.get("bright", 0)),
        "k_recharge": int(pool_counts.get("recharge", 0)),
        "k_consensus": int(pool_counts.get("consensus", 0)),
        "net_charge_delta": sum(net_charge_delta(m) for m in ordered),
        "bucket": bucket,
    }


def rows_to_frame(rows: list[dict]) -> pd.DataFrame:
    """Build a candidate DataFrame from rows and validate the carrier schema."""
    df = pd.DataFrame(rows)
    validate_candidates(df)
    return df
