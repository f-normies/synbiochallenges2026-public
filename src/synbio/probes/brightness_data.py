"""Build the brightness training table from the Sarkisyan multi-background DMS.

Keeps `backgrounds` only, applies each variant's mutations onto its matching reference
(offset +1), drops WT-mismatched/out-of-frame variants, and per-type WT-centers the target.
"""

import logging
from pathlib import Path

import pandas as pd

from synbio.probes.mutations import MutationError, apply_mutations, parse_mutations

__all__ = ["build_brightness_dataset", "read_multi_fasta"]

logger = logging.getLogger(__name__)


def read_multi_fasta(path: str | Path) -> dict[str, str]:
    """Parse a multi-record FASTA; record name = first whitespace/'|' token after '>'."""
    refs: dict[str, str] = {}
    name: str | None = None
    buf: list[str] = []
    for line in Path(path).read_text().splitlines():
        line = line.rstrip()
        if line.startswith(">"):
            if name is not None:
                refs[name] = "".join(buf)
            name = line[1:].split("|")[0].strip().split()[0]
            buf = []
        elif line:
            buf.append(line)
    if name is not None:
        refs[name] = "".join(buf)
    return refs


def build_brightness_dataset(
    df: pd.DataFrame,
    refs: dict[str, str],
    backgrounds: list[str],
    offset: int = 1,
) -> tuple[pd.DataFrame, int, dict[str, float]]:
    """Return (table, n_dropped, wt_brightness_by_type).

    Table columns: gfp_type, sequence, nmut, mutset (frozenset), brightness, y (WT-centered).
    """
    df = df.rename(columns={"aaMutations": "mutations", "GFP type": "gfp_type", "Brightness": "brightness"})
    df = df[df["gfp_type"].isin(backgrounds)]

    wt_bright: dict[str, float] = {}
    for t in backgrounds:
        wt_rows = df[(df["gfp_type"] == t) & (df["mutations"] == "WT")]["brightness"]
        if len(wt_rows) == 0:
            raise ValueError(f"no WT row for background {t}")
        wt_bright[t] = float(wt_rows.iloc[0])

    rows: list[dict] = []
    dropped = 0
    for r in df.itertuples(index=False):
        muts = parse_mutations(r.mutations)
        try:
            seq = apply_mutations(refs[r.gfp_type], muts, offset)
        except MutationError:
            dropped += 1
            continue
        rows.append(
            {
                "gfp_type": r.gfp_type,
                "sequence": seq,
                "nmut": len(muts),
                "mutset": frozenset(muts),
                "brightness": float(r.brightness),
                "y": float(r.brightness) - wt_bright[r.gfp_type],
            }
        )
    if dropped:
        logger.warning("dropped %d brightness variants (WT-mismatch/out-of-frame)", dropped)
    return pd.DataFrame(rows), dropped, wt_bright
