"""Redesignable-position selection for the LigandMPNN branch (pure)."""

from pathlib import Path

from synbio.wt.structure import read_pdb_residues

__all__ = ["modeled_resids", "redesignable_positions", "residue_tokens"]


def modeled_resids(pdb_path: str | Path, chain: str = "A") -> list[int]:
    """Sorted ATOM resseqs for `chain` (the CRO gap at 65-67 is naturally absent)."""
    return sorted(
        r.resseq
        for r in read_pdb_residues(pdb_path)
        if r.record == "ATOM" and r.chain == chain
    )


def redesignable_positions(
    free_mutate: list[int], modeled: list[int], nterm_freeze: int = 10
) -> list[int]:
    """Tolerant positions LigandMPNN may design: free_mutate ∩ modeled, N-term frozen."""
    allowed = set(free_mutate) & set(modeled)
    return sorted(p for p in allowed if p > nterm_freeze)


def residue_tokens(positions: list[int], chain: str = "A") -> str:
    """LigandMPNN residue selection string, e.g. 'A12 A100'."""
    return " ".join(f"{chain}{p}" for p in positions)
