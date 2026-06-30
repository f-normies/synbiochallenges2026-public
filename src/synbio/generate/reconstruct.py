"""Overlay LigandMPNN modeled-only designs onto the full WT sequence."""

from .apply import Mutation

__all__ = ["reconstruct_full", "mutations_from_diff"]


def reconstruct_full(designed_seq: str, modeled: list[int], wt_seq: str) -> str:
    """Place each designed residue at its WT position; keep WT for unmodeled positions.

    `modeled` is sorted ascending and aligns 1:1 with `designed_seq` (LigandMPNN emits
    residues in ascending resseq order). Positions absent from `modeled` (Met1, the
    65-67 chromophore, the C-terminal tail) keep their WT residue.
    """
    order = sorted(modeled)
    if len(designed_seq) != len(order):
        raise ValueError(
            f"designed_seq length {len(designed_seq)} != modeled count {len(order)}"
        )
    full = list(wt_seq)
    for aa, resid in zip(designed_seq, order):
        full[resid - 1] = aa
    return "".join(full)


def mutations_from_diff(wt_seq: str, seq: str) -> list[Mutation]:
    """Derive 1-based sfGFP-relative Mutations at every position where seq differs from WT."""
    if len(wt_seq) != len(seq):
        raise ValueError(f"length mismatch: {len(wt_seq)} != {len(seq)}")
    return [
        Mutation(pos=i + 1, wt=w, target=m)
        for i, (w, m) in enumerate(zip(wt_seq, seq))
        if w != m
    ]
