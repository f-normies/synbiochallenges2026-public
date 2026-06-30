"""Pocket token-index resolution for chromophore-aware pooling (pure python; .venv-testable).

Token index = canonical avGFP/sfGFP residue position (Met=1): the ESMC tokenizer wraps the sequence
as `<cls> res_1 … res_L <eos>` (verified in docker/repos/esm/.../sequence_tokenizer.py), so residue p
sits at token p. avGFP and amacGFP are colinear at the pocket (both 238 aa, no indels — verified
against data/gfps_wt.fasta), so one identity map serves both. Residue identity is asserted at the
invariant anchors to catch any tokenizer off-by-one or a wrong reference.
"""

import logging

__all__ = ["CANON_POCKET", "LOCAL_POOLS", "ANCHORS", "PocketResolutionError", "resolve_indices"]

logger = logging.getLogger(__name__)

# Canonical avGFP/sfGFP numbering (Met=1); from PROJECT_PLAN §4.2 / do_not_mutate.
CANON_POCKET: dict[str, list[int]] = {
    "chromo": [65, 66, 67],
    "pocket": [65, 66, 67, 94, 96, 148, 203, 205, 222],
    "aromatic": [27, 46, 64, 66, 92, 106, 145, 148, 151, 200, 223],
}
LOCAL_POOLS: tuple[str, ...] = tuple(CANON_POCKET)
# Invariant across avGFP/amacGFP (subset of do_not_mutate): residue identity must hold here.
ANCHORS: dict[int, str] = {66: "Y", 67: "G", 96: "R", 148: "H", 222: "E"}


class PocketResolutionError(ValueError):
    """Raised when a reference fails an anchor identity check (off-by-one or wrong reference)."""


def resolve_indices(ref_seq: str) -> dict[str, list[int]]:
    """Return {pool: [token indices]} for the local pools; assert the invariant anchors.

    Raises PocketResolutionError if any anchor residue does not match (token = canonical position).
    """
    for pos, aa in ANCHORS.items():
        got = ref_seq[pos - 1] if 0 < pos <= len(ref_seq) else "?"
        if got != aa:
            raise PocketResolutionError(
                f"anchor {aa}{pos} != {got!r} — tokenizer off-by-one or wrong reference"
            )
    return {pool: list(positions) for pool, positions in CANON_POCKET.items()}
