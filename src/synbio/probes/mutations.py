"""Parse Sarkisyan mutation strings and apply them to a background sequence.

Dataset numbering is offset +1: a mutation `XnnnY` edits 0-based index `nnn`
(the leading Met is uncounted). Verified 100% on avGFP/amacGFP/cgreGFP/ppluGFP.
"""

import re

__all__ = ["MutationError", "parse_mutations", "apply_mutations"]

_MUT_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


class MutationError(ValueError):
    """Raised when a mutation token is malformed, out of bounds, or WT-mismatched."""


def parse_mutations(aa_mutations: str) -> tuple[str, ...]:
    """Split a ':'-separated mutation string; 'WT' (or empty) -> ()."""
    if not isinstance(aa_mutations, str) or aa_mutations == "WT" or aa_mutations == "":
        return ()
    return tuple(aa_mutations.split(":"))


def apply_mutations(background: str, mutations: tuple[str, ...], offset: int = 1) -> str:
    """Apply mutations onto `background`; index = pos - 1 + offset.

    Raises MutationError on a malformed token, an out-of-bounds index, or a
    WT-residue mismatch (the background residue must equal the token's WT letter).
    """
    seq = list(background)
    for token in mutations:
        m = _MUT_RE.match(token)
        if m is None:
            raise MutationError(f"malformed mutation token: {token!r}")
        wt, pos, mut = m.group(1), int(m.group(2)), m.group(3)
        idx = pos - 1 + offset
        if not 0 <= idx < len(seq):
            raise MutationError(f"{token}: index {idx} out of bounds (len {len(seq)})")
        if seq[idx] != wt:
            raise MutationError(f"{token}: background has {seq[idx]!r} at index {idx}, not {wt!r}")
        seq[idx] = mut
    return "".join(seq)
