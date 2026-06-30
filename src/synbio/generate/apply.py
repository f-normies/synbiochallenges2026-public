"""Shared point-mutation primitives for the generation stage.

A Mutation is 1-based and sfGFP-relative (wt = the sfGFP residue at pos), so its
token (e.g. "V11I") and apply step are self-checking. Used by all three generation
branches; pure functions, no heavy deps.
"""

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from synbio.io.constraints import STANDARD_AA, validate_sequence

__all__ = [
    "Mutation",
    "parse_mutation",
    "apply_mutations",
    "hamming",
    "gate_single",
    "build_and_validate",
]

_TOKEN = re.compile(r"^([A-Z])(\d+)([A-Z])$")

# Free surface cysteines are excluded everywhere we generate: cell-free expression is
# a reducing environment, so engineered/free Cys mis-pair and aggregate — and 72 °C
# readout is dominated by irreversible aggregation (PROJECT_PLAN §4.3, §4.5).
FORBIDDEN_TARGETS = frozenset("C")


@dataclass(frozen=True)
class Mutation:
    """A 1-based point substitution; wt is the sfGFP residue at pos."""

    pos: int
    wt: str
    target: str

    @property
    def token(self) -> str:
        return f"{self.wt}{self.pos}{self.target}"


def parse_mutation(token: str) -> Mutation:
    """Parse a token like 'V11I' into a Mutation."""
    m = _TOKEN.match(token)
    if not m:
        raise ValueError(f"bad mutation token: {token!r}")
    return Mutation(pos=int(m.group(2)), wt=m.group(1), target=m.group(3))


def apply_mutations(wt_seq: str, muts: Iterable[Mutation]) -> str:
    """Apply 1-based point mutations to wt_seq.

    Raises:
        ValueError: two mutations share a position, or a mutation's wt residue
            disagrees with wt_seq at that position.
    """
    chars = list(wt_seq)
    seen: set[int] = set()
    for mut in muts:
        if mut.pos in seen:
            raise ValueError(f"duplicate mutation position: {mut.pos}")
        if not (1 <= mut.pos <= len(chars)):
            raise ValueError(f"position {mut.pos} out of range 1..{len(chars)}")
        if chars[mut.pos - 1] != mut.wt:
            raise ValueError(
                f"wt mismatch at {mut.pos}: seq has {chars[mut.pos - 1]}, mutation says {mut.wt}"
            )
        chars[mut.pos - 1] = mut.target
        seen.add(mut.pos)
    return "".join(chars)


def hamming(a: str, b: str) -> int:
    """Number of differing positions; sequences must be equal length."""
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} != {len(b)}")
    return sum(1 for x, y in zip(a, b) if x != y)


def gate_single(
    pos: int,
    target: str,
    wt_seq: str,
    free_mutate: set[int],
    do_not_mutate: set[int],
    min_pos: int = 11,
) -> bool:
    """True iff a single sfGFP[pos]->target substitution is allowed."""
    return (
        pos >= min_pos
        and pos in free_mutate
        and pos not in do_not_mutate
        and target in STANDARD_AA
        and target not in FORBIDDEN_TARGETS
        and target != wt_seq[pos - 1]
    )


def build_and_validate(
    wt_seq: str,
    muts: Sequence[Mutation],
    exclusion: set[str],
    require_nterm: bool,
) -> tuple[str, bool, list[str]]:
    """Apply mutations then run the hard sequence constraints.

    Returns (sequence, is_valid, violations).
    """
    seq = apply_mutations(wt_seq, muts)
    ok, why = validate_sequence(seq, exclusion, require_nterm=require_nterm)
    return seq, ok, why
