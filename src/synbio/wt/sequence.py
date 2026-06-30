"""WT FASTA parsing, chromophore-anchor check, and fasta↔pdb consistency."""

from pathlib import Path

from synbio.wt.structure import THREE_TO_ONE, PdbResidue

__all__ = [
    "SequenceError",
    "CRO_SPAN",
    "CHROMOPHORE_TRIAD",
    "read_fasta_sequence",
    "check_anchor",
    "check_fasta_pdb_consistency",
]

CRO_SPAN: tuple[int, int, int] = (65, 66, 67)
CHROMOPHORE_TRIAD: str = "TYG"
_STANDARD_AA: frozenset[str] = frozenset("ACDEFGHIKLMNPQRSTVWY")


class SequenceError(ValueError):
    """Raised when the WT sequence or its consistency with the PDB is invalid."""


def read_fasta_sequence(path: str | Path) -> str:
    """Read a single-record FASTA; validate M-start and the 20-AA alphabet."""
    headers: list[str] = []
    seq_parts: list[str] = []
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            headers.append(line)
            if len(headers) > 1:
                raise SequenceError("expected a single FASTA record")
            continue
        seq_parts.append(line.strip())
    if not headers:
        raise SequenceError("no FASTA header found")
    seq = "".join(seq_parts).upper()
    if not seq.startswith("M"):
        raise SequenceError("sequence must start with M")
    bad = sorted(set(seq) - _STANDARD_AA)
    if bad:
        raise SequenceError(f"non-standard residues present: {''.join(bad)}")
    return seq


def check_anchor(
    seq: str,
    *,
    span: tuple[int, ...] = CRO_SPAN,
    triad: str = CHROMOPHORE_TRIAD,
) -> None:
    """Assert the chromophore triad sits at `span` (1-based start). Raise on mismatch."""
    start = span[0] - 1
    got = seq[start : start + len(triad)]
    if got != triad:
        raise SequenceError(f"chromophore anchor at {span}: expected {triad}, got {got!r}")


def check_fasta_pdb_consistency(
    seq: str,
    residues: list[PdbResidue],
    *,
    cro_span: tuple[int, int, int] = CRO_SPAN,
) -> None:
    """Every modeled ATOM residue must match the FASTA residue at that index."""
    diffs: list[tuple[int, str, str]] = []
    for r in residues:
        if r.record != "ATOM" or r.resseq in cro_span:
            continue
        got = THREE_TO_ONE.get(r.resname, "X")
        if not 1 <= r.resseq <= len(seq):
            diffs.append((r.resseq, "?", got))
            continue
        expected = seq[r.resseq - 1]
        if got != expected:
            diffs.append((r.resseq, expected, got))
    if diffs:
        raise SequenceError(f"fasta↔pdb mismatch at {sorted(diffs)[:10]}")
