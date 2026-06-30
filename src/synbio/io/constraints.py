"""Hard sequence constraints, applied at the output of every generation sub-stage.

Pure functions only; centralised so the rules cannot drift across stages.
Rules (from spec §9.1): length 220–250, start with M, 20 standard upper-case AA,
not in the exclusion list, optional N-terminus prefix freeze.
"""

__all__ = [
    "STANDARD_AA",
    "MIN_LEN",
    "MAX_LEN",
    "NTERM_PREFIX",
    "validate_sequence",
]

STANDARD_AA: frozenset[str] = frozenset("ACDEFGHIKLMNPQRSTVWY")
MIN_LEN: int = 220
MAX_LEN: int = 250
NTERM_PREFIX: str = "MSKGEELFTG"


def validate_sequence(
    seq: str,
    exclusion: set[str],
    require_nterm: bool = False,
) -> tuple[bool, list[str]]:
    """Validate one amino-acid sequence against the hard constraints.

    Args:
        seq: Upper-case amino-acid string.
        exclusion: Set of forbidden full-length sequences (Exclusion_List + FPbase).
        require_nterm: If True, the first 10 residues must equal NTERM_PREFIX.

    Returns:
        (is_valid, violations) where violations is a list of human-readable reasons.
    """
    violations: list[str] = []

    if not (MIN_LEN <= len(seq) <= MAX_LEN):
        violations.append(f"length {len(seq)} outside [{MIN_LEN}, {MAX_LEN}]")
    if not seq.startswith("M"):
        violations.append("must start with methionine (M)")
    bad = sorted(set(seq) - STANDARD_AA)
    if bad:
        violations.append(f"non-standard characters present: {''.join(bad)}")
    if seq in exclusion:
        violations.append("sequence is on the exclusion list")
    if require_nterm and not seq.startswith(NTERM_PREFIX):
        violations.append(f"N-terminus must match prefix {NTERM_PREFIX}")

    return (not violations, violations)
