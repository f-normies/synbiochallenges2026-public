"""Fixed-column PDB reader + WT monomer validation (no BioPython)."""

from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "PdbResidue",
    "THREE_TO_ONE",
    "StructureError",
    "read_pdb_residues",
    "validate_monomer",
]

THREE_TO_ONE: dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


class StructureError(ValueError):
    """Raised when the WT PDB violates the cleaned-monomer assumptions."""


@dataclass(frozen=True)
class PdbResidue:
    """One residue parsed from a PDB ATOM/HETATM block."""

    record: str  # "ATOM" or "HETATM"
    chain: str
    resseq: int
    resname: str
    has_altloc: bool


def read_pdb_residues(path: str | Path) -> list[PdbResidue]:
    """Parse ATOM/HETATM records into one PdbResidue per (record, chain, resseq).

    has_altloc is True if any atom of that residue carried a non-blank altLoc.
    """
    residues: dict[tuple[str, str, int], PdbResidue] = {}
    for line in Path(path).read_text().splitlines():
        record = line[0:6].strip()
        if record not in ("ATOM", "HETATM"):
            continue
        altloc = line[16:17].strip()
        resname = line[17:20].strip()
        chain = line[21:22].strip()
        try:
            resseq = int(line[22:26])
        except ValueError:
            continue
        key = (record, chain, resseq)
        existing = residues.get(key)
        if existing is None:
            residues[key] = PdbResidue(record, chain, resseq, resname, bool(altloc))
        elif altloc:
            residues[key] = PdbResidue(record, chain, resseq, existing.resname, True)
    return list(residues.values())


def validate_monomer(
    residues: list[PdbResidue],
    *,
    expect_chain: str = "A",
    expect_range: tuple[int, int] = (2, 232),
    cro_resid: int = 66,
    cro_span: tuple[int, int, int] = (65, 66, 67),
) -> None:
    """Validate the cleaned single-chain WT monomer; raise StructureError on any violation.

    cro_span lists the residue numbers occupied by the fused CRO chromophore in the cleaned
    PDB (65/66/67); these positions must NOT appear as ATOM records (only CRO at cro_resid).
    """
    if cro_resid not in cro_span:
        raise StructureError(f"cro_resid {cro_resid} must be within cro_span {cro_span}")
    chains = {r.chain for r in residues}
    if chains != {expect_chain}:
        raise StructureError(f"expected single chain {expect_chain!r}, found {sorted(chains)}")

    alt = sorted(r.resseq for r in residues if r.has_altloc)
    if alt:
        raise StructureError(f"alternate-location records present at residues {alt}")

    het = [r for r in residues if r.record == "HETATM"]
    if len(het) != 1 or het[0].resname != "CRO" or het[0].resseq != cro_resid:
        raise StructureError(
            f"expected exactly one CRO HETATM at {cro_resid}, "
            f"found {[(h.resname, h.resseq) for h in het]}"
        )

    lo, hi = expect_range
    expected = set(range(lo, hi + 1)) - set(cro_span)
    present = {r.resseq for r in residues if r.record == "ATOM"}
    missing = expected - present
    extra = present - expected
    if missing:
        raise StructureError(f"missing ATOM residues: {sorted(missing)}")
    if extra:
        raise StructureError(f"unexpected ATOM residues: {sorted(extra)}")
