from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class AtomRecord:
    resid: int
    atom: str
    resname: str
    xyz: np.ndarray
    hetero: bool = False


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def parse_pdb_atoms(text: str) -> list[AtomRecord]:
    atoms: list[AtomRecord] = []
    for line in text.splitlines():
        record = line[:6].strip()
        if record not in {"ATOM", "HETATM"}:
            continue
        atoms.append(
            AtomRecord(
                resid=int(line[22:26]),
                atom=line[12:16].strip(),
                resname=line[17:20].strip(),
                xyz=np.array(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                    dtype=float,
                ),
                hetero=record == "HETATM",
            )
        )
    return atoms


def ca_contact_pairs(
    atoms: Iterable[AtomRecord],
    *,
    min_seq_sep: int,
    cutoff: float,
) -> list[tuple[int, int]]:
    ca = {a.resid: a.xyz for a in atoms if a.atom == "CA" and not a.hetero}
    pairs: list[tuple[int, int]] = []
    for i in sorted(ca):
        for j in sorted(ca):
            if j <= i or j - i < int(min_seq_sep):
                continue
            if _dist(ca[i], ca[j]) <= float(cutoff):
                pairs.append((i, j))
    return pairs


def ca_coordinates(atoms: Iterable[AtomRecord]) -> dict[int, np.ndarray]:
    return {a.resid: a.xyz for a in atoms if a.atom == "CA" and not a.hetero}


def contact_retention_fraction(
    pairs: Iterable[tuple[int, int]],
    ca_coords: dict[int, np.ndarray],
    *,
    cutoff: float,
    slack: float,
) -> float:
    pairs = list(pairs)
    if not pairs:
        return 1.0
    kept = 0
    for i, j in pairs:
        if i in ca_coords and j in ca_coords and _dist(ca_coords[i], ca_coords[j]) <= cutoff + slack:
            kept += 1
    return kept / len(pairs)


_BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O", "OXT"})


def _is_heavy_atom(atom_name: str) -> bool:
    return not atom_name.strip().upper().startswith("H")


def _is_sidechain_heavy(atom_name: str) -> bool:
    name = atom_name.strip().upper()
    return _is_heavy_atom(name) and name not in _BACKBONE_ATOMS


def min_heavy_atom_distance(
    residue_atoms: Iterable[AtomRecord],
    chrom_atoms: Iterable[AtomRecord],
    *,
    exclude_backbone_left: bool = False,
) -> float:
    """Minimum heavy-atom distance between a residue and the chromophore.

    With `exclude_backbone_left` the residue side is restricted to side-chain
    heavy atoms (spec §7.3: catalytic side-chain proximity is the meaningful
    signal). If the residue has no side-chain heavy atoms (e.g. Gly) we fall
    back to all heavy atoms so the distance stays defined.
    """
    residue_atoms = list(residue_atoms)
    if exclude_backbone_left:
        left = [a for a in residue_atoms if _is_sidechain_heavy(a.atom)]
        if not left:
            left = [a for a in residue_atoms if _is_heavy_atom(a.atom)]
    else:
        left = [a for a in residue_atoms if _is_heavy_atom(a.atom)]
    right = [a for a in chrom_atoms if _is_heavy_atom(a.atom)]
    if not left or not right:
        return float("nan")
    return min(_dist(a.xyz, b.xyz) for a in left for b in right)


def pocket_distance_deltas(wt: dict[int, float], candidate: dict[int, float]) -> dict[int, float]:
    out: dict[int, float] = {}
    for pos, wt_value in wt.items():
        cand_value = candidate.get(pos, float("nan"))
        out[pos] = abs(float(cand_value) - float(wt_value))
    return out


def pocket_distances_to_chromophore(
    atoms: Iterable[AtomRecord],
    *,
    pocket_positions: Iterable[int],
    chromophore_positions: Iterable[int],
    exclude_backbone_left: bool = False,
) -> tuple[dict[int, float], bool]:
    atoms = list(atoms)
    cro_atoms = [a for a in atoms if a.resname == "CRO"]
    used_cro = bool(cro_atoms)
    chrom_atoms = cro_atoms or [a for a in atoms if a.resid in set(chromophore_positions)]
    out: dict[int, float] = {}
    for pos in pocket_positions:
        residue_atoms = [a for a in atoms if a.resid == int(pos)]
        out[int(pos)] = min_heavy_atom_distance(
            residue_atoms, chrom_atoms, exclude_backbone_left=exclude_backbone_left
        )
    return out, used_cro
