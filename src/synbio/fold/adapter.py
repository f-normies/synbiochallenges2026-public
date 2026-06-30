"""Pure ESMFold2 output adapter for stage 07 fold sanity.

Turns a `MolecularComplexResult` into residue-indexed atoms / pLDDT / pAE that
the geometry checks consume. Import-safe (numpy only; no torch/esm), so it can
be unit-tested in the dev `.venv` against fake complex objects.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from .geometry import AtomRecord

__all__ = [
    "safe_float",
    "records_by_residue",
    "adapt_fold_result",
    "adapt_molecular_complex",
    "mean_by_positions",
    "pocket_pae",
]


def safe_float(value) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def records_by_residue(raw_atoms: list[dict]) -> dict[int, list[AtomRecord]]:
    grouped: dict[int, list[AtomRecord]] = defaultdict(list)
    for atom in raw_atoms:
        rec = AtomRecord(
            resid=int(atom["resid"]),
            atom=str(atom["atom"]),
            resname=str(atom["resname"]),
            xyz=np.asarray(atom["xyz"], dtype=float),
            hetero=bool(atom.get("hetero", False)),
        )
        grouped[rec.resid].append(rec)
    return dict(grouped)


def adapt_fold_result(result, sequence: str) -> dict[str, Any]:
    n_tokens = len(result.complex.sequence)
    if n_tokens < len(sequence):
        # CRO (and other CCD modifications) can collapse several sequence
        # residues into one token. resid = token_idx + 1 would then shift every
        # residue past the chromophore, silently corrupting pocket distances and
        # barrel contacts. Fail loud instead of trusting a misaligned mapping.
        raise ValueError(
            f"ESMFold2 returned {n_tokens} tokens for a {len(sequence)}-residue "
            "sequence; token->resid alignment is unsafe (likely CRO collapse)"
        )
    full_residue_numbers = list(range(1, len(sequence) + 1))
    return adapt_molecular_complex(
        result.complex,
        full_residue_numbers=full_residue_numbers,
        pae=None if result.pae is None else result.pae.detach().cpu().numpy(),
    )


def adapt_molecular_complex(complex_obj, *, full_residue_numbers: list[int], pae=None) -> dict[str, Any]:
    raw_atoms: list[dict[str, Any]] = []
    plddt_by_resid: dict[int, float] = {}
    token_to_resid: dict[int, int] = {}
    token_resid_cursor = 0
    previous_resid = None
    for token_idx, token in enumerate(complex_obj.sequence):
        if token_idx < len(full_residue_numbers):
            resid = int(full_residue_numbers[token_idx])
        else:
            resid = int(full_residue_numbers[min(token_resid_cursor, len(full_residue_numbers) - 1)])
        if resid != previous_resid:
            token_resid_cursor += 1
            previous_resid = resid
        token_to_resid[token_idx] = resid
        plddt_by_resid.setdefault(resid, safe_float(complex_obj.plddt[token_idx]))
        start, end = complex_obj.token_to_atoms[token_idx]
        for atom_idx in range(int(start), int(end)):
            atom_name = (
                str(complex_obj.atom_names[atom_idx])
                if getattr(complex_obj, "atom_names", None) is not None
                else f"X{atom_idx}"
            )
            raw_atoms.append(
                {
                    "resid": resid,
                    "atom": atom_name.upper().strip(),
                    "resname": str(token).upper().strip(),
                    "xyz": np.asarray(complex_obj.atom_positions[atom_idx], dtype=float).tolist(),
                    "hetero": bool(getattr(complex_obj, "atom_hetero", [False] * len(complex_obj.atom_positions))[atom_idx]),
                }
            )
    pae_by_resid: dict[tuple[int, int], float] = {}
    if pae is not None:
        pae_arr = np.asarray(pae, dtype=float)
        n = min(pae_arr.shape[0], len(token_to_resid))
        for i in range(n):
            for j in range(n):
                pae_by_resid[(token_to_resid[i], token_to_resid[j])] = safe_float(pae_arr[i, j])
    return {"raw_atoms": raw_atoms, "plddt_by_resid": plddt_by_resid, "pae_by_resid": pae_by_resid}


def mean_by_positions(values: dict[int, float], positions: list[int]) -> float:
    vals = [float(values[pos]) for pos in positions if pos in values and np.isfinite(values[pos])]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def pocket_pae(pae_by_resid: dict[tuple[int, int], float], positions: list[int]) -> float:
    vals = []
    for i in positions:
        for j in positions:
            if i == j:
                continue
            value = pae_by_resid.get((int(i), int(j)), float("nan"))
            if np.isfinite(value):
                vals.append(float(value))
    if not vals:
        return float("nan")
    return float(np.mean(vals))
