"""Position tolerance map from ortholog DMS (avGFP/amacGFP), keyed to sfGFP positions.

A position is "tolerant" when most measured single substitutions keep brightness near
WT; those are where generation may place brightness / surface-recharge edits. The map
is the POSITIVE signal only — the hand-curated do_not_mutate mask stays the authoritative
forbid, because the single-mutant DMS is sparse (~5 subs/position for avGFP) and can
mislabel catalytic positions (R96/E222/T203/H148) as tolerant. So free_mutate_positions
is the tolerant set MINUS do_not_mutate.

The `aaMutations` token's number is a 0-based index into the ortholog's FASTA sequence,
not a 1-based position (verified 100% on single-substitution rows: the token's source
letter equals fasta[number], not fasta[number-1]). It is converted to a 1-based sfGFP
position via +1. avGFP, amacGFP and sfGFP are all 238 aa and still align 1:1 by residue
index (gapless: chromophore S/T65-Y66-G67, catalytic R96/E222 line up), so once
converted to 1-based, positions transfer directly to sfGFP. Functional calls are made
per ortholog against that ortholog's own WT brightness.
"""

import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from synbio.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["build_position_tolerance"]

_MUT = re.compile(r"^([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])$")
_DEFAULT_TYPES: tuple[str, ...] = ("avGFP", "amacGFP")


def build_position_tolerance(
    dms_path: str | Path,
    length: int,
    do_not_mutate: set[int],
    *,
    gfp_types: tuple[str, ...] = _DEFAULT_TYPES,
    eps: float = 0.8,
    tol_hi: float = 0.6,
    tol_lo: float = 0.2,
    min_cov: int = 3,
) -> dict[str, Any]:
    """Build the per-position tolerance map from ortholog single-mutant DMS.

    Args:
        dms_path: sarkisyan_dms.csv (columns: aaMutations, GFP type, Brightness).
        length: WT length; positions are numbered 1..length.
        do_not_mutate: Hard-mask positions, excluded from free_mutate_positions.
        gfp_types: Orthologs to pool (must align 1:1 to sfGFP; default avGFP+amacGFP).
        eps: A substitution is "functional" if Brightness >= WT_of_its_type - eps.
        tol_hi: tier "tolerant" if functional fraction >= tol_hi.
        tol_lo: tier "intolerant" if functional fraction <= tol_lo.
        min_cov: positions with fewer single subs than this are tier "unknown".

    Returns:
        Dict with wt, length, numbering, source, params, wt_brightness, positions
        (one per residue, sorted), free_mutate_positions, and a summary count block.
    """
    types = tuple(gfp_types)
    wt_brightness: dict[str, list[float]] = defaultdict(list)
    raw: dict[int, list[tuple[str, float]]] = defaultdict(list)  # pos -> [(type, b)]

    with open(dms_path, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            gtype = row["GFP type"].strip()
            if gtype not in types:
                continue
            try:
                brightness = float(row["Brightness"])
            except (ValueError, KeyError):
                continue
            mut = row["aaMutations"].strip()
            if mut in ("", "WT"):
                wt_brightness[gtype].append(brightness)
                continue
            parts = [p for p in mut.split(":") if p]
            if len(parts) != 1:
                continue
            m = _MUT.match(parts[0])
            if not m:
                continue
            # aaMutations' number is 0-based into the FASTA; +1 -> 1-based sfGFP pos.
            pos = int(m.group(2)) + 1
            if 1 <= pos <= length:
                raw[pos].append((gtype, brightness))

    wt_mean = {t: sum(v) / len(v) for t, v in wt_brightness.items() if v}
    for t in types:
        if t not in wt_mean:
            logger.warning("no WT row for GFP type %s in DMS; its singles are dropped", t)

    positions: list[dict[str, Any]] = []
    summary = {"tolerant": 0, "sensitive": 0, "intolerant": 0, "unknown": 0}
    free: list[int] = []
    for pos in range(1, length + 1):
        flags = [b >= wt_mean[t] - eps for t, b in raw.get(pos, []) if t in wt_mean]
        n = len(flags)
        in_dnm = pos in do_not_mutate
        frac: float | None = None
        if n < min_cov:
            tier = "unknown"
        else:
            frac = sum(flags) / n
            tier = "tolerant" if frac >= tol_hi else "intolerant" if frac <= tol_lo else "sensitive"
        summary[tier] += 1
        positions.append({
            "pos": pos,
            "n_subs": n,
            "frac_functional": round(frac, 3) if frac is not None else None,
            "tier": tier,
            "in_do_not_mutate": in_dnm,
        })
        if tier == "tolerant" and not in_dnm:
            free.append(pos)

    return {
        "wt": "sfGFP",
        "length": length,
        "numbering": "1-based; matches PDB 2B3P chain A; DMS orthologs align 1:1",
        "source": list(types),
        "params": {"eps": eps, "tol_hi": tol_hi, "tol_lo": tol_lo, "min_cov": min_cov},
        "wt_brightness": {t: round(v, 3) for t, v in wt_mean.items()},
        "positions": positions,
        "free_mutate_positions": free,
        "summary": {**summary, "free_mutate": len(free)},
    }
