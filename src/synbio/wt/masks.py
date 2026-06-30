"""WT mutation masks (do_not_mutate) and stability hotspots (stab_hotspots).

Position numbering is 1-based and aligned to PDB 2B3P chain A. wt_residue is always
read from the actual sfGFP sequence, never from the (avGFP-canonical) plan table.
"""

from typing import Any

__all__ = [
    "HARD_POSITIONS",
    "HIGH_RISK_POSITIONS",
    "TABLE_RESIDUE_OVERRIDES",
    "build_do_not_mutate",
    "build_stab_hotspots",
]

# pos -> (reason, extra allowed residues beyond the WT residue)
HARD_POSITIONS: dict[int, tuple[str, tuple[str, ...]]] = {
    64: ("packs the central helix (sfGFP carries the F64L folding mutation)", ()),
    65: ("chromophore Thr; cyclisation kinetics", ()),
    66: ("chromophore Tyr; chromophore aromatic", ()),
    67: ("chromophore Gly; backbone geometry", ()),
    68: ("preorganises the Gly67 psi angle", ()),
    94: ("H-bonds the imidazolinone carbonyl", ()),
    96: ("catalytic Arg; stabilises oxyanion + anionic phenolate", ()),
    145: ("chromophore-pocket aromatic (sfGFP carries Y145F)", ("Y",)),
    148: ("proton-wire His; H-bond donor to the phenolate", ()),
    165: ("buried core hydrophobic", ()),
    167: ("buried hydrophobic", ()),
    181: ("buried polar; coordinates the internal water network", ()),
    203: ("proton-wire Thr; H-bonds the chromophore phenol", ()),
    205: ("proton-wire Ser; bridge to water and Glu222", ()),
    220: ("packs the Glu222 rotamer", ()),
    222: ("catalytic Glu; phenol (de)protonation via the wire", ()),
    224: ("buried hydrophobic against the central helix", ()),
}

# Conserved cis-prolines + chromophore-facing aromatics not already in HARD.
HIGH_RISK_POSITIONS: dict[int, str] = {
    56: "conserved cis-proline; slow folding isomerisation",
    75: "conserved cis-proline; slow folding isomerisation",
    89: "conserved cis-proline; slow folding isomerisation",
    27: "chromophore-facing aromatic",
    46: "chromophore-facing aromatic",
    92: "chromophore-facing aromatic",
    106: "chromophore-facing aromatic",
    151: "chromophore-facing aromatic",
    200: "chromophore-facing aromatic",
    223: "chromophore-facing aromatic",
}

# Positions where the plan §4.2 table residue disagrees with the real sfGFP residue.
TABLE_RESIDUE_OVERRIDES: dict[int, str] = {181: "D"}


def _entry(seq: str, pos: int, tier: str, reason: str, extra: tuple[str, ...]) -> dict[str, Any]:
    wt = seq[pos - 1]
    allowed = [wt] + [a for a in extra if a != wt]
    entry: dict[str, Any] = {
        "pos": pos,
        "wt_residue": wt,
        "tier": tier,
        "reason": reason,
        "allowed_residues": allowed,
    }
    claimed = TABLE_RESIDUE_OVERRIDES.get(pos)
    if claimed is not None and claimed != wt:
        entry["note"] = (
            f"plan §4.2 table lists {claimed} here, but sfGFP residue {pos} is {wt}; "
            f"kept as a hard mask, residue recorded from the sequence"
        )
    return entry


def build_do_not_mutate(seq: str) -> dict[str, Any]:
    """Build the tiered do_not_mutate mask from the WT sequence.

    Args:
        seq: Full sfGFP amino acid sequence (1-based indexing used internally).

    Returns:
        Dict with keys: wt, length, numbering, chromophore, positions.
        positions is a list of dicts sorted by pos, each with:
        pos, wt_residue, tier (hard | high_risk), reason, allowed_residues,
        and optionally note for documented table anomalies.
    """
    positions: list[dict[str, Any]] = []
    for pos, (reason, extra) in HARD_POSITIONS.items():
        positions.append(_entry(seq, pos, "hard", reason, extra))
    for pos, reason in HIGH_RISK_POSITIONS.items():
        if pos in HARD_POSITIONS:
            continue  # hard wins on overlap
        positions.append(_entry(seq, pos, "high_risk", reason, ()))
    positions.sort(key=lambda e: e["pos"])
    return {
        "wt": "sfGFP",
        "length": len(seq),
        "numbering": "1-based; matches PDB 2B3P chain A",
        "chromophore": {
            "residues": [65, 66, 67],
            "triad": "TYG",
            "pdb_resname": "CRO",
            "pdb_resid": 66,
        },
        "positions": positions,
    }


def build_stab_hotspots() -> dict[str, Any]:
    """Build the stability-hotspot prior: three §4.3 classes + sfGFP-resident mutations.

    Surface recharge is the carrying thermostability lever (the 72C readout is dominated
    by irreversible aggregation); cavity fill and avGFP-family consensus are secondary.
    Engineered disulfides are excluded (cell-free expression is a reducing environment).
    """
    return {
        "wt": "sfGFP",
        "numbering": "1-based; matches PDB 2B3P chain A",
        "classes": [
            {
                "name": "surface_recharge_negative",
                "priority": 1,
                "strategy": (
                    "TGP-style: solvent-exposed K -> E/Q, lower pI; addresses "
                    "irreversible aggregation at 72C (carrying lever)"
                ),
                "positions": [],
                "needs": "exposed-Lys enumeration (SASA) - deferred to the generate stage",
                "source": "TGP",
            },
            {
                "name": "buried_cavity_fill",
                "priority": 2,
                "strategy": (
                    "V -> I/L/F in the core distal to the chromophore, rotamer-guided, "
                    "only at DMS-neutral positions; +3-8C Tm"
                ),
                "source": "general",
            },
            {
                "name": "consensus_avgfp_family",
                "priority": 3,
                "strategy": (
                    "consensus over the avGFP family; amacGFP (~83% id) is the reliable "
                    "transfer source, cgreGFP/ppluGFP contribute MSA consensus signal only"
                ),
                "positions": [],
                "needs": "family MSA alignment onto sfGFP - deferred to the generate stage",
                "source": "amacGFP (avGFP-family MSA)",
            },
        ],
        "already_present_in_sfgfp": [
            "S30R", "Y39N", "F64L", "F99S", "N105T", "Y145F",
            "M153T", "V163A", "I171V", "A206V",
        ],
    }
