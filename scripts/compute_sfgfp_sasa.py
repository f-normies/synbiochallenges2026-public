"""Offline: relative SASA of the frozen sfGFP monomer -> data/sfgfp_sasa.json.

WT is frozen, so SASA is a constant; computing it once (dev .venv, real freesasa)
keeps the runtime dnatools env free of a structural dependency. Re-run only if the
WT PDB changes:  python scripts/compute_sfgfp_sasa.py
"""

import json
from pathlib import Path

import freesasa

ROOT = Path(__file__).resolve().parents[1]
PDB = ROOT / "data" / "sfgfp_wt.pdb"
OUT = ROOT / "data" / "sfgfp_sasa.json"


def compute_rel_sasa(pdb_path: str) -> dict[int, float]:
    """Return {1-based residue number -> relative total SASA} (non-null only)."""
    structure = freesasa.Structure(pdb_path)
    result = freesasa.calc(structure)
    areas = result.residueAreas()
    chain = sorted(areas.keys())[0]
    rel: dict[int, float] = {}
    for resnum, area in areas[chain].items():
        if area.relativeTotal is not None:
            rel[int(resnum)] = round(float(area.relativeTotal), 6)
    return dict(sorted(rel.items()))


def main() -> None:
    rel = compute_rel_sasa(str(PDB))
    payload = {
        "source": "data/sfgfp_wt.pdb",
        "method": "freesasa Shrake-Rupley (default params)",
        "numbering": "1-based; 2B3P chain A",
        "n_residues": len(rel),
        "rel_sasa": {str(k): v for k, v in rel.items()},
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT} ({len(rel)} residues)")


if __name__ == "__main__":
    main()
