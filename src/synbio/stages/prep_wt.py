"""Stage 01: prepare WT references + masks (owned by P2)."""

import hashlib
import json
import shutil
from pathlib import Path

from synbio.orchestrator import register_stage
from synbio.orchestrator.stage import Decision, StageConfig, StageResult, StageSpec, cli

SPEC = StageSpec(
    name="prep_wt",
    module="prep_wt",
    env="dnatools",
    inputs=("sfgfp_wt_fasta", "sfgfp_wt_pdb", "sarkisyan_dms"),
    outputs=("wt", "do_not_mutate", "stab_hotspots", "position_tolerance"),
)


def _sha12(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


@register_stage(SPEC)
def run(cfg: StageConfig) -> StageResult:
    # Lazy imports: keep the module light (synbio.io pulls pandas via its __init__).
    from synbio.io.constraints import NTERM_PREFIX
    from synbio.wt import (
        build_do_not_mutate,
        build_position_tolerance,
        build_stab_hotspots,
        check_anchor,
        check_fasta_pdb_consistency,
        read_fasta_sequence,
        read_pdb_residues,
        validate_monomer,
    )

    stage_dir = Path(cfg.stage_dir)
    fasta_path = Path(cfg.inputs["sfgfp_wt_fasta"])
    pdb_path = Path(cfg.inputs["sfgfp_wt_pdb"])

    seq = read_fasta_sequence(fasta_path)
    check_anchor(seq)
    residues = read_pdb_residues(pdb_path)
    validate_monomer(residues)
    check_fasta_pdb_consistency(seq, residues)

    # WT is valid: build masks first, then the DMS-derived tolerance map.
    dnm = build_do_not_mutate(seq)
    hot = build_stab_hotspots()
    dnm_positions = {e["pos"] for e in dnm["positions"]}
    tol = build_position_tolerance(
        Path(cfg.inputs["sarkisyan_dms"]),
        len(seq),
        dnm_positions,
        **cfg.params.get("tolerance", {}),
    )

    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "do_not_mutate.json").write_text(json.dumps(dnm, indent=2))
    (stage_dir / "stab_hotspots.json").write_text(json.dumps(hot, indent=2))
    (stage_dir / "position_tolerance.json").write_text(json.dumps(tol, indent=2))

    wt_dir = stage_dir / "wt"
    wt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(fasta_path, wt_dir / "wt.fasta")
    shutil.copyfile(pdb_path, wt_dir / "wt.pdb")
    wt_json = {
        "sequence": seq,
        "length": len(seq),
        "chromophore": dnm["chromophore"],
        "numbering": dnm["numbering"],
        "nterm_freeze": {"prefix": NTERM_PREFIX, "pos2_allowed": ["S", "V"]},
        "source_hashes": {"fasta": _sha12(fasta_path), "pdb": _sha12(pdb_path)},
    }
    (wt_dir / "wt.json").write_text(json.dumps(wt_json, indent=2))

    n_hard = sum(1 for e in dnm["positions"] if e["tier"] == "hard")
    n_high = sum(1 for e in dnm["positions"] if e["tier"] == "high_risk")
    tsum = tol["summary"]
    return StageResult(
        outputs={
            "wt": "wt",
            "do_not_mutate": "do_not_mutate.json",
            "stab_hotspots": "stab_hotspots.json",
            "position_tolerance": "position_tolerance.json",
        },
        decisions=[
            Decision(
                name="wt_validated",
                note="238aa; 2B3P monomer 2-232 + CRO@66; fasta-pdb consistent",
            ),
            Decision(
                name="mask_frozen",
                kept=n_hard + n_high,
                note=f"{n_hard} hard + {n_high} high_risk",
            ),
            Decision(
                name="tolerance_map_built",
                kept=tsum["free_mutate"],
                note=(
                    f"{tsum['tolerant']} tolerant -> {tsum['free_mutate']} free-mutate "
                    f"(avGFP+amacGFP singles, minus do_not_mutate)"
                ),
            ),
        ],
        metrics={
            "length": len(seq),
            "n_hard": n_hard,
            "n_high_risk": n_high,
            "n_tolerant": tsum["tolerant"],
            "n_free_mutate": tsum["free_mutate"],
        },
    )


if __name__ == "__main__":
    cli()
