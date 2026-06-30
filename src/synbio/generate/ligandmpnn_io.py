"""Build the LigandMPNN run.py argv and parse its output FASTA."""

import re
from dataclasses import dataclass
from pathlib import Path

__all__ = ["DesignRecord", "build_argv", "parse_fasta"]

_KV = re.compile(r"(\w+)=([0-9.eE+-]+)")


@dataclass(frozen=True)
class DesignRecord:
    """One LigandMPNN design parsed from the output FASTA."""

    id: str
    sequence: str
    overall_confidence: float
    ligand_confidence: float


def build_argv(
    *,
    run_script: str,
    pdb_path: str,
    out_folder: str,
    redesigned_residues: str,
    bias_json_path: str,
    omit_aa: str,
    temperature: float,
    seed: int,
    batch_size: int,
    number_of_batches: int,
    checkpoint: str = "",
    model_type: str = "ligand_mpnn",
    use_atom_context: int = 1,
    use_side_chain_context: int = 1,
) -> list[str]:
    """Construct the `python run.py ...` argv for one LigandMPNN invocation."""
    argv = [
        "python", run_script,
        "--model_type", model_type,
        "--pdb_path", pdb_path,
        "--out_folder", out_folder,
        "--temperature", str(temperature),
        "--seed", str(seed),
        "--batch_size", str(batch_size),
        "--number_of_batches", str(number_of_batches),
        "--redesigned_residues", redesigned_residues,
        "--bias_AA_per_residue", bias_json_path,
        "--omit_AA", omit_aa,
        "--ligand_mpnn_use_atom_context", str(use_atom_context),
        "--ligand_mpnn_use_side_chain_context", str(use_side_chain_context),
        "--save_stats", "1",
    ]
    if checkpoint:
        argv += ["--checkpoint_ligand_mpnn", checkpoint]
    return argv


def parse_fasta(fasta_path: str | Path) -> list[DesignRecord]:
    """Parse design records (headers with `id=`); skip the leading native record."""
    text = Path(fasta_path).read_text()
    records: list[DesignRecord] = []
    header: str | None = None
    seq_lines: list[str] = []

    def flush() -> None:
        if header is None:
            return
        kv = dict(_KV.findall(header))
        if "id" in kv:  # design record (native record has no id=)
            records.append(
                DesignRecord(
                    id=kv["id"],
                    sequence="".join(seq_lines),
                    overall_confidence=float(kv.get("overall_confidence", "nan")),
                    ligand_confidence=float(kv.get("ligand_confidence", "nan")),
                )
            )

    for line in text.splitlines():
        if line.startswith(">"):
            flush()
            header = line[1:]
            seq_lines = []
        elif line.strip():
            seq_lines.append(line.strip())
    flush()
    return records
