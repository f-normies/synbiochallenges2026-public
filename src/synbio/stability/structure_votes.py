"""Shared WT-only structure mapping, gate, and model seams for stage 05.

The structure votes all see the same cleaned sfGFP WT PDB: chain A, residue span 2..232,
with the chromophore positions 65/66/67 unmodelled as ATOM records. This module keeps
that numbering in one place so ThermoMPNN-D, SPURS, and ProteinMPNN-ddG cannot silently
disagree about mutation coordinates.

Two orthogonal per-candidate flags, never conflated:

- **gate_pass** — biological *veto*. False iff the candidate mutates a do_not_mutate
  (catalytic / chromophore) position. A vetoed candidate is hard-dropped from the whole
  funnel by `rank_combine` (§7-05).
- **scoreable** — *coverage*. False iff some mutation falls outside the WT PDB span
  (e.g. the disordered C-terminal tail 233..238) or on an unmodelled residue. Such a
  candidate is NOT vetoed: the WT-structure tools simply can't score it, so each
  structure vote emits ``ddg=NaN`` for it and it flows through on whatever votes *can*
  see it — for the upside slot that is the backbone-independent sequence-only ESMC vote
  (PROJECT_PLAN §5/§7-05). Conflating coverage with veto would silently delete the
  upside slot, which is exactly what stage 04's `upside_source_tokens` bypass keeps alive.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from synbio.wt import (
    build_do_not_mutate,
    check_fasta_pdb_consistency,
    read_pdb_residues,
    validate_monomer,
)

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ALPHABET)}
_MUT_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


@dataclass(frozen=True)
class StructureMutation:
    """One candidate substitution in sfGFP and WT-PDB model coordinates."""

    full_pos: int
    wt: str
    mut: str
    model_pos: int | None
    model_index: int | None
    modelable: bool

    @property
    def full_string(self) -> str:
        """Mutation in sfGFP/PDB numbering, e.g. K101E."""
        return f"{self.wt}{self.full_pos}{self.mut}"

    @property
    def tool_string(self) -> str | None:
        """Mutation in 1-based model-span numbering, e.g. full position 68 -> V67A."""
        if self.model_pos is None:
            return None
        return f"{self.wt}{self.model_pos}{self.mut}"


@dataclass(frozen=True)
class WtStructureContext:
    """WT sequence plus the coordinate map shared by the three structure votes."""

    sequence: str
    pdb_path: Path
    chain: str
    model_span: tuple[int, ...]
    modelable_positions: frozenset[int]
    protected_positions: Mapping[int, str]

    @property
    def length(self) -> int:
        return len(self.sequence)

    @property
    def model_start(self) -> int:
        return self.model_span[0]

    @property
    def model_end(self) -> int:
        return self.model_span[-1]

    @property
    def model_length(self) -> int:
        return len(self.model_span)

    def in_model_span(self, full_pos: int) -> bool:
        return self.model_start <= full_pos <= self.model_end

    def model_pos(self, full_pos: int) -> int | None:
        """Return 1-based model-span position, or None outside the PDB span."""
        if not self.in_model_span(full_pos):
            return None
        return full_pos - self.model_start + 1

    def model_index(self, full_pos: int) -> int | None:
        """Return 0-based model-span index, or None outside the PDB span."""
        pos = self.model_pos(full_pos)
        return None if pos is None else pos - 1

    def is_modelable(self, full_pos: int) -> bool:
        return full_pos in self.modelable_positions


@dataclass(frozen=True)
class CandidateStructurePlan:
    """Structure-gate result and coordinate-normalized mutations for one candidate.

    `gate_pass` is the biological veto (do_not_mutate positions); `scoreable` is WT-structure
    coverage (all mutations in-span and modelable). The two are independent — see the module
    docstring. A candidate with `gate_pass=True, scoreable=False` is deferred, not dropped.
    """

    id: Any
    mutations: tuple[StructureMutation, ...]
    gate_pass: bool
    gate_reason: str
    scoreable: bool
    coverage_reason: str

    @property
    def model_mutations(self) -> tuple[StructureMutation, ...]:
        return tuple(m for m in self.mutations if m.modelable)


def _compact_reasons(reasons: Iterable[str], limit: int = 8) -> str:
    uniq = list(dict.fromkeys(reasons))
    if not uniq:
        return "ok"
    if len(uniq) <= limit:
        return ";".join(uniq)
    return ";".join(uniq[:limit] + [f"...(+{len(uniq) - limit})"])


def load_structure_context(
    wt_dir: str | Path,
    *,
    chain: str = "A",
    gate_tiers: Sequence[str] = ("hard", "high_risk"),
) -> WtStructureContext:
    """Load and validate the prep_wt artifact used by WT-only structure models."""
    wt_dir = Path(wt_dir)
    wt = json.loads((wt_dir / "wt.json").read_text())
    seq = str(wt["sequence"])
    pdb_path = wt_dir / "wt.pdb"
    residues = read_pdb_residues(pdb_path)
    validate_monomer(residues, expect_chain=chain)
    check_fasta_pdb_consistency(seq, residues)

    chain_residues = [r for r in residues if r.chain == chain]
    if not chain_residues:
        raise ValueError(f"no residues for chain {chain!r} in {pdb_path}")
    lo = min(r.resseq for r in chain_residues)
    hi = max(r.resseq for r in chain_residues)
    model_span = tuple(range(lo, hi + 1))
    modelable = frozenset(r.resseq for r in chain_residues if r.record == "ATOM")

    wanted_tiers = set(gate_tiers)
    protected: dict[int, str] = {}
    for entry in build_do_not_mutate(seq)["positions"]:
        tier = str(entry["tier"])
        if tier in wanted_tiers:
            protected[int(entry["pos"])] = f"{tier}:{entry['reason']}"

    return WtStructureContext(
        sequence=seq,
        pdb_path=pdb_path,
        chain=chain,
        model_span=model_span,
        modelable_positions=modelable,
        protected_positions=protected,
    )


def model_sequence(seq: str, ctx: WtStructureContext, *, gap_token: str = "X") -> str:
    """Project a full sfGFP sequence onto the WT PDB span.

    ProteinMPNN's parser represents missing PDB residues as `X` after featurization. Keeping
    those three chromophore-gap slots preserves alignment after residue 64.
    """
    if len(seq) != ctx.length:
        raise ValueError(f"sequence length {len(seq)} != WT length {ctx.length}")
    return "".join(seq[pos - 1] if ctx.is_modelable(pos) else gap_token for pos in ctx.model_span)


def candidate_structure_plan(
    candidate_id: Any,
    seq: str,
    ctx: WtStructureContext,
) -> CandidateStructurePlan:
    """Diff a candidate to WT and apply the structure hard gate."""
    seq = str(seq).upper()
    if len(seq) != ctx.length:
        reason = f"length_mismatch:{len(seq)}!={ctx.length}"
        return CandidateStructurePlan(
            id=candidate_id, mutations=(), gate_pass=False, gate_reason=reason,
            scoreable=False, coverage_reason=reason,
        )
    bad = sorted(set(seq) - set(AA_ALPHABET))
    if bad:
        reason = f"nonstandard:{''.join(bad)}"
        return CandidateStructurePlan(
            id=candidate_id, mutations=(), gate_pass=False, gate_reason=reason,
            scoreable=False, coverage_reason=reason,
        )

    muts: list[StructureMutation] = []
    veto: list[str] = []       # do_not_mutate (catalytic / chromophore) -> hard-drop
    coverage: list[str] = []   # out of WT-structure coverage -> NaN this vote, NOT a drop
    for full_pos, (wt, aa) in enumerate(zip(ctx.sequence, seq, strict=True), start=1):
        if aa == wt:
            continue
        model_pos = ctx.model_pos(full_pos)
        model_index = ctx.model_index(full_pos)
        modelable = ctx.is_modelable(full_pos)
        muts.append(StructureMutation(full_pos, wt, aa, model_pos, model_index, modelable))
        if full_pos in ctx.protected_positions:
            veto.append(f"protected:{full_pos}")  # biological forbid wins over coverage
        elif model_pos is None:
            coverage.append(f"outside_structure_span:{full_pos}")
        elif not modelable:
            coverage.append(f"unmodelled:{full_pos}")

    return CandidateStructurePlan(
        id=candidate_id,
        mutations=tuple(muts),
        gate_pass=not veto,
        gate_reason=_compact_reasons(veto),
        scoreable=not coverage,
        coverage_reason=_compact_reasons(coverage),
    )


def plan_candidates(df: pd.DataFrame, ctx: WtStructureContext) -> dict[Any, CandidateStructurePlan]:
    """Build structure plans for a candidate table carrying `id` and `sequence`."""
    return {
        row.id: candidate_structure_plan(row.id, row.sequence, ctx)
        for row in df[["id", "sequence"]].itertuples(index=False)
    }


def structure_vote_frame(
    df: pd.DataFrame,
    plans: Mapping[Any, CandidateStructurePlan],
    ddg_by_id: Mapping[Any, float],
    *,
    score_status: Mapping[Any, str] | None = None,
) -> pd.DataFrame:
    """Create a vote parquet frame with the shared optional gate/debug columns."""
    score_status = {} if score_status is None else score_status
    rows: list[dict[str, Any]] = []
    for row in df[["id"]].itertuples(index=False):
        plan = plans[row.id]
        gate_pass = bool(plan.gate_pass)
        scoreable = bool(plan.scoreable)
        ddg = float(ddg_by_id[row.id]) if gate_pass and row.id in ddg_by_id else np.nan
        if not gate_pass:
            status = "gate_failed"
        elif not scoreable:
            status = score_status.get(row.id, f"not_scoreable:{plan.coverage_reason}")
        else:
            status = score_status.get(row.id, "ok" if np.isfinite(ddg) else "score_missing")
        rows.append(
            {
                "id": row.id,
                "ddg": ddg,
                "gate_pass": gate_pass,
                "gate_reason": plan.gate_reason,
                "scoreable": scoreable,
                "coverage_reason": plan.coverage_reason,
                "score_status": status,
                "n_mutations": len(plan.mutations),
                "n_model_mutations": len(plan.model_mutations),
                "mutations": ";".join(m.full_string for m in plan.mutations),
            }
        )
    return pd.DataFrame(rows)


# ThermoMPNN-D renumber_pdb (single mode) builds the key as wt + idx_to_pdb_num(...) + mut, where
# idx_to_pdb_num returns chain+resid — so the token embeds the chain: {wt}{chain}{resnum}{mut}
# (e.g. 'SA2A'). We key candidate mutations as {wt}{resnum}{mut} (StructureMutation.full_string), so
# drop the optional chain letter. Tokens already without a chain (precomputed CSVs) pass through.
_THERMO_MUT_RE = re.compile(r"^([A-Za-z])([A-Za-z]?)(\d+)([A-Za-z])$")


def _normalize_thermompnn_token(token: str) -> str:
    m = _THERMO_MUT_RE.match(token)
    if m is None:
        return token
    wt, _chain, num, mut = m.groups()
    return f"{wt}{num}{mut}"


def read_thermompnn_single_csv(path: str | Path) -> dict[str, float]:
    """Read ThermoMPNN-D single-SSM CSV into `{mutation_string: ddg}` (chain letter normalized out)."""
    df = pd.read_csv(path)
    if "Mutation" not in df.columns:
        raise ValueError(f"{path}: missing ThermoMPNN 'Mutation' column")
    ddg_col = next((c for c in ("ddG (kcal/mol)", "ddG", "ddg") if c in df.columns), None)
    if ddg_col is None:
        raise ValueError(f"{path}: missing ThermoMPNN ddG column")
    out: dict[str, float] = {}
    for mut, ddg in df[["Mutation", ddg_col]].itertuples(index=False):
        token = _normalize_thermompnn_token(str(mut).strip())
        if token and np.isfinite(float(ddg)):
            out[token] = float(ddg)
    return out


def score_from_single_mutation_lookup(
    plans: Mapping[Any, CandidateStructurePlan],
    lookup: Mapping[str, float],
) -> tuple[dict[Any, float], dict[Any, str]]:
    """Sum single-mutant ΔΔG values for each gate-passing candidate."""
    scores: dict[Any, float] = {}
    status: dict[Any, str] = {}
    for cid, plan in plans.items():
        if not plan.gate_pass:
            continue
        if not plan.scoreable:
            status[cid] = f"not_scoreable:{plan.coverage_reason}"  # defer to ESMC, do not score
            continue
        if not plan.mutations:
            scores[cid] = 0.0
            status[cid] = "wt_or_noop"
            continue
        total = 0.0
        missing: list[str] = []
        for mut in plan.mutations:
            keys = [mut.full_string]
            if mut.tool_string is not None:
                keys.append(mut.tool_string)
            value = next((lookup[k] for k in keys if k in lookup), None)
            if value is None:
                missing.append(mut.full_string)
            else:
                total += float(value)
        if missing:
            status[cid] = "score_missing:" + ",".join(missing[:8])
            continue
        scores[cid] = total
        status[cid] = "ok"
    return scores, status


def ensure_thermompnn_ssm_csv(
    pdb_path: str | Path,
    out_dir: str | Path,
    params: Mapping[str, Any],
) -> Path:
    """Return a ThermoMPNN-D single-SSM CSV, running upstream v2_ssm.py if needed."""
    cached = str(params.get("thermompnn_ssm_csv", "") or "")
    if cached:
        path = Path(cached)
        if not path.exists():
            raise FileNotFoundError(f"thermompnn_ssm_csv not found: {path}")
        return path

    repo = Path(params.get("thermompnn_repo", "docker/repos/ThermoMPNN-D"))
    # Absolute paths: the subprocess runs with cwd=repo, so any repo-root-relative path (script,
    # --out) would resolve against repo and corrupt (doubling the script path). pdb_path is already
    # absolute from the caller.
    script = (repo / "v2_ssm.py").resolve()
    if not script.exists():
        raise FileNotFoundError(
            f"ThermoMPNN-D script not found at {script}; initialize submodules or set "
            "score_stability.thermompnn_repo"
        )
    out_prefix = (Path(out_dir) / "thermompnn_single_ssm").resolve()
    argv = [
        str(params.get("python", "python")),
        str(script),
        "--mode",
        "single",
        "--pdb",
        str(pdb_path),
        "--chains",
        str(params.get("pdb_chain", "A")),
        "--threshold",
        str(float(params.get("thermompnn_threshold", 100.0))),
        "--batch_size",
        str(int(params.get("thermompnn_batch_size", 256))),
        "--out",
        str(out_prefix),
    ]
    subprocess.run(argv, cwd=repo, check=True)
    csv_path = out_prefix.with_suffix(".csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"ThermoMPNN-D did not create expected CSV: {csv_path}")
    return csv_path


def _write_fasta(records: Sequence[tuple[str, str]], path: Path) -> None:
    lines: list[str] = []
    for name, seq in records:
        lines.extend([f">{name}", seq])
    path.write_text("\n".join(lines) + "\n")


def run_proteinmpnn_score_only(
    pdb_path: str | Path,
    records: Sequence[tuple[str, str]],
    out_dir: str | Path,
    params: Mapping[str, Any],
    seed: int,
) -> dict[str, float]:
    """Run upstream ProteinMPNN score-only and return global scores keyed by FASTA header."""
    if not records:
        return {}
    # Absolute paths: the subprocess runs with cwd=repo, so repo-root-relative paths (script,
    # --path_to_fasta, --out_folder) would resolve against repo and corrupt (doubling the script).
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    repo = Path(params.get("proteinmpnn_repo", "docker/repos/ProteinMPNN"))
    script = (repo / "protein_mpnn_run.py").resolve()
    if not script.exists():
        raise FileNotFoundError(
            f"ProteinMPNN script not found at {script}; initialize submodules or set "
            "score_stability.proteinmpnn_repo"
        )

    num_seq = int(params.get("proteinmpnn_num_seq_per_target", 4))
    batch_size = int(params.get("proteinmpnn_batch_size", 1))
    if num_seq < batch_size or num_seq % batch_size != 0:
        raise ValueError("proteinmpnn_num_seq_per_target must be >= and divisible by batch_size")

    fasta_path = out_dir / "proteinmpnn_score_input.fa"
    _write_fasta(records, fasta_path)
    argv = [
        str(params.get("python", "python")),
        str(script),
        "--score_only",
        "1",
        "--pdb_path",
        str(pdb_path),
        "--pdb_path_chains",
        str(params.get("pdb_chain", "A")),
        "--path_to_fasta",
        str(fasta_path),
        "--out_folder",
        str(out_dir / "proteinmpnn_out"),
        "--num_seq_per_target",
        str(num_seq),
        "--batch_size",
        str(batch_size),
        "--seed",
        str(int(seed)),
        "--suppress_print",
        "1",
        "--model_name",
        str(params.get("proteinmpnn_model_name", "v_48_020")),
    ]
    weights = str(params.get("proteinmpnn_weights", "") or "")
    if weights:
        argv.extend(["--path_to_model_weights", weights])
    subprocess.run(argv, cwd=repo, check=True)

    score_dir = out_dir / "proteinmpnn_out" / "score_only"
    pdb_files = sorted(score_dir.glob("*_pdb.npz"))
    if len(pdb_files) != 1:
        raise FileNotFoundError(f"expected one ProteinMPNN *_pdb.npz in {score_dir}, found {pdb_files}")
    scores = {"__wt__": float(np.load(pdb_files[0])["global_score"].mean())}
    for i, (name, _) in enumerate(records, start=1):
        hits = sorted(score_dir.glob(f"*_fasta_{i}.npz"))
        if len(hits) != 1:
            raise FileNotFoundError(
                f"expected one ProteinMPNN *_fasta_{i}.npz in {score_dir}, found {hits}"
            )
        scores[name] = float(np.load(hits[0])["global_score"].mean())
    return scores


def spurs_padded_tensors(
    mutation_lists: Sequence[Sequence[StructureMutation]],
) -> tuple[np.ndarray, np.ndarray]:
    """Build padded SPURS mut_ids and append_tensors arrays for variable mutation counts."""
    max_mut = max(1, max(len(muts) for muts in mutation_lists))
    mut_ids = np.full((len(mutation_lists), max_mut), -1, dtype=np.int64)
    append = np.full((len(mutation_lists), max_mut, 2), -1, dtype=np.int64)
    for i, muts in enumerate(mutation_lists):
        for j, mut in enumerate(muts):
            if mut.model_index is None:
                raise ValueError(f"mutation {mut.full_string} is not model-indexed")
            mut_ids[i, j] = mut.model_index
            append[i, j, 0] = AA_TO_IDX[mut.wt]
            append[i, j, 1] = AA_TO_IDX[mut.mut]
    return mut_ids, append


def predict_spurs_multi(
    pdb_path: str | Path,
    mutation_lists: Sequence[Sequence[StructureMutation]],
    params: Mapping[str, Any],
) -> np.ndarray:
    """Run SPURS multi-mutant inference; heavy and intended for the server."""
    if not mutation_lists:
        return np.asarray([], dtype=float)
    import torch
    from spurs.inference import get_SPURS_multi_from_hub, parse_pdb

    device = str(params.get("spurs_device", "auto"))
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    repo_id = str(params.get("spurs_repo_id", "cyclization9/SPURS"))
    chain = str(params.get("pdb_chain", "A"))
    batch_size = int(params.get("spurs_batch_size", 64))
    score_sign = float(params.get("spurs_score_sign", -1.0))

    model, cfg = get_SPURS_multi_from_hub(repo_id=repo_id, device=device)
    model.eval()
    pdb = parse_pdb(str(pdb_path), Path(pdb_path).stem, chain, cfg, device=device)

    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(mutation_lists), batch_size):
            chunk = mutation_lists[start : start + batch_size]
            mut_ids, append = spurs_padded_tensors(chunk)
            pdb["mut_ids"] = torch.tensor(mut_ids, device=device)
            pdb["append_tensors"] = torch.tensor(append, device=device)
            pred = model(pdb).detach().cpu().numpy()
            out.append(np.atleast_1d(pred).astype(float))
    return score_sign * np.concatenate(out)


def parse_mutation_token(token: str) -> tuple[str, int, str]:
    """Parse an AA-position-AA mutation token."""
    m = _MUT_RE.match(token.strip())
    if not m:
        raise ValueError(f"bad mutation token: {token!r}")
    return m.group(1), int(m.group(2)), m.group(3)
