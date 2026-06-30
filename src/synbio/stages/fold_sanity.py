"""Stage 07: chromophore-aware ESMFold2 negative filter (owned by P4)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from synbio.orchestrator import register_stage
from synbio.orchestrator.stage import Decision, StageConfig, StageResult, StageSpec, cli

from synbio.fold import (
    AtomRecord,
    ca_contact_pairs,
    contact_retention_fraction,
    decide_fold_pass,
    parse_pdb_atoms,
    pocket_distance_deltas,
    pocket_distances_to_chromophore,
    precheck_sequence,
)
from synbio.fold.adapter import (
    adapt_fold_result as _adapt_fold_result,
    mean_by_positions as _mean_by_positions,
    pocket_pae as _pocket_pae,
    records_by_residue as _records_by_residue,
    safe_float as _safe_float,
)
from synbio.fold.geometry import ca_coordinates

SPEC = StageSpec(
    name="fold_sanity",
    module="fold_sanity",
    env="esm",
    inputs=("candidates_ranked", "wt"),
    outputs=("candidates_folded",),
)


@register_stage(SPEC)
def run(cfg: StageConfig) -> StageResult:
    from synbio.io.artifacts import read_candidates, write_candidates

    stage_dir = Path(cfg.stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    df = read_candidates(cfg.inputs["candidates_ranked"]).reset_index(drop=True)
    wt_dir = Path(cfg.inputs["wt"])
    wt_seq = json.loads((wt_dir / "wt.json").read_text())["sequence"]
    wt_atoms = parse_pdb_atoms((wt_dir / "wt.pdb").read_text())
    baseline_atoms, baseline_mode = _select_wt_baseline(wt_seq, wt_atoms, cfg.params, cfg.seed)
    wt_context = _build_wt_context(baseline_atoms, cfg.params)

    rows: list[dict[str, Any]] = []
    metrics_path = stage_dir / "fold_metrics.jsonl"
    n_pass = 0
    n_precheck_failed = 0
    n_fold_failed = 0
    with metrics_path.open("w") as metrics_fh:
        fold_top_n = int(cfg.params.get("fold_top_n", len(df)))
        for idx, row in enumerate(df.itertuples(index=False)):
            candidate_id = str(row.id)
            pre_ok, pre_reasons = precheck_sequence(str(row.sequence))
            metrics = _failed_metrics("precheck_failed")
            forced_reason: str | None = None
            if idx >= fold_top_n:
                metrics = _failed_metrics("not_folded")
                forced_reason = "outside_fold_top_n"
            elif pre_ok:
                try:
                    metrics = _fold_and_measure(
                        str(row.sequence),
                        wt_seq,
                        stage_dir,
                        cfg.params,
                        cfg.seed,
                        candidate_id,
                        wt_context,
                    )
                except Exception as exc:  # candidate-level model/schema failures stay auditable
                    if _is_oom_error(exc):
                        raise
                    n_fold_failed += 1
                    metrics = _failed_metrics("api_failed")
                    metrics["fold_reason_extra"] = f"{type(exc).__name__}:{str(exc)[:160]}"
            else:
                n_precheck_failed += 1
                forced_reason = ";".join(pre_reasons) or "precheck_failed"

            if metrics["fold_mode"] == "api_failed" and metrics.get("fold_reason_extra"):
                forced_reason = str(metrics["fold_reason_extra"])

            if forced_reason is None:
                decision = decide_fold_pass(
                    precheck_reasons=pre_reasons,
                    contact_fraction=float(metrics["fold_barrel_contact_frac"]),
                    pocket_max_delta=float(metrics["fold_pocket_max_delta_a"]),
                    pocket_pae=float(metrics["fold_pae_pocket"]),
                    pocket_plddt=float(metrics["fold_plddt_pocket"]),
                    thresholds=cfg.params,
                    coarse_pocket=not bool(metrics.get("fold_used_cro", True)),
                )
                reason = decision.reason
                passed = bool(decision.passed)
            else:
                reason = forced_reason
                passed = False
            if metrics.get("fold_reason_extra"):
                extra = str(metrics["fold_reason_extra"])
                if reason != extra:
                    reason = extra if reason == "ok" else f"{reason};{extra}"
            out_row = row._asdict()
            out_row.update({k: v for k, v in metrics.items() if k != "fold_reason_extra"})
            out_row["fold_pass"] = passed
            out_row["fold_reason"] = reason
            rows.append(out_row)
            n_pass += int(passed)
            metrics_fh.write(json.dumps(out_row, default=_json_default, sort_keys=True) + "\n")

    out = type(df)(rows)
    out_rel = "candidates_folded.parquet"
    write_candidates(out, stage_dir / out_rel)
    # Hand-off integrity: the parquet fold_pass the portfolio consumes must match
    # decide_fold_pass(persisted geometry). Re-read and audit so a write-path /
    # serialization defect that decouples the verdict from its geometry fails
    # loud here instead of silently mis-selecting downstream.
    _assert_fold_pass_matches_geometry(read_candidates(stage_dir / out_rel), cfg.params)

    return StageResult(
        outputs={"candidates_folded": out_rel},
        decisions=[
            Decision(
                name="fold_sanity",
                threshold="negative ESMFold2 sanity filter",
                kept=n_pass,
                dropped=len(out) - n_pass,
                note=f"flags only; portfolio consumes fold_pass; baseline={baseline_mode}",
            )
        ],
        metrics={
            "n_in": int(len(df)),
            "n_out": int(len(out)),
            "n_fold_pass": int(n_pass),
            "n_precheck_failed": int(n_precheck_failed),
            "n_fold_failed": int(n_fold_failed),
            "baseline_mode": baseline_mode,
        },
    )


_FOLDED_MODES = frozenset({"cro", "protein_only_fallback"})


def _assert_fold_pass_matches_geometry(df: Any, params: dict[str, Any]) -> None:
    """Fail loud if any folded row's fold_pass disagrees with its own geometry.

    For folded rows (precheck passed, model ran) fold_pass must equal
    decide_fold_pass(stored geometry, thresholds). Forced rows (precheck/api/
    not-folded) carry sentinel geometry and a forced verdict, so they are
    skipped. A mismatch means the verdict column was decoupled from the geometry
    (stale/overwritten/serialization defect) — exactly the corruption that fed
    the portfolio a relaxed fold_pass in run smoke_full_20260630_0254.
    """
    mismatches: list[str] = []
    for row in df.itertuples(index=False):
        if str(getattr(row, "fold_mode", "")) not in _FOLDED_MODES:
            continue
        decision = decide_fold_pass(
            precheck_reasons=[],
            contact_fraction=float(row.fold_barrel_contact_frac),
            pocket_max_delta=float(row.fold_pocket_max_delta_a),
            pocket_pae=float(row.fold_pae_pocket),
            pocket_plddt=float(row.fold_plddt_pocket),
            thresholds=params,
            coarse_pocket=not bool(row.fold_used_cro),
        )
        if bool(decision.passed) != bool(row.fold_pass):
            mismatches.append(
                f"{row.id}: stored fold_pass={bool(row.fold_pass)} but "
                f"geometry implies {bool(decision.passed)} ({decision.reason})"
            )
    if mismatches:
        raise ValueError(
            "candidates_folded.parquet fold_pass is inconsistent with its "
            "geometry for %d row(s): %s" % (len(mismatches), "; ".join(mismatches[:10]))
        )


def _failed_metrics(mode: str) -> dict[str, Any]:
    return {
        "fold_mode": mode,
        "fold_used_cro": False,
        "fold_plddt_mean": float("nan"),
        "fold_plddt_pocket": float("nan"),
        "fold_pae_pocket": float("nan"),
        "fold_barrel_contact_frac": 0.0,
        "fold_pocket_max_delta_a": float("inf"),
        "fold_artifact": "",
    }


def _build_wt_context(wt_atoms: list[AtomRecord], params: dict[str, Any]) -> dict[str, Any]:
    pocket_distances, _ = pocket_distances_to_chromophore(
        wt_atoms,
        pocket_positions=params["pocket_distance_positions"],
        chromophore_positions=[65, 66, 67],
        exclude_backbone_left=True,
    )
    return {
        "contacts": ca_contact_pairs(
            wt_atoms,
            min_seq_sep=int(params["barrel_contact_min_seq_sep"]),
            cutoff=float(params["barrel_contact_cutoff_a"]),
        ),
        "pocket_distances": pocket_distances,
    }


def _select_wt_baseline(
    wt_seq: str,
    crystal_atoms: list[AtomRecord],
    params: dict[str, Any],
    seed: int,
) -> tuple[list[AtomRecord], str]:
    """Pick the WT geometry baseline for candidate deltas.

    Candidate distances/contacts come from an ESMFold2 prediction, so a crystal
    baseline leaves an uncancelled prediction-vs-crystal offset on the pocket
    delta gate. When `calibrate_with_wt_fold` is set we fold WT through the same
    path and use that predicted geometry as the baseline (deltas then live in one
    prediction domain), and assert WT passes its own absolute gates. WT-fold
    failure aborts the stage when `require_wt_fold` is set (spec §8); otherwise we
    fall back to the crystal geometry.
    """
    if not bool(params.get("calibrate_with_wt_fold", True)):
        return crystal_atoms, "crystal"
    try:
        ref = _fold_wt_reference(wt_seq, params, seed)
        wt_calibration_check(ref["pocket_plddt"], ref["pocket_pae"], params)
        return ref["atoms"], "predicted_wt_fold"
    except Exception as exc:
        if _is_oom_error(exc):
            raise
        if bool(params.get("require_wt_fold", True)):
            raise
        return crystal_atoms, "crystal"


def wt_calibration_check(pocket_plddt: float, pocket_pae: float, params: dict[str, Any]) -> None:
    """Fail loud if the WT fold cannot clear its own absolute pocket gates.

    With WT folded under the same parameters as candidates, the pLDDT/pAE gates
    are calibrated against the prediction domain. If WT itself fails them, the
    thresholds are miscalibrated and every candidate veto would be untrustworthy.
    """
    decision = decide_fold_pass(
        precheck_reasons=[],
        contact_fraction=1.0,
        pocket_max_delta=0.0,
        pocket_pae=pocket_pae,
        pocket_plddt=pocket_plddt,
        thresholds=params,
    )
    if not decision.passed:
        raise ValueError(
            f"WT ESMFold2 fold fails its own fold_sanity gates ({decision.reason}); "
            "pocket pLDDT/pAE thresholds are miscalibrated for the prediction domain"
        )


def _fold_wt_reference(wt_sequence: str, params: dict[str, Any], seed: int) -> dict[str, Any]:
    result, _ = _fold_sequence_with_fallback(wt_sequence, params, seed, "wt_reference")
    adapted = _adapt_fold_result(result, wt_sequence)
    atoms_by_residue = _records_by_residue(adapted["raw_atoms"])
    atoms = [atom for records in atoms_by_residue.values() for atom in records]
    return {
        "atoms": atoms,
        "pocket_plddt": _mean_by_positions(adapted["plddt_by_resid"], params["pocket_positions"]),
        "pocket_pae": _pocket_pae(adapted["pae_by_resid"], params["pocket_positions"]),
        "used_cro": any(a.resname == "CRO" for a in atoms),
    }


def _fold_and_measure(
    sequence: str,
    wt_sequence: str,
    stage_dir: Path,
    params: dict[str, Any],
    seed: int,
    candidate_id: str,
    wt_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del wt_sequence
    result, mode = _fold_sequence_with_fallback(sequence, params, seed, candidate_id)
    adapted = _adapt_fold_result(result, sequence)
    atoms_by_residue = _records_by_residue(adapted["raw_atoms"])
    atoms = [atom for records in atoms_by_residue.values() for atom in records]
    context = wt_context
    if context is None:
        raise ValueError("wt_context is required for fold_sanity metrics")

    contact_frac = contact_retention_fraction(
        context["contacts"],
        ca_coordinates(atoms),
        cutoff=float(params["barrel_contact_cutoff_a"]),
        slack=float(params.get("barrel_contact_slack_a", 2.0)),
    )
    candidate_distances, used_cro = pocket_distances_to_chromophore(
        atoms,
        pocket_positions=params["pocket_distance_positions"],
        chromophore_positions=[65, 66, 67],
        exclude_backbone_left=True,
    )
    deltas = pocket_distance_deltas(context["pocket_distances"], candidate_distances)
    pocket_max_delta = max((v for v in deltas.values() if np.isfinite(v)), default=float("nan"))
    pocket_plddt = _mean_by_positions(adapted["plddt_by_resid"], params["pocket_positions"])
    pocket_pae = _pocket_pae(adapted["pae_by_resid"], params["pocket_positions"])
    artifact = _write_structure_artifact(stage_dir, candidate_id, result, params)

    return {
        "fold_mode": mode,
        "fold_used_cro": used_cro,
        "fold_reason_extra": "" if used_cro else "coarse_pocket_proxy",
        "fold_plddt_mean": _safe_float(np.nanmean(list(adapted["plddt_by_resid"].values()))),
        "fold_plddt_pocket": pocket_plddt,
        "fold_pae_pocket": pocket_pae,
        "fold_barrel_contact_frac": contact_frac,
        "fold_pocket_max_delta_a": pocket_max_delta,
        "fold_artifact": artifact,
    }


_MODEL_CACHE: dict[tuple[str, str], Any] = {}


def _build_esmfold2(model_id: str, torch_dtype: str):
    import torch
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    dtype = getattr(torch, torch_dtype)
    return ESMFold2Model.from_pretrained(model_id, torch_dtype=dtype).cuda().eval()


def _load_esmfold2(model_id: str, torch_dtype: str):
    key = (str(model_id), str(torch_dtype))
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = _build_esmfold2(str(model_id), str(torch_dtype))
    return _MODEL_CACHE[key]


def _fold_sequence_with_fallback(sequence: str, params: dict[str, Any], seed: int, candidate_id: str):
    if bool(params.get("try_cro", True)):
        try:
            return _fold_sequence(sequence, params, seed, candidate_id, use_cro=True), "cro"
        except Exception as exc:
            if _is_oom_error(exc):
                raise
            if not bool(params.get("fallback_without_cro", True)):
                raise
    return _fold_sequence(sequence, params, seed, candidate_id, use_cro=False), "protein_only_fallback"


def _is_oom_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda oom" in msg


def _fold_sequence(
    sequence: str,
    params: dict[str, Any],
    seed: int,
    candidate_id: str,
    *,
    use_cro: bool,
):
    import torch
    from esm.models.esmfold2 import ESMFold2InputBuilder, Modification, ProteinInput
    from esm.models.esmfold2 import StructurePredictionInput

    model = _load_esmfold2(str(params["model_id"]), str(params.get("torch_dtype", "float16")))
    modifications = None
    if use_cro:
        modifications = [Modification(position=int(params.get("cro_position", 65)), ccd="CRO")]
    protein = ProteinInput(id="A", sequence=sequence, modifications=modifications)
    spi = StructurePredictionInput(sequences=[protein])
    with torch.no_grad():
        return ESMFold2InputBuilder().fold(
            model,
            spi,
            num_loops=int(params["num_loops"]),
            num_sampling_steps=int(params["num_sampling_steps"]),
            num_diffusion_samples=int(params["num_diffusion_samples"]),
            seed=int(seed),
            complex_id=candidate_id,
        )


def _write_structure_artifact(stage_dir: Path, candidate_id: str, result, params: dict[str, Any]) -> str:
    if not bool(params.get("write_structures", True)):
        return ""
    folds = stage_dir / "folds"
    folds.mkdir(parents=True, exist_ok=True)
    safe_id = "".join(c if c.isalnum() or c in "._-" else "_" for c in candidate_id)
    rel = Path("folds") / f"{safe_id}.cif"
    (stage_dir / rel).write_text(result.complex.to_mmcif())
    return rel.as_posix()


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


if __name__ == "__main__":
    cli()
