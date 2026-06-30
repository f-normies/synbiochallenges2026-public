"""Stage 05: four-vote ΔΔG scoring, aggregated by rank in stage 06.

Three structure-based votes on the 2B3P backbone (ThermoMPNN-D / SPURS / ProteinMPNN-ddG — Ч4)
+ one sequence-only ESMC-6B vote (`run_esmc` — Ч3). The ESMC vote is decorrelated from the three
structure votes (no 2B3P dependence) and is the load-bearing voice for the far upside slot.

**ddg artifact contract** (every vote emits this; `rank_combine` joins them on `id`): a parquet
with columns ``["id", "ddg"]``, one row per candidate, ``ddg`` = ΔΔG_funnel vs sfGFP in the funnel
convention **positive = destabilizing** (so lower ranks better, consistent across all four votes).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from synbio.orchestrator import register_stage
from synbio.orchestrator.stage import Decision, StageConfig, StageResult, StageSpec, cli
from synbio.stability.structure_votes import (
    ensure_thermompnn_ssm_csv,
    load_structure_context,
    model_sequence,
    plan_candidates,
    predict_spurs_multi,
    read_thermompnn_single_csv,
    run_proteinmpnn_score_only,
    score_from_single_mutation_lookup,
    structure_vote_frame,
)

__all__ = [
    "SPEC_THERMOMPNN",
    "SPEC_SPURS",
    "SPEC_PROTEINMPNN",
    "SPEC_ESMC",
    "ddg_funnel_vs_wt",
    "run_thermompnn",
    "run_spurs",
    "run_proteinmpnn",
    "run_esmc",
]

SPEC_THERMOMPNN = StageSpec(
    name="score_stability.thermompnn",
    module="score_stability",
    env="thermompnn",
    inputs=("candidates_bright", "wt"),
    outputs=("ddg_thermompnn",),
)
SPEC_SPURS = StageSpec(
    name="score_stability.spurs",
    module="score_stability",
    env="spurs",
    inputs=("candidates_bright", "wt"),
    outputs=("ddg_spurs",),
)
SPEC_PROTEINMPNN = StageSpec(
    name="score_stability.proteinmpnn",
    module="score_stability",
    env="proteinmpnn",
    inputs=("candidates_bright", "wt"),
    outputs=("ddg_proteinmpnn",),
)
SPEC_ESMC = StageSpec(
    name="score_stability.esmc",
    module="score_stability",
    env="esm",
    inputs=("candidates_bright", "stability_probe", "wt"),
    outputs=("ddg_esmc",),
)


@register_stage(SPEC_THERMOMPNN)
def run_thermompnn(cfg: StageConfig) -> StageResult:
    """ThermoMPNN-D vote: WT single-mutant SSM summed over candidate substitutions."""
    from synbio.io.artifacts import read_candidates

    p = cfg.params
    out_dir = Path(cfg.stage_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = read_candidates(cfg.inputs["candidates_bright"]).reset_index(drop=True)
    ctx = _structure_context(cfg)
    plans = plan_candidates(df, ctx)

    ssm_csv = ensure_thermompnn_ssm_csv(ctx.pdb_path, out_dir, p)
    lookup = read_thermompnn_single_csv(ssm_csv)
    scores, status = score_from_single_mutation_lookup(plans, lookup)
    vote = structure_vote_frame(df, plans, scores, score_status=status)
    vote.to_parquet(out_dir / "ddg_thermompnn.parquet", index=False)
    return _structure_vote_result(
        "ddg_thermompnn", vote, "ThermoMPNN-D single-SSM additive vote"
    )


@register_stage(SPEC_SPURS)
def run_spurs(cfg: StageConfig) -> StageResult:
    """SPURS multi-mutant vote on the WT PDB."""
    from synbio.io.artifacts import read_candidates

    out_dir = Path(cfg.stage_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = read_candidates(cfg.inputs["candidates_bright"]).reset_index(drop=True)
    ctx = _structure_context(cfg)
    plans = plan_candidates(df, ctx)

    ids_to_score: list[object] = []
    mutation_lists = []
    scores: dict[object, float] = {}
    status: dict[object, str] = {}
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
        ids_to_score.append(cid)
        mutation_lists.append(plan.model_mutations)

    if mutation_lists:
        ddg = predict_spurs_multi(ctx.pdb_path, mutation_lists, cfg.params)
        for cid, value in zip(ids_to_score, ddg, strict=True):
            scores[cid] = float(value)
            status[cid] = "ok"

    vote = structure_vote_frame(df, plans, scores, score_status=status)
    vote.to_parquet(out_dir / "ddg_spurs.parquet", index=False)
    return _structure_vote_result("ddg_spurs", vote, "SPURS multi-mutant structure vote")


@register_stage(SPEC_PROTEINMPNN)
def run_proteinmpnn(cfg: StageConfig) -> StageResult:
    """ProteinMPNN-ddG vote: score-only global-score delta vs WT."""
    from synbio.io.artifacts import read_candidates

    out_dir = Path(cfg.stage_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = read_candidates(cfg.inputs["candidates_bright"]).reset_index(drop=True)
    ctx = _structure_context(cfg)
    plans = plan_candidates(df, ctx)

    records: list[tuple[str, str]] = []
    header_to_id: dict[str, object] = {}
    scores: dict[object, float] = {}
    status: dict[object, str] = {}
    for i, row in enumerate(df[["id", "sequence"]].itertuples(index=False), start=1):
        plan = plans[row.id]
        if not plan.gate_pass:
            continue
        if not plan.scoreable:
            status[row.id] = f"not_scoreable:{plan.coverage_reason}"  # defer to ESMC, do not score
            continue
        if not plan.mutations:
            scores[row.id] = 0.0
            status[row.id] = "wt_or_noop"
            continue
        header = f"cand_{i:06d}"
        header_to_id[header] = row.id
        records.append((header, model_sequence(row.sequence, ctx)))

    if records:
        raw_scores = run_proteinmpnn_score_only(ctx.pdb_path, records, out_dir, cfg.params, cfg.seed)
        wt_score = float(raw_scores["__wt__"])
        for header, cid in header_to_id.items():
            if header in raw_scores:
                scores[cid] = float(raw_scores[header]) - wt_score
                status[cid] = "ok"
            else:
                status[cid] = f"score_missing:{header}"

    vote = structure_vote_frame(df, plans, scores, score_status=status)
    vote.to_parquet(out_dir / "ddg_proteinmpnn.parquet", index=False)
    return _structure_vote_result(
        "ddg_proteinmpnn", vote, "ProteinMPNN score-only global-score delta"
    )


def _structure_context(cfg: StageConfig):
    p = cfg.params
    return load_structure_context(
        cfg.inputs["wt"],
        chain=str(p.get("pdb_chain", "A")),
        gate_tiers=tuple(p.get("gate_tiers", ("hard", "high_risk"))),
    )


def _structure_vote_result(output_key: str, vote, note: str) -> StageResult:
    n = int(len(vote))
    gate_pass = vote["gate_pass"].astype(bool).to_numpy()
    scoreable = vote["scoreable"].astype(bool).to_numpy()
    finite = np.isfinite(vote["ddg"].to_numpy(dtype=float))
    n_gate = int((~gate_pass).sum())                          # vetoed (catalytic/chromophore) -> dropped in 06
    n_not_scoreable = int((gate_pass & ~scoreable).sum())     # out of WT-structure coverage -> deferred to ESMC
    n_scored = int(finite.sum())
    n_missing = int((gate_pass & scoreable & ~finite).sum())  # in coverage but no model score (investigate!)
    median = float(np.median(vote.loc[finite, "ddg"])) if n_scored else float("nan")
    return StageResult(
        outputs={output_key: f"{output_key}.parquet"},
        decisions=[
            Decision(
                name=output_key,
                kept=n_scored,
                dropped=n_gate,
                threshold="WT-only structure veto + ddG vote",
                note=f"{note}; {n_gate}/{n} vetoed, {n_not_scoreable} deferred-to-ESMC "
                     f"(out of WT-structure coverage), {n_missing} score-missing",
            ),
        ],
        metrics={
            "n": n,
            "n_scored": n_scored,
            "n_gate_failed": n_gate,
            "n_not_scoreable": n_not_scoreable,
            "n_score_missing": n_missing,
            "median_ddg": median,
        },
    )


def ddg_funnel_vs_wt(dg_wt: float, dg_mut: np.ndarray) -> np.ndarray:
    """ΔΔG_funnel = dG_pred(wt) − dG_pred(mut), positive = destabilizing — pure numpy.

    The stage-02 probe predicts absolute folding stability dG (higher = more stable); subtracting
    the mutant's dG from sfGFP's gives the funnel ΔΔG in the shared convention (see contract above).
    """
    return float(dg_wt) - np.asarray(dg_mut, dtype=float)


@register_stage(SPEC_ESMC)
def run_esmc(cfg: StageConfig) -> StageResult:
    """Sequence-only ESMC-6B ΔΔG vote (Ч3): frozen stability probe, no 2B3P backbone."""
    from pathlib import Path

    import pandas as pd

    from synbio.esmc import EmbeddingCache, embed_sequences, load_esmc
    from synbio.io.artifacts import read_candidates
    from synbio.probes import RidgeProbe

    p = cfg.params
    out_dir = Path(cfg.stage_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_candidates(cfg.inputs["candidates_bright"]).reset_index(drop=True)

    probe_dir = Path(cfg.inputs["stability_probe"])
    probe = RidgeProbe.load(probe_dir / "dir.npz")
    meta = json.loads((probe_dir / "meta.json").read_text())
    layer, pool = int(meta["layer"]), meta["pool"]

    wt_seq = json.loads((Path(cfg.inputs["wt"]) / "wt.json").read_text())["sequence"]

    handle = load_esmc(p["esmc_6b_model_id"])
    cache = EmbeddingCache(p["embed_cache_dir"], model_tag="esmc6b", layer=layer, pool=pool)
    emb = embed_sequences(handle, [wt_seq] + df["sequence"].tolist(), layer,
                          int(p["batch_size"]), cache, pool)
    dg = probe.predict(emb)
    ddg = ddg_funnel_vs_wt(float(dg[0]), dg[1:])

    out = pd.DataFrame({"id": df["id"].to_numpy(), "ddg": ddg})
    out.to_parquet(out_dir / "ddg_esmc.parquet", index=False)

    return StageResult(
        outputs={"ddg_esmc": "ddg_esmc.parquet"},
        decisions=[Decision(name="ddg_esmc", note=f"L{layer}/{pool} sequence-only vote, "
                            f"n={len(df)}, median ΔΔG={float(np.median(ddg)):.2f}")],
        metrics={"n": int(len(df)), "median_ddg": float(np.median(ddg)),
                 "frac_destabilizing": float(np.mean(ddg > 0))},
    )


if __name__ == "__main__":
    cli()
