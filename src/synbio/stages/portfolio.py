"""Stage 08: diversified 6-sequence portfolio selection + submission.csv (owned by team)."""

import logging
from pathlib import Path

from synbio.orchestrator import register_stage
from synbio.orchestrator.stage import Decision, StageConfig, StageResult, StageSpec, cli

logger = logging.getLogger(__name__)

SPEC = StageSpec(
    name="portfolio",
    module="portfolio",
    env="dnatools",
    inputs=("candidates_folded", "exclusion_list"),
    outputs=("portfolio6", "submission"),
)


@register_stage(SPEC)
def run(cfg: StageConfig) -> StageResult:
    from synbio.io.artifacts import read_candidates
    from synbio.portfolio import (
        annotate_eligibility,
        load_exclusion_set,
        select_portfolio,
        write_submission,
    )

    stage_dir = Path(cfg.stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    p = cfg.params

    df = read_candidates(cfg.inputs["candidates_folded"])
    exclusion = load_exclusion_set(cfg.inputs["exclusion_list"])
    annotated = annotate_eligibility(df, exclusion, float(p.get("min_brightness_margin", 0.6)))

    result = select_portfolio(
        annotated,
        n_select=int(p.get("n_select", 6)),
        max_per_pair=int(p.get("max_per_pair", 2)),
        upside_sources=tuple(p.get("upside_sources", ("sampler",))),
    )

    portfolio6_rel = "portfolio6.parquet"
    submission_rel = "submission.csv"
    result.annotated.to_parquet(stage_dir / portfolio6_rel, index=False)
    write_submission(result.selected, stage_dir / submission_rel, team_name=str(p.get("team_name", "DOGMA")))

    n_eligible = int(annotated["eligible"].sum())
    n_selected = int(len(result.selected))
    n_select_target = int(p.get("n_select", 6))
    if n_selected < n_select_target:
        logger.warning(
            "portfolio selected only %d of %d sequences — submission will be short",
            n_selected,
            n_select_target,
        )
    decisions = [
        Decision(
            name="eligibility",
            threshold=f"b_hat>={p.get('min_brightness_margin', 0.6)} & constraints & not excluded",
            kept=n_eligible,
            dropped=int(len(annotated) - n_eligible),
            note="hard admission gate",
        ),
        Decision(
            name="portfolio",
            threshold="6 slots: 2 special noms + upside + 3 core",
            kept=n_selected,
            dropped=int(n_eligible - n_selected),
            note="; ".join(result.relaxations) or "all §8.3 targets met",
        ),
    ]
    return StageResult(
        outputs={"portfolio6": portfolio6_rel, "submission": submission_rel},
        decisions=decisions,
        metrics={
            "n_in": int(len(df)),
            "n_eligible": n_eligible,
            "n_selected": n_selected,
            "n_strategies": int(result.selected["source"].nunique()) if n_selected else 0,
            "baskets_covered": int(result.selected["bucket"].nunique()) if n_selected else 0,
            "relaxations": result.relaxations,
        },
    )


if __name__ == "__main__":
    cli()
