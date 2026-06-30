"""Stage 06: rank aggregation R = B̂ × σ_стаб (Ч3).

Funnel target is `F_final` (§2.3), which telescopes to absolute residual brightness. We rank by
``R = B̂ × σ_стаб`` where:
  - **B̂** = calibrated predicted brightness in ×WT (magnitude preserved — the brightness axis is
    in-distribution / calibrated, carried in from stage 04 as `b_hat`);
  - **σ_стаб** = stability rank-percentile in [0, 1] (order only — the 72 °C axis is OOD, so we
    trust the *rank* of the ΔΔG ensemble, never its absolute numbers).

σ_стаб averages each candidate's per-vote rank across the four ΔΔG votes (stages 05 ThermoMPNN-D /
SPURS / ProteinMPNN-ddG + the sequence-only ESMC-6B vote), then maps that mean rank to a percentile
in [0, 1] (lowest ΔΔG_funnel = most stable = percentile 1.0) — order only, no magnitude. Ties on R
break toward the recharge prior (more acidic surface = lower net charge), then `id` for determinism.
Pure CPU (env `dnatools`); fully .venv-testable.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from synbio.orchestrator import register_stage
from synbio.orchestrator.stage import Decision, StageConfig, StageResult, StageSpec, cli

logger = logging.getLogger(__name__)

__all__ = [
    "SPEC",
    "DDG_INPUTS",
    "average_rank",
    "stability_percentile",
    "combine_sigma",
    "net_charge",
    "rank_candidates",
    "run",
]

# The four ΔΔG vote artifacts, in input order; each is a parquet with columns ["id", "ddg"].
DDG_INPUTS: tuple[str, ...] = ("ddg_thermompnn", "ddg_spurs", "ddg_proteinmpnn", "ddg_esmc")

SPEC = StageSpec(
    name="rank_combine",
    module="rank_combine",
    env="dnatools",
    inputs=("candidates_bright", *DDG_INPUTS),
    outputs=("candidates_ranked",),
)


def average_rank(values: np.ndarray) -> np.ndarray:
    """Average (tie-corrected) ranks, 1..N ascending — pure numpy `scipy.stats.rankdata` equivalent."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0  # mean of the 1-based tied positions
        i = j + 1
    return ranks


def stability_percentile(ddg: np.ndarray) -> np.ndarray:
    """Map ΔΔG_funnel (positive = destabilizing) to a stability percentile in [0, 1].

    Lowest ΔΔG (most stable) → 1.0, highest → 0.0. Single candidate → 0.5 (no order information).
    NaN ΔΔG → NaN percentile (excluded downstream when averaging votes).
    """
    ddg = np.asarray(ddg, dtype=float)
    out = np.full(len(ddg), np.nan)
    finite = np.isfinite(ddg)
    n = int(finite.sum())
    if n == 0:
        return out
    if n == 1:
        out[finite] = 0.5
        return out
    # rank ascending by ddg, then invert so low ddg = high percentile
    r = average_rank(ddg[finite])
    out[finite] = (n - r) / (n - 1)
    return out


def combine_sigma(ddg_matrix: np.ndarray) -> np.ndarray:
    """σ_стаб per candidate: average each candidate's per-vote rank, then percentile the mean rank.

    Per PROJECT_PLAN §6.4/§7-06 ("усреднение рангов четырёх голосов → перцентиль"): rank candidates
    within each ΔΔG vote (ascending ΔΔG_funnel → rank 1 = most stable), average those ranks over the
    votes a candidate is scored by (NaN votes ignored), then map the mean rank to a stability
    percentile in [0, 1] (lowest mean rank = most stable = 1.0). This keeps σ a pure-*order* quantity
    (the 72 °C axis is OOD); averaging per-vote percentiles instead leaks each vote's rank-spacing
    into σ's value, and since R = B̂ × σ multiplies σ's magnitude, that difference reorders the top-N.

    `ddg_matrix` is [N, n_votes] of ΔΔG_funnel. A row with every vote NaN yields NaN (no signal).
    """
    ddg_matrix = np.asarray(ddg_matrix, dtype=float)
    n, n_votes = ddg_matrix.shape
    ranks = np.full((n, n_votes), np.nan)
    for k in range(n_votes):
        col = ddg_matrix[:, k]
        finite = np.isfinite(col)
        if finite.any():
            ranks[finite, k] = average_rank(col[finite])  # 1 = lowest ΔΔG = most stable
    finite = np.isfinite(ranks)
    counts = finite.sum(axis=1)
    totals = np.where(finite, ranks, 0.0).sum(axis=1)
    mean_rank = np.divide(totals, counts, out=np.full(n, np.nan), where=counts > 0)
    return stability_percentile(mean_rank)  # re-rank the mean ranks into [0, 1]


# Charged residues (Henderson-Hasselbalch-free: net charge ≈ basic − acidic at neutral pH).
_BASIC, _ACIDIC = ("K", "R"), ("D", "E")


def net_charge(seq: str) -> int:
    """Approximate net charge at neutral pH: (#K+#R) − (#D+#E). Lower = more acidic surface.

    Used only as the R tie-breaker (recharge / low-pI thermo prior, §4.3); not a scored axis.
    """
    return sum(seq.count(a) for a in _BASIC) - sum(seq.count(a) for a in _ACIDIC)


def rank_candidates(
    cand: pd.DataFrame, ddg: pd.DataFrame, top_n: int
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Join votes onto candidates, compute σ_стаб and R = B̂ × σ_стаб, rank, take top-N.

    `cand` carries `id`, `sequence`, `b_hat` (from stage 04); `ddg` is the candidate table joined
    with the four `ddg_*` columns (one per vote). Returns the ranked top-N (with `sigma_stab`, `R`,
    `net_charge` columns) and a decision-count dict. Pure — no IO, .venv-testable.
    """
    df = cand.merge(ddg, on="id", how="left")
    vote_cols = [c for c in df.columns if c.startswith("ddg_")]
    if not vote_cols:
        raise ValueError("no ddg_* vote columns to aggregate")
    gate_cols = [c for c in df.columns if c.startswith("gate_pass_")]

    df = df.copy()
    df["sigma_stab"] = combine_sigma(df[vote_cols].to_numpy(dtype=float))
    df["R"] = df["b_hat"].to_numpy(dtype=float) * df["sigma_stab"].to_numpy(dtype=float)
    df["net_charge"] = df["sequence"].apply(net_charge)
    # `gate_pass` is the biological *veto* only (catalytic / chromophore positions). Candidates
    # merely out of WT-structure coverage keep gate_pass=True (their structure ddG is NaN) and
    # flow through on whatever votes scored them — e.g. the sequence-only ESMC vote that carries
    # the upside slot. Only a True->False veto drops a candidate here.
    if gate_cols:
        gate = df[gate_cols].fillna(True).astype(bool)
        df["_gate_failed"] = (~gate).any(axis=1).to_numpy(dtype=bool)
    else:
        df["_gate_failed"] = False

    n_gate = int(df["_gate_failed"].sum())
    n_no_stab = int((~df["_gate_failed"] & df["sigma_stab"].isna()).sum())
    ranked = df.loc[~df["_gate_failed"]].dropna(subset=["sigma_stab"]).sort_values(
        ["R", "net_charge", "id"], ascending=[False, True, True], kind="stable"
    ).head(int(top_n)).reset_index(drop=True).drop(columns=["_gate_failed"])

    counts = {
        "n_in": int(len(cand)),
        "n_gate_dropped": n_gate,
        "n_no_stab_dropped": n_no_stab,
        "n_out": int(len(ranked)),
    }
    return ranked, counts


@register_stage(SPEC)
def run(cfg: StageConfig) -> StageResult:
    from synbio.io.artifacts import read_candidates

    p = cfg.params
    out_dir = Path(cfg.stage_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cand = read_candidates(cfg.inputs["candidates_bright"])
    if "b_hat" not in cand.columns:
        raise ValueError("candidates_bright missing 'b_hat' (stage 04 magnitude) for R = B̂ × σ")

    ddg = cand[["id"]].copy()
    for key in DDG_INPUTS:
        raw = pd.read_parquet(cfg.inputs[key])
        v = raw[["id", "ddg"]].rename(columns={"ddg": key})
        if "gate_pass" in raw.columns:
            v[f"gate_pass_{key}"] = raw["gate_pass"]
        ddg = ddg.merge(v, on="id", how="left")

    ranked, counts = rank_candidates(cand, ddg, int(p["top_n"]))
    ranked.to_parquet(out_dir / "candidates_ranked.parquet", index=False)

    return StageResult(
        outputs={"candidates_ranked": "candidates_ranked.parquet"},
        decisions=[
            Decision(name="rank_combine", threshold=f"top {p['top_n']} by R = B̂ × σ_стаб",
                     kept=counts["n_out"], dropped=counts["n_in"] - counts["n_out"],
                     note=f"{counts['n_gate_dropped']} dropped by structure veto (catalytic/chromophore), "
                          f"{counts['n_no_stab_dropped']} dropped for no usable ΔΔG vote; "
                          "tie-break: net charge (recharge prior)"),
        ],
        metrics=counts,
    )


if __name__ == "__main__":
    cli()
