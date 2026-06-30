from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

__all__ = ["SelectionResult", "select_portfolio"]


@dataclass(frozen=True)
class SelectionResult:
    selected: pd.DataFrame
    relaxations: list[str]
    annotated: pd.DataFrame


def select_portfolio(
    df: pd.DataFrame,
    n_select: int = 6,
    max_per_pair: int = 2,
    upside_sources: tuple[str, ...] = ("sampler",),
) -> SelectionResult:
    """Select up to n_select sequences with graded diversification relaxation.

    Every slot prefers fold_pass=True: a fold_pass=False candidate has a
    genuinely catastrophic predicted chromophore pocket (stage-07 negative
    filter), so it cannot fluoresce and must never outrank a clean-fold
    sequence. Failed-fold rows are NOT hard-vetoed — they stay selectable if the
    pool is too small.

    Selection order (deterministic, eligible rows only):
    1. best_brightness: argmax b_hat, then fold_pass DESC, then
       fold_plddt_pocket DESC. b_hat saturates at ~1.0 on the real pool, so the
       clamp is broken by predicted fold quality (a candidate only fluoresces if
       the pocket actually forms) — NOT by sigma_stab, which would degenerate the
       nomination into "top OOD stability rank".
    2. best_thermo: fold_pass DESC, then most-negative net_charge_delta (surface
       recharge / lowered pI is the bearing 72 C lever — PROJECT_PLAN 4.3 — since
       the readout is aggregation-dominated and the dG rank is OOD), then
       sigma_stab DESC as tiebreak.
    3. upside: fold_pass DESC then best R, restricted to upside_sources; if none,
       relaxed → best remaining by fold_pass then R (logged).
    4. core (n_select - 3 slots): sorted by fold_pass DESC then R DESC, subject
       to <= max_per_pair per (source, bucket) pair; if all remaining exceed the
       cap, relax the cap and log.

    Args:
        df: DataFrame annotated by annotate_eligibility; must have columns
            id, sequence, source, bucket, b_hat, sigma_stab, R, fold_pass,
            eligible. Optional ranking columns (net_charge_delta,
            fold_plddt_pocket) default to neutral values when absent.
        n_select: Target number of selected sequences.
        max_per_pair: Maximum candidates per (source, bucket) pair in core.
        upside_sources: Source labels that qualify for the upside slot.

    Returns:
        SelectionResult with selected, relaxations, and annotated DataFrames.
    """
    pool = df[df["eligible"]].copy()
    # Optional ranking columns may be absent in minimal/synthetic pools; default
    # to neutral values so the fold-aware / recharge-aware tiebreaks never raise.
    if "fold_pass" not in pool.columns:
        pool["fold_pass"] = False
    if "net_charge_delta" not in pool.columns:
        pool["net_charge_delta"] = 0
    if "fold_plddt_pocket" not in pool.columns:
        pool["fold_plddt_pocket"] = 0.0
    relaxations: list[str] = []
    chosen: list[tuple[str, str]] = []  # (id, slot_role)
    taken_ids: set[str] = set()
    taken_seqs: set[str] = set()

    def take(row: pd.Series | None, role: str) -> None:
        if row is None:
            return
        taken_ids.add(row["id"])
        taken_seqs.add(row["sequence"])
        chosen.append((row["id"], role))

    def best(
        by: list[str],
        asc: list[bool],
        pred: Callable[[pd.DataFrame], pd.Series] | None = None,
    ) -> pd.Series | None:
        avail = pool[
            ~pool["id"].isin(taken_ids) & ~pool["sequence"].isin(taken_seqs)
        ]
        if pred is not None:
            avail = avail[pred(avail)]
        if avail.empty:
            return None
        return avail.sort_values(by=by, ascending=asc).iloc[0]

    # Slot 1: best brightness — break the b_hat clamp by fold credibility
    take(
        best(["b_hat", "fold_pass", "fold_plddt_pocket"], [False, False, False]),
        "best_brightness",
    )

    # Slot 2: best thermo — clean fold, then surface recharge (lowered pI)
    take(
        best(["fold_pass", "net_charge_delta", "sigma_stab"], [False, True, False]),
        "best_thermo",
    )

    # Slot 3: upside from designated sources; relax if none available
    upside = best(["fold_pass", "R"], [False, False], pred=lambda a: a["source"].isin(upside_sources))
    if upside is None:
        relaxations.append(
            "no upside-source candidate; substituted best remaining by R"
        )
        upside = best(["fold_pass", "R"], [False, False])
    take(upside, "upside")

    # Track per-(source, bucket) counts including the three special slots
    pair_counts: dict[tuple[str, str], int] = {}
    for _id, _role in chosen:
        rows = pool.loc[pool["id"] == _id]
        if rows.empty:
            continue
        r = rows.iloc[0]
        pair = (r["source"], r["bucket"])
        pair_counts[pair] = pair_counts.get(pair, 0) + 1

    # Slots 4–6: core — descending R, fold_pass preferred, diversity enforced
    n_core = n_select - 3
    while len([c for c in chosen if c[1] == "core"]) < n_core and len(chosen) < n_select:
        avail = pool[
            ~pool["id"].isin(taken_ids) & ~pool["sequence"].isin(taken_seqs)
        ].copy()
        if avail.empty:
            break
        avail = avail.sort_values(by=["fold_pass", "R"], ascending=[False, False])
        picked = None
        for _, cand in avail.iterrows():
            pair = (cand["source"], cand["bucket"])
            if pair_counts.get(pair, 0) >= max_per_pair:
                continue
            picked = cand
            break
        if picked is None:
            relaxations.append(
                f"per-pair cap {max_per_pair} relaxed: pool lacks diversity"
            )
            picked = avail.iloc[0]
        pair = (picked["source"], picked["bucket"])
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
        take(picked, "core")

    role_by_id = {i: r for i, r in chosen}
    order = [i for i, _ in chosen]
    sel = pool[pool["id"].isin(order)].copy()
    sel["slot_role"] = sel["id"].map(role_by_id)
    sel = sel.set_index("id").loc[order].reset_index()
    sel["selected"] = True
    sel["seq_id"] = range(1, len(sel) + 1)
    sel["relaxations"] = "; ".join(relaxations)

    annotated = df.copy()
    annotated["selected"] = annotated["id"].isin(order)
    annotated["slot_role"] = annotated["id"].map(role_by_id).fillna("")
    seq_id_map = dict(zip(sel["id"], sel["seq_id"]))
    annotated["seq_id"] = annotated["id"].map(seq_id_map).fillna(0).astype(int)
    annotated["relaxations"] = "; ".join(relaxations)
    return SelectionResult(selected=sel, relaxations=relaxations, annotated=annotated)
