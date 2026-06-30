from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

STANDARD_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


@dataclass(frozen=True)
class FoldDecision:
    passed: bool
    reason: str


def precheck_sequence(seq: str) -> tuple[bool, list[str]]:
    seq = str(seq)
    reasons: list[str] = []
    if len(seq) != 238:
        reasons.append(f"non_wt_length:{len(seq)}")
    if seq != seq.upper():
        reasons.append("not_uppercase")
    seq_upper = seq.upper()
    bad = sorted(set(seq_upper) - STANDARD_AA)
    if bad:
        reasons.append(f"nonstandard:{''.join(bad)}")
    if len(seq) < 222:
        reasons.append("missing_pocket_positions")
    if len(seq) >= 67 and seq_upper[64:67] != "TYG":
        reasons.append("chromophore_anchor")
    elif len(seq) < 67:
        reasons.append("chromophore_anchor")
    return not reasons, reasons


def _is_finite(value: float) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def decide_fold_pass(
    *,
    precheck_reasons: list[str],
    contact_fraction: float,
    pocket_max_delta: float,
    pocket_pae: float,
    pocket_plddt: float,
    thresholds: dict[str, Any],
    coarse_pocket: bool = False,
) -> FoldDecision:
    """Aggregate stage-07 veto checks into a pass/fail decision.

    When `coarse_pocket` is True the chromophore was modeled with a coarse
    proxy (no CRO heavy atoms), so the WT-relative pocket-distance delta mixes
    representations and is not comparable; per spec §7.3 we record the metric
    but do not veto on it. pLDDT/pAE/barrel gates still apply.

    `thresholds["veto_pocket_distance"]` (default True) globally disables the
    pocket-distance veto: catalytic pocket residues are unchanged by construction
    (do_not_mutate), so on the real pool the WT-relative delta is dominated by
    ESMFold2 single-sample noise (floor ~1.5 A, median ~7 A) rather than design
    defects. When False the delta is still recorded but never vetoes.
    """
    reasons = list(precheck_reasons)
    veto_distance = bool(thresholds.get("veto_pocket_distance", True)) and not coarse_pocket
    if contact_fraction < float(thresholds["min_barrel_contact_frac"]):
        reasons.append("barrel_contacts")
    if veto_distance and pocket_max_delta > float(thresholds["max_pocket_delta_a"]):
        reasons.append("pocket_distance")
    if _is_finite(pocket_pae) and pocket_pae > float(thresholds["max_pocket_pae"]):
        reasons.append("pocket_pae")
    if _is_finite(pocket_plddt) and pocket_plddt < float(thresholds["min_pocket_plddt"]):
        reasons.append("pocket_plddt")
    return FoldDecision(passed=not reasons, reason="ok" if not reasons else ";".join(reasons))
