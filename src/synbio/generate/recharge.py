"""Shared surface-recharge enumeration (TGP-style K/R -> E/Q, lower pI).

Reads the committed SASA constant; exposure is rel.SASA > threshold, gated by the
tolerance map and do_not_mutate. The 72C readout is dominated by irreversible
aggregation; lowering surface charge is the carrying thermostability lever.
"""

import json
from pathlib import Path

from .apply import Mutation, gate_single

__all__ = [
    "load_sasa",
    "exposed_positions",
    "net_charge_delta",
    "recharge_singles",
]

CHARGE: dict[str, int] = {"K": 1, "R": 1, "D": -1, "E": -1}


def load_sasa(path: str | Path) -> dict[int, float]:
    """Load {1-based residue -> relative SASA} from a sfgfp_sasa.json file."""
    data = json.loads(Path(path).read_text())
    return {int(k): float(v) for k, v in data["rel_sasa"].items()}


def exposed_positions(sasa: dict[int, float], threshold: float) -> set[int]:
    """Positions whose relative SASA exceeds the threshold."""
    return {pos for pos, rel in sasa.items() if rel > threshold}


def net_charge_delta(mut: Mutation) -> int:
    """Charge(target) - charge(wt) for a recharge mutation (e.g. K->E = -2)."""
    return CHARGE.get(mut.target, 0) - CHARGE.get(mut.wt, 0)


def recharge_singles(
    wt_seq: str,
    sasa: dict[int, float],
    free_mutate: set[int],
    do_not_mutate: set[int],
    *,
    threshold: float,
    targets: dict[str, list[str]],
    min_pos: int = 11,
) -> list[Mutation]:
    """Exposed K/R at mutable positions -> one Mutation per allowed target residue."""
    exposed = exposed_positions(sasa, threshold)
    out: list[Mutation] = []
    for pos in sorted(exposed):
        wt = wt_seq[pos - 1]
        if wt not in targets:
            continue
        for target in targets[wt]:
            if gate_single(pos, target, wt_seq, free_mutate, do_not_mutate, min_pos):
                out.append(Mutation(pos=pos, wt=wt, target=target))
    return out
