"""Brightness and consensus mutation pools for the combinatorial generator.

Brightness has two tiers: 'proven' (previous-year winners mined vs sfGFP, plus the
rare strictly-enhancing DMS singles) and 'neutral_or_better' (DMS singles that do not
hurt brightness). Consensus uses amacGFP (1:1 with sfGFP, ~83% id). All singles are
sfGFP-relative and pass the common gate.
"""

import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .apply import Mutation, gate_single

__all__ = [
    "BrightnessPools",
    "read_fasta_records",
    "brightness_pools",
    "consensus_pool",
]

_MUT = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYX](\d+)([ACDEFGHIKLMNPQRSTVWY])$")


@dataclass(frozen=True)
class BrightnessPools:
    proven: tuple[Mutation, ...]
    neutral_or_better: tuple[Mutation, ...]


def read_fasta_records(path: str | Path) -> dict[str, str]:
    """Parse a (multi-record) FASTA into {record_id: sequence}."""
    records: dict[str, str] = {}
    name: str | None = None
    buf: list[str] = []
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            if name is not None:
                records[name] = "".join(buf)
            name = line[1:].strip().split()[0]
            buf = []
        elif line.strip():
            buf.append(line.strip())
    if name is not None:
        records[name] = "".join(buf)
    return records


def consensus_pool(
    wt_seq: str,
    amac_seq: str,
    free_mutate: set[int],
    do_not_mutate: set[int],
    *,
    min_pos: int = 11,
) -> list[Mutation]:
    """amacGFP (1:1) consensus: positions where amacGFP differs from sfGFP, gated."""
    if len(amac_seq) != len(wt_seq):
        raise ValueError(f"amacGFP len {len(amac_seq)} != sfGFP {len(wt_seq)}")
    out: list[Mutation] = []
    for i, (w, a) in enumerate(zip(wt_seq, amac_seq), start=1):
        if a != w and gate_single(i, a, wt_seq, free_mutate, do_not_mutate, min_pos):
            out.append(Mutation(pos=i, wt=w, target=a))
    return out


def _winner_mutations(
    wt_seq: str,
    winners_path: str | Path,
    free_mutate: set[int],
    do_not_mutate: set[int],
    min_pos: int,
) -> dict[str, Mutation]:
    """Distinct gated singles carried by previous-year winners vs sfGFP (token-keyed)."""
    out: dict[str, Mutation] = {}
    with open(winners_path, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            seq = row["sequence"].strip()
            if len(seq) != len(wt_seq):
                continue
            for i, (w, s) in enumerate(zip(wt_seq, seq), start=1):
                if s != w and gate_single(i, s, wt_seq, free_mutate, do_not_mutate, min_pos):
                    mut = Mutation(pos=i, wt=w, target=s)
                    out[mut.token] = mut
    return out


def _dms_rel_brightness(
    dms_path: str | Path,
    length: int,
    types: tuple[str, ...],
) -> dict[tuple[int, str], float]:
    """Max relative brightness per (pos, target): max_bright(type) - typeWT(type)."""
    wt_bright: dict[str, list[float]] = defaultdict(list)
    subs: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    with open(dms_path, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            gtype = row["GFP type"].strip()
            if gtype not in types:
                continue
            try:
                brightness = float(row["Brightness"])
            except (ValueError, KeyError):
                continue
            mut = row["aaMutations"].strip()
            if mut in ("", "WT"):
                wt_bright[gtype].append(brightness)
                continue
            parts = [p for p in mut.split(":") if p]
            if len(parts) != 1:
                continue
            m = _MUT.match(parts[0])
            if not m:
                continue
            # aaMutations' number is a 0-based index into the FASTA sequence
            # (verified 100% on single-sub rows); +1 gives the 1-based sfGFP position.
            pos = int(m.group(1)) + 1
            if 1 <= pos <= length:
                subs[(pos, m.group(2), gtype)].append(brightness)
    wt_mean = {t: sum(v) / len(v) for t, v in wt_bright.items() if v}
    best: dict[tuple[int, str], float] = {}
    for (pos, target, gtype), vals in subs.items():
        if gtype not in wt_mean:
            continue
        rel = max(vals) - wt_mean[gtype]
        key = (pos, target)
        if key not in best or rel > best[key]:
            best[key] = rel
    return best


def brightness_pools(
    wt_seq: str,
    dms_path: str | Path,
    winners_path: str | Path,
    free_mutate: set[int],
    do_not_mutate: set[int],
    *,
    eps: float,
    enhancing_margin: float,
    min_pos: int = 11,
    types: tuple[str, ...] = ("avGFP", "amacGFP"),
) -> BrightnessPools:
    """Build the proven and neutral_or_better brightness pools (sfGFP-relative)."""
    rel = _dms_rel_brightness(dms_path, len(wt_seq), types)
    winners = _winner_mutations(wt_seq, winners_path, free_mutate, do_not_mutate, min_pos)

    proven: dict[str, Mutation] = dict(winners)
    neutral: dict[str, Mutation] = {}
    for (pos, target), score in rel.items():
        if not gate_single(pos, target, wt_seq, free_mutate, do_not_mutate, min_pos):
            continue
        mut = Mutation(pos=pos, wt=wt_seq[pos - 1], target=target)
        if score >= enhancing_margin:
            proven[mut.token] = mut
        if score >= -eps:
            neutral[mut.token] = mut

    neutral_only = [m for tok, m in neutral.items() if tok not in proven]
    return BrightnessPools(
        proven=tuple(sorted(proven.values(), key=lambda m: m.pos)),
        neutral_or_better=tuple(sorted(neutral_only, key=lambda m: m.pos)),
    )
