"""Split-budget layered sampler: build near-WT designs from the three pools.

Pocket-relevant layers (bright + consensus) are capped tight (predictor-trust limit
~8); surface recharge is dosed higher (orthogonal to the aromatic-pooled brightness
probe, low mutual epistasis). Deterministic: a single seeded RNG; pure-layer anchors
first, then random layered draws; dedup on the final sequence.
"""

from dataclasses import dataclass

import numpy as np

from synbio.io.constraints import validate_sequence
from synbio.utils.logging import get_logger

from .apply import Mutation, apply_mutations, hamming
from .pools import BrightnessPools

__all__ = ["BudgetConfig", "Design", "bucket_of", "sample_designs"]

logger = get_logger(__name__)


@dataclass(frozen=True)
class BudgetConfig:
    n_designs: int
    max_mutations: int
    pocket_cap: int
    k_bright: tuple[int, int]
    k_consensus: tuple[int, int]
    k_recharge: tuple[int, int]
    seed_ladders: bool
    low_bucket_max: int
    max_attempts_factor: int


@dataclass(frozen=True)
class Design:
    sequence: str
    mutations: tuple[Mutation, ...]
    pool_counts: dict[str, int]
    bucket: str


def bucket_of(n_mut: int, low_bucket_max: int = 8) -> str:
    """'low' if total mutations <= low_bucket_max, else 'mid'."""
    return "low" if n_mut <= low_bucket_max else "mid"


def _make_design(wt_seq: str, muts: list[Mutation], counts: dict[str, int],
                 low_bucket_max: int) -> Design | None:
    """Apply + validate; return a Design or None if invalid."""
    seq = apply_mutations(wt_seq, muts)
    ok, _ = validate_sequence(seq, set(), require_nterm=True)
    if not ok:
        return None
    n = hamming(wt_seq, seq)
    return Design(seq, tuple(muts), counts, bucket_of(n, low_bucket_max))


def _draw_without_collision(
    rng: np.random.Generator, pool: list[Mutation], k: int, used: set[int]
) -> list[Mutation]:
    """Pick up to k mutations from pool at positions not already used."""
    if k <= 0 or not pool:
        return []
    idx = rng.permutation(len(pool))
    picked: list[Mutation] = []
    for i in idx:
        mut = pool[int(i)]
        if mut.pos in used:
            continue
        picked.append(mut)
        used.add(mut.pos)
        if len(picked) == k:
            break
    return picked


def _first_distinct(pool: list[Mutation], k: int) -> list[Mutation]:
    """First k mutations from pool with unique positions (pools may repeat a position,
    e.g. recharge carries K->E and K->Q at the same site)."""
    out: list[Mutation] = []
    used: set[int] = set()
    for mut in pool:
        if mut.pos in used:
            continue
        out.append(mut)
        used.add(mut.pos)
        if len(out) == k:
            break
    return out


def _ladders(wt_seq: str, bright: BrightnessPools, recharge: list[Mutation],
             budget: BudgetConfig) -> list[Design]:
    """Deterministic pure-layer anchors: pure-recharge and pure-brightness ladders."""
    designs: list[Design] = []
    bright_seq = bright.proven + bright.neutral_or_better
    for k in range(1, budget.k_recharge[1] + 1):
        muts = _first_distinct(recharge, k)
        if len(muts) < k:
            break
        d = _make_design(wt_seq, muts, {"bright": 0, "recharge": k, "consensus": 0},
                         budget.low_bucket_max)
        if d is not None:
            designs.append(d)
    for k in range(1, budget.pocket_cap + 1):
        muts = _first_distinct(bright_seq, k)
        if len(muts) < k:
            break
        d = _make_design(wt_seq, muts, {"bright": k, "recharge": 0, "consensus": 0},
                         budget.low_bucket_max)
        if d is not None:
            designs.append(d)
    return designs


def sample_designs(
    wt_seq: str,
    bright: BrightnessPools,
    consensus: list[Mutation],
    recharge: list[Mutation],
    budget: BudgetConfig,
    seed: int,
) -> list[Design]:
    """Generate up to n_designs unique near-WT designs under the split budget."""
    rng = np.random.default_rng(seed)
    bright_pool = bright.proven + bright.neutral_or_better
    designs: list[Design] = []
    seen: set[str] = set()

    def _add(d: Design | None) -> None:
        if d is not None and d.sequence not in seen and d.sequence != wt_seq:
            seen.add(d.sequence)
            designs.append(d)

    if budget.seed_ladders:
        for d in _ladders(wt_seq, bright, recharge, budget):
            _add(d)
            if len(designs) >= budget.n_designs:
                return designs[: budget.n_designs]

    attempts = 0
    max_attempts = budget.n_designs * budget.max_attempts_factor
    while len(designs) < budget.n_designs and attempts < max_attempts:
        attempts += 1
        kb = int(rng.integers(budget.k_bright[0], budget.k_bright[1] + 1))
        kc = int(rng.integers(budget.k_consensus[0], budget.k_consensus[1] + 1))
        if kb + kc > budget.pocket_cap:
            continue
        kr = int(rng.integers(budget.k_recharge[0], budget.k_recharge[1] + 1))
        if kb + kc + kr == 0 or kb + kc + kr > budget.max_mutations:
            continue
        used: set[int] = set()
        mb = _draw_without_collision(rng, bright_pool, kb, used)
        mc = _draw_without_collision(rng, consensus, kc, used)
        mr = _draw_without_collision(rng, recharge, kr, used)
        muts = mb + mc + mr
        if not muts:
            continue
        counts = {"bright": len(mb), "consensus": len(mc), "recharge": len(mr)}
        _add(_make_design(wt_seq, muts, counts, budget.low_bucket_max))

    if len(designs) < budget.n_designs:
        logger.warning("combinatorial: produced %d/%d designs after %d attempts",
                       len(designs), budget.n_designs, attempts)
    return designs
