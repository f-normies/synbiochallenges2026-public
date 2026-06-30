"""Sampling-position selection for the ESMC-600M sampler (pure).

The sampler may mask only DMS-tolerant positions past the frozen N-terminus. The
tolerant set already excludes the hard mask (catalytic / chromophore / core), so no
further mask input is needed.
"""

import numpy as np

__all__ = ["sampling_set", "pick_subset"]


def sampling_set(free_mutate: list[int], nterm_freeze: int = 10) -> list[int]:
    """Tolerant positions the sampler may mask: free_mutate with pos > nterm_freeze, sorted."""
    return sorted(p for p in set(free_mutate) if p > nterm_freeze)


def pick_subset(positions: list[int], k: int, rng: np.random.Generator) -> list[int]:
    """Pick k distinct positions (sorted) from `positions` using `rng`."""
    pool = list(positions)
    if k > len(pool):
        raise ValueError(f"k={k} exceeds available positions {len(pool)}")
    idx = rng.choice(len(pool), size=k, replace=False)
    return sorted(pool[int(i)] for i in idx)
