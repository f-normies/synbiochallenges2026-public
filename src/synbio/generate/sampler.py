"""Iterative Gibbs masked sampling over a fixed position set (pure given a logits_fn).

The model call is injected as `logits_fn(seq, positions) -> [len(positions), 20]` over
ALPHABET, so the loop is unit-testable without a GPU. With `exclude_wt` each chosen
position is forced off its WT residue, so the Hamming distance equals len(positions).
"""

import numpy as np

__all__ = ["ALPHABET", "gibbs_sample"]

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
_AA_INDEX = {a: i for i, a in enumerate(ALPHABET)}


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    z = np.asarray(logits, dtype=float) / max(temperature, 1e-6)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def gibbs_sample(
    wt_seq: str,
    positions: list[int],
    logits_fn,
    rng: np.random.Generator,
    *,
    n_passes: int,
    temperature: float,
    exclude_wt: bool,
    omit: frozenset[str] = frozenset(),
) -> str:
    """Resample `positions` (1-based) by Gibbs passes; other positions stay WT.

    `omit` is a set of amino acids that can never be sampled (e.g. "C" to exclude
    free surface cysteines in the reducing CFPS environment, PROJECT_PLAN §4.3).
    """
    seq = list(wt_seq)
    for _ in range(n_passes):
        for p in positions:
            logits = np.asarray(logits_fn("".join(seq), [p]))[0]
            probs = _softmax(logits, temperature)
            for aa in omit:
                if aa in _AA_INDEX:
                    probs[_AA_INDEX[aa]] = 0.0
            if exclude_wt:
                wt_aa = wt_seq[p - 1]
                if wt_aa in _AA_INDEX:
                    probs[_AA_INDEX[wt_aa]] = 0.0
            if omit or exclude_wt:
                total = probs.sum()
                if total > 0:
                    probs = probs / total
            aa_idx = int(rng.choice(len(ALPHABET), p=probs))
            seq[p - 1] = ALPHABET[aa_idx]
    return "".join(seq)
