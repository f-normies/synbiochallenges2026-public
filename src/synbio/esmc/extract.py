"""ESMC multi-(layer,pool) feature extraction: pure pooling fns + a heavy one-forward extractor.

The forward is the cost, so one `model(..., output_hidden_states=True)` per batch feeds every
(layer, pool) cache. Pooling math is pure numpy (testable in .venv); the GPU forward is lazy-torch
and cluster-only, mirroring embed_sequences.
"""

import logging
from typing import TYPE_CHECKING

import numpy as np

from synbio.esmc.embed import masked_mean_pool

if TYPE_CHECKING:
    from synbio.esmc.cache import EmbeddingCache

__all__ = ["pool_cls", "masked_max_pool", "pool_indices", "extract_features"]

logger = logging.getLogger(__name__)


def pool_cls(hidden: np.ndarray) -> np.ndarray:
    """The CLS/BOS token vector (token index 0). hidden [..., L, d] -> [..., d] fp32."""
    return np.asarray(hidden, dtype=np.float64)[..., 0, :].astype(np.float32)


def masked_max_pool(hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Max over real (mask==1) positions. hidden [..., L, d], mask [..., L] -> [..., d] fp32."""
    hidden = np.asarray(hidden, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    neg = np.where(mask[..., None] > 0, hidden, -np.inf)
    return neg.max(axis=-2).astype(np.float32)


def pool_indices(hidden: np.ndarray, indices: list[int]) -> np.ndarray:
    """Mean over the given token indices. hidden [..., L, d] -> [..., d] fp32.

    No mask applied: callers must guarantee every index falls on a real (non-pad) token. Safe here
    because all dataset sequences are the same fixed length (substitution-only 238-aa GFP variants),
    so the canonical pocket positions (max 223) always land on real residues, never <eos>/padding.
    """
    hidden = np.asarray(hidden, dtype=np.float64)
    return hidden[..., list(indices), :].mean(axis=-2).astype(np.float32)


def _pool(pool: str, hidden: np.ndarray, mask: np.ndarray, local_indices: dict) -> np.ndarray:
    """Dispatch a single pool over one layer's hidden states [B, L, d] -> [B, d] fp32."""
    if pool == "mean":
        return masked_mean_pool(hidden, mask)
    if pool == "cls":
        return pool_cls(hidden)
    if pool == "max":
        return masked_max_pool(hidden, mask)
    if pool in local_indices:
        return pool_indices(hidden, local_indices[pool])
    raise ValueError(f"unknown pool: {pool}")


def extract_features(
    handle,
    sequences: list[str],
    layers: list[int],
    pools: list[str],
    local_indices: dict[str, list[int]],
    caches: dict[tuple[int, str], "EmbeddingCache"],
    batch_size: int,
) -> None:
    """One forward per batch → every (layer, pool) cache. Resumable; writes are the product.

    `caches` maps (layer, pool) -> EmbeddingCache. `local_indices` maps each local pool name
    (chromo/pocket/aromatic) to its token indices (background-independent: colinear backgrounds).
    Only sequences missing from at least one target cache are forwarded.
    """
    import torch  # lazy: keeps the module importable without torch

    model, tokenizer = handle.model, handle.tokenizer
    device = next(model.parameters()).device
    todo = sorted({s for c in caches.values() for s in c.filter_uncached(sequences)})
    logger.info("extract_features: %d/%d sequences need a forward (%d caches)",
                len(todo), len(sequences), len(caches))

    for start in range(0, len(todo), batch_size):
        batch = todo[start : start + batch_size]
        enc = tokenizer(batch, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        mask = enc["attention_mask"].cpu().numpy()
        for layer in layers:
            hidden = out.hidden_states[layer].float().cpu().numpy()  # [B, L, d] fp32
            for pool in pools:
                caches[(layer, pool)].save_batch(batch, _pool(pool, hidden, mask, local_indices))
        if (start // batch_size) % 50 == 0:
            logger.info("extract_features: %d/%d sequences done", start + len(batch), len(todo))

    for cache in caches.values():
        cache.consolidate(sequences)
    logger.info("extract_features: consolidated %d caches", len(caches))
