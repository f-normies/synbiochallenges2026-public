"""ESMC embedding extraction: masked mean-pool (pure) + a cached batched forward (heavy)."""

import logging

import numpy as np

from synbio.esmc.cache import EmbeddingCache

__all__ = ["masked_mean_pool", "embed_sequences", "LOCAL_POOLS"]

logger = logging.getLogger(__name__)


def masked_mean_pool(hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean over real (mask==1) positions. hidden [..., L, d], mask [..., L] -> [..., d]."""
    hidden = np.asarray(hidden, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    summed = (hidden * mask[..., None]).sum(axis=-2)
    counts = np.clip(mask.sum(axis=-1, keepdims=True), 1.0, None)
    return (summed / counts).astype(np.float32)


LOCAL_POOLS = ("chromo", "pocket", "aromatic")


def embed_sequences(
    handle,
    sequences: list[str],
    layer: int,
    batch_size: int,
    cache: EmbeddingCache,
    pool: str = "mean",
    local_indices: list[int] | None = None,
) -> np.ndarray:
    """Return [N, d] fp32 pooled embeddings at `layer`, computing only cache misses.

    pool ∈ {mean, cls, max} (whole-sequence) or {chromo, pocket, aromatic} (mean over
    `local_indices`, required for those). `mean` + `local_indices=None` is identical to the
    previous behavior. The pooling helpers are imported lazily from `synbio.esmc.extract`
    (module-level would be circular: `extract` imports `masked_mean_pool` from this module).

    HF ESMCModel forward (fp16 trunk on V100): tokenize the batch, run
    `model(**enc, output_hidden_states=True)`, take `hidden_states[layer]`, mask out pad with the
    attention_mask, pool, cast fp32. `hidden_states` is a tuple of length n_layers+1 (index 0 =
    embeddings, 1..n = block outputs; ESMC-6B → 81 entries). Not .venv-tested past the guard.
    """
    if pool in LOCAL_POOLS and local_indices is None:
        raise ValueError(f"pool={pool!r} requires local_indices")
    import torch  # lazy: keeps the module importable without torch

    model, tokenizer = handle.model, handle.tokenizer
    device = next(model.parameters()).device

    def compute(batch: list[str]) -> np.ndarray:
        """Forward one batch → [len(batch), d] fp32. The cache batches + saves incrementally."""
        from synbio.esmc.extract import masked_max_pool, pool_cls, pool_indices

        enc = tokenizer(batch, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        hidden = out.hidden_states[layer].float().cpu().numpy()  # [B, L, d] fp32
        mask = enc["attention_mask"].cpu().numpy()  # [B, L]
        if pool == "mean":
            return masked_mean_pool(hidden, mask)
        if pool == "cls":
            return pool_cls(hidden)
        if pool == "max":
            return masked_max_pool(hidden, mask)
        return pool_indices(hidden, local_indices)

    return cache.get_or_compute(sequences, compute, batch_size)
