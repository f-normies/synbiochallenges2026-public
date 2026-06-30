"""ESMC access layer (env esm; torch/esm/transformers/peft imported lazily inside functions)."""

from synbio.esmc.cache import EmbeddingCache
from synbio.esmc.embed import embed_sequences, masked_mean_pool
from synbio.esmc.extract import extract_features
from synbio.esmc.model import EsmcHandle, load_esmc
from synbio.esmc.pocket import resolve_indices

__all__ = [
    "EmbeddingCache",
    "embed_sequences",
    "masked_mean_pool",
    "extract_features",
    "EsmcHandle",
    "load_esmc",
    "resolve_indices",
]
