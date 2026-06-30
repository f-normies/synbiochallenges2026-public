"""Apply the stage-02 brightness probe to sequences → B̂ in ×WT.

Shared by the ESMC sampler (stage 03 rerank) and stage 04 filter_brightness: loads the
CalibratedRidge + meta.json contract, embeds at the probe's layer/pool, linearizes to ×WT.
`embed_sequences` is module-level so callers can monkeypatch it in .venv tests.
"""

import json
from pathlib import Path

import numpy as np

from synbio.esmc import EmbeddingCache, embed_sequences
from synbio.probes.ridge import CalibratedRidge

__all__ = ["score_brightness"]


def score_brightness(
    seqs: list[str],
    probe_dir: str | Path,
    handle,
    batch_size: int = 8,
    cache_dir: str | Path = "workspace/embed_cache",
) -> np.ndarray:
    """Predicted brightness in ×WT for each sequence, via the stage-02 CalibratedRidge.

    Reads `probe_dir/{ridge.npz, meta.json}`; embeds with ESMC-6B at the probe's
    layer/pool (aromatic local pool if meta carries `aromatic_indices`); linearizes
    `10**(predict - sfgfp_anchor)`.
    """
    probe_dir = Path(probe_dir)
    meta = json.loads((probe_dir / "meta.json").read_text())
    probe = CalibratedRidge.load(probe_dir / "ridge.npz")
    layer = int(meta["layer"])
    pool = meta["pool"]
    local_idx = meta.get("aromatic_indices")
    anchor = float(meta["sfgfp_anchor"])
    cache = EmbeddingCache(cache_dir, model_tag="esmc6b", layer=layer, pool=pool)
    emb = embed_sequences(handle, list(seqs), layer, batch_size, cache, pool, local_idx)
    raw = probe.predict(emb)
    return np.asarray(10.0 ** (raw - anchor), dtype=float)
