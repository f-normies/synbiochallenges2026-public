"""Embedding cache: durable per-sequence `.npy` + a consolidated read matrix.

Per-sequence files (`<sha256>.npy`, key = sha256(seq | model_tag | layer | pool)) stay the durable
source of truth: `get_or_compute` computes missing sequences in batches and writes each vector
atomically (temp + rename) before the next batch, so a crash loses at most one batch.

On top of that, a single consolidated matrix (`_matrix.npy` + `_keys.json`) accelerates reads. A
fully-cached pass becomes one big read + a fancy index instead of one `stat` + one `np.load` per
sequence — the per-seq pattern is ~tens of minutes for ~10^5 tiny files on a small-file-hostile
networked FS (CephFS EC). The consolidated file is built/extended cumulatively from the per-seq
files, so an existing per-seq cache is reused (no recompute); the first pass after it appears pays
the per-seq read once, every later pass is seconds.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Callable

import numpy as np

__all__ = ["EmbeddingCache"]


class EmbeddingCache:
    """Disk cache of fp32 embeddings: durable per-seq `.npy` + a consolidated read matrix."""

    def __init__(self, root: str | Path, model_tag: str, layer: int, pool: str,
                 dtype: str = "float32") -> None:
        self.dir = Path(root) / f"{model_tag}_L{layer}_{pool}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._prefix = f"{model_tag}|{layer}|{pool}|"
        self._dtype = np.dtype(dtype)
        self._matrix = self.dir / "_matrix.npy"
        self._keys = self.dir / "_keys.json"

    def _hash(self, seq: str) -> str:
        return hashlib.sha256((self._prefix + seq).encode()).hexdigest()

    def _path(self, seq: str) -> Path:
        return self.dir / f"{self._hash(seq)}.npy"

    def has(self, seq: str) -> bool:
        """True if `seq`'s embedding is already on disk (per-seq file)."""
        return self._path(seq).exists()

    def _save(self, seq: str, vec: np.ndarray) -> None:
        """Atomically write one vector (temp file + os.replace) so a crash never leaves a partial."""
        path = self._path(seq)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            np.save(fh, np.asarray(vec, dtype=self._dtype))
        os.replace(tmp, path)

    def _load_consolidated(self) -> tuple[dict[str, int], np.ndarray] | None:
        """Return (hash -> row, matrix) if the consolidated cache exists, else None."""
        if not (self._matrix.exists() and self._keys.exists()):
            return None
        keys = json.loads(self._keys.read_text())
        mat = np.load(self._matrix)
        return {h: i for i, h in enumerate(keys)}, mat

    def _write_consolidated(self, keys: list[str], mat: np.ndarray) -> None:
        """Atomically persist the consolidated keys + matrix (temp + os.replace)."""
        tmp_m = self._matrix.with_suffix(".npy.tmp")
        with open(tmp_m, "wb") as fh:
            np.save(fh, mat.astype(self._dtype))
        os.replace(tmp_m, self._matrix)
        tmp_k = self._keys.with_suffix(".json.tmp")
        tmp_k.write_text(json.dumps(keys))
        os.replace(tmp_k, self._keys)

    def _extend_consolidated(
        self, uniq: list[str], cons: tuple[dict[str, int], np.ndarray] | None
    ) -> tuple[dict[str, int], np.ndarray]:
        """Ensure the consolidated cache covers `uniq`, reading per-seq files only for new keys."""
        keys = list(cons[0]) if cons else []
        present = set(keys)
        mat = cons[1] if cons else None
        to_add = [s for s in uniq if self._hash(s) not in present]
        if to_add:
            add_mat = np.stack([np.load(self._path(s)) for s in to_add]).astype(self._dtype)
            mat = add_mat if mat is None else np.vstack([mat, add_mat])
            keys = keys + [self._hash(s) for s in to_add]
            self._write_consolidated(keys, mat)
        return {h: i for i, h in enumerate(keys)}, mat

    def get_or_compute(
        self,
        sequences: list[str],
        compute_fn: Callable[[list[str]], np.ndarray],
        batch_size: int,
    ) -> np.ndarray:
        """Return [N, d] fp32 embeddings aligned to `sequences`, computing only cache misses.

        Fast path: if the consolidated matrix already covers every requested sequence, return it
        with one big read + a fancy index. Otherwise compute the missing sequences to durable
        per-seq files (each batch saved before the next), then extend the consolidated matrix and
        return from it. `compute_fn(batch)` must return an array aligned to `batch`.
        """
        req = [self._hash(s) for s in sequences]
        cons = self._load_consolidated()
        if cons is not None and all(h in cons[0] for h in req):
            pos, mat = cons
            return mat[[pos[h] for h in req]].astype(np.float32)

        uniq = list(dict.fromkeys(sequences))
        on_disk = {p.stem for p in self.dir.glob("*.npy")}  # one listing, not one stat per seq
        missing = [s for s in uniq if self._hash(s) not in on_disk]
        for start in range(0, len(missing), batch_size):
            batch = missing[start : start + batch_size]
            vecs = np.asarray(compute_fn(batch), dtype=np.float32)
            for seq, vec in zip(batch, vecs):
                self._save(seq, vec)

        pos, mat = self._extend_consolidated(uniq, cons)
        return mat[[pos[h] for h in req]].astype(np.float32)

    def cached_hashes(self) -> set[str]:
        """All stored sequence-hashes (consolidated keys + per-seq files); one listing, no stat."""
        cons = self._load_consolidated()
        keys = set(cons[0]) if cons else set()
        return keys | {p.stem for p in self.dir.glob("*.npy")}

    def filter_uncached(self, sequences: list[str]) -> list[str]:
        """Subset of `sequences` not yet stored (one consolidated read + glob; CephFS-friendly)."""
        have = self.cached_hashes()
        return [s for s in sequences if self._hash(s) not in have]

    def save_batch(self, sequences: list[str], vectors: np.ndarray) -> None:
        """Durably write per-seq vectors (atomic; first-write-wins). No consolidation — the cheap
        incremental path for bulk extraction; call consolidate() once when the bulk write is done."""
        vectors = np.asarray(vectors)
        if len(vectors) != len(sequences):
            raise ValueError(f"{len(vectors)} vectors for {len(sequences)} sequences")
        seen: set[str] = set()
        for s, v in zip(sequences, vectors):
            if s in seen:
                continue
            seen.add(s)
            if not self._path(s).exists():
                self._save(s, v)

    def consolidate(self, sequences: list[str]) -> None:
        """Ensure the consolidated read-matrix covers `sequences` (one read + at most one write)."""
        self._extend_consolidated(list(dict.fromkeys(sequences)), self._load_consolidated())

    def put(self, sequences: list[str], vectors: np.ndarray) -> np.ndarray:
        """Store precomputed [N, d] vectors (durable per-seq + consolidated extend); return [N, d] fp32.

        Convenience for one-shot callers. For incremental bulk extraction prefer save_batch() per
        chunk then a single consolidate() at the end. Existing per-seq files are not overwritten.
        """
        vectors = np.asarray(vectors)
        if len(vectors) != len(sequences):
            raise ValueError(f"{len(vectors)} vectors for {len(sequences)} sequences")
        if not sequences:
            return np.empty((0, vectors.shape[1] if vectors.ndim == 2 else 0), dtype=np.float32)
        self.save_batch(sequences, vectors)
        pos, mat = self._extend_consolidated(list(dict.fromkeys(sequences)), self._load_consolidated())
        return mat[[pos[self._hash(s)] for s in sequences]].astype(np.float32)
