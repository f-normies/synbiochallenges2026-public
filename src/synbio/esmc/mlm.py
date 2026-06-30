"""ESMC-600M masked-LM logits for the Gibbs sampler (env esm; torch/transformers lazy).

HPC-only model path (compute nodes load ESMC-600M via the HF transformers fork). The Gibbs
loop consumes `make_logits_fn(handle)` → `logits_fn(seq, positions) -> [len(positions), 20]`
over `sampler.ALPHABET`. Not exercised in .venv (the loop is tested with a fake logits_fn).
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from synbio.generate.sampler import ALPHABET

__all__ = ["EsmcMlmHandle", "load_esmc_mlm", "logits_at", "make_logits_fn"]

logger = logging.getLogger(__name__)


@dataclass
class EsmcMlmHandle:
    """Loaded ESMC masked-LM + tokenizer with the AA→token-id map for 20-way logits."""

    model: Any
    tokenizer: Any
    mask_id: int
    aa_token_ids: list[int]


def load_esmc_mlm(model_id: str, dtype: str = "float16") -> EsmcMlmHandle:
    """Load the HF ESMC masked-LM head (e.g. `biohub/ESMC-600M`), fp16, device_map=auto."""
    import torch
    from transformers import AutoTokenizer, ESMCForMaskedLM

    tok = AutoTokenizer.from_pretrained(model_id)
    model = ESMCForMaskedLM.from_pretrained(
        model_id, torch_dtype=getattr(torch, dtype), device_map="auto"
    ).eval()
    aa_token_ids = [tok.convert_tokens_to_ids(a) for a in ALPHABET]
    logger.info("loaded HF ESMC MLM %s (mask_id=%s)", model_id, tok.mask_token_id)
    return EsmcMlmHandle(
        model=model, tokenizer=tok, mask_id=int(tok.mask_token_id), aa_token_ids=aa_token_ids
    )


def logits_at(handle: EsmcMlmHandle, seq: str, positions: list[int]) -> np.ndarray:
    """Per-AA logits ([len(positions), 20] over ALPHABET) at each masked 1-based position.

    Residue p (1-based) maps to token index p (CLS at index 0). Verify on HPC if the
    tokenizer's special-token layout differs.
    """
    import torch

    tok, model = handle.tokenizer, handle.model
    device = next(model.parameters()).device
    enc = tok([seq], return_tensors="pt").to(device)
    ids = enc["input_ids"]
    for p in positions:
        ids[0, p] = handle.mask_id
    with torch.no_grad():
        out = model(**enc)
    logits = out.logits[0]  # [L_tok, vocab]
    aa = torch.tensor(handle.aa_token_ids, device=device)
    rows = [logits[p][aa].float().cpu().numpy() for p in positions]
    return np.stack(rows)


def make_logits_fn(handle: EsmcMlmHandle):
    """Adapt a handle into the `logits_fn(seq, positions)` callable the Gibbs loop expects."""
    def fn(seq: str, positions: list[int]) -> np.ndarray:
        return logits_at(handle, seq, positions)
    return fn
