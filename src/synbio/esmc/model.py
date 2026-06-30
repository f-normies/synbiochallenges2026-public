"""Load ESMC for frozen embeddings via the HF transformers fork (env esm). Heavy: lazy imports.

We use the HF `ESMCModel` (not the ESM SDK `ESMC.from_pretrained`) for two reasons verified on
the cluster: (1) the SDK's `ESMC_6B_202412` builder hits `NotImplementedError: Cannot copy out of
meta tensor` on `model.to(device)`; (2) the HF repo `biohub/ESMC-6B` is the SAME one branch B
fine-tunes, so the 24 GB weights are cached once instead of pulling a second SDK-format copy.
"""

import logging
from dataclasses import dataclass
from typing import Any

__all__ = ["EsmcHandle", "load_esmc"]

logger = logging.getLogger(__name__)


@dataclass
class EsmcHandle:
    """Loaded ESMC encoder + tokenizer with layer/width metadata."""

    model: Any
    tokenizer: Any
    n_layers: int
    d_model: int


def load_esmc(model_id: str, dtype: str = "float16") -> EsmcHandle:
    """Load the HF ESMC base encoder for embedding extraction.

    `model_id` is an HF repo id (e.g. `biohub/ESMC-6B`). Loaded with `device_map="auto"` so
    accelerate places materialized weights directly on the GPU (avoids the meta-tensor `.to()`
    failure). `n_layers`/`d_model` are measured from a 1-token forward — no config-attr guessing.
    """
    import torch
    from transformers import AutoTokenizer, ESMCModel

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = ESMCModel.from_pretrained(
        model_id, torch_dtype=getattr(torch, dtype), device_map="auto"
    ).eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        enc = tokenizer(["M"], return_tensors="pt").to(device)
        hidden_states = model(**enc, output_hidden_states=True).hidden_states
    n_layers = len(hidden_states) - 1  # HF tuple: index 0 = embeddings, 1..n = block outputs
    d_model = int(hidden_states[0].shape[-1])
    logger.info("loaded HF ESMC %s: %d layers (+embed), d=%d, dtype=%s", model_id, n_layers, d_model, dtype)
    return EsmcHandle(model=model, tokenizer=tokenizer, n_layers=n_layers, d_model=d_model)
