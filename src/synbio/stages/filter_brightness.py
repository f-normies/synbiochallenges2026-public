"""Stage 04: brightness filter (PPL gate -> probe -> 0.6× floor -> rank top-500; owned by P3/Ч3).

Pipeline of stage 04, in order:
  (a) **foldability PPL gate** — drop designs whose ESMC pseudo-perplexity is far above sfGFP
      (high PPL → folds worse, DOI 10.64898/2026.06.03.729735 fig S11); the main target is the
      ESMC-600M sampler's high-risk junk (upside slot), cheap insurance against the 0.3× cliff.
  (b) **0.6× brightness floor** — the CalibratedRidge probe, used in-distribution near sfGFP.
  (c) **rank with magnitude** — top-N by ×WT brightness `B̂`, blended with a weak ESMC-6B LLR
      voice (PROJECT_PLAN §5/§7-04в: raw LLR is weak, ρ≈0.2–0.3, so a low-weight second opinion).
      The blend reorders only the top-N *cut*; the `b_hat` magnitude carried downstream is untouched.

PPL method = **single-pass OFS** (arXiv 2407.07265): read p(xᵢ|whole seq) in one forward, no
masking; ≈ true PLL for fitness ranking, ~minutes not hours. One `ESMCForMaskedLM` forward yields
BOTH logits (PPL + the wt-marginal LLR) and hidden_states (the L24-aromatic brightness embedding),
so the heavy passes fuse — the LLR voice reuses sfGFP's own logits and is effectively free. The
masked-LM loader/forward live module-local (the shared `synbio.esmc` package stays on `ESMCModel`);
only the pure pooling helper is reused from it.

The pure, GPU-free core (×WT linearization, PPL math, gate masks, floor/top-N) is .venv-testable
against the fixed stage-02 `meta.json` contract; the fused forward is cluster-only.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from synbio.orchestrator import register_stage
from synbio.orchestrator.stage import Decision, StageConfig, StageResult, StageSpec, cli

logger = logging.getLogger(__name__)

__all__ = [
    "SPEC",
    "LINEARIZATION",
    "assert_linearization",
    "brightness_xwt",
    "pseudo_log_likelihood",
    "wt_marginal_llr",
    "ppl_gate",
    "upside_mask",
    "select_brightness",
    "run",
]

SPEC = StageSpec(
    name="filter_brightness",
    module="filter_brightness",
    env="esm",
    inputs=("candidates", "brightness_probe", "wt"),
    outputs=("candidates_bright",),
)

# The stage-02 probe records this exact string in meta.json; raw predictions are avGFP-WT-centered
# log10-brightness, so dividing by the sfGFP forward (anchor) in log space = ×sfGFP magnitude.
LINEARIZATION = "10**(predict(b) - sfgfp_anchor)"


def assert_linearization(meta: dict) -> None:
    """Fail loudly if the probe's meta.json linearization contract drifted from `LINEARIZATION`."""
    got = meta.get("linearization")
    if got != LINEARIZATION:
        raise ValueError(
            f"brightness probe linearization contract drift: expected {LINEARIZATION!r}, got {got!r}"
        )


def brightness_xwt(raw_pred: np.ndarray, anchor: float) -> np.ndarray:
    """Linearize raw probe predictions to brightness in ×sfGFP units: 10**(pred - anchor).

    `raw_pred` is the CalibratedRidge output (avGFP-WT-centered log10-brightness); `anchor` is the
    probe's own sfGFP forward (meta.json `sfgfp_anchor`). Sharing the anchor cancels the probe's
    distance-from-avGFP pessimism for near-WT designs (see docs/experiments §4).
    """
    return np.power(10.0, np.asarray(raw_pred, dtype=float) - float(anchor))


def pseudo_log_likelihood(
    logits: np.ndarray, input_ids: np.ndarray, first: int, last: int
) -> float:
    """Mean log p(xᵢ | seq) over residue positions [first, last) — single-pass OFS, pure numpy.

    `logits` is [L_tok, vocab] from one un-masked forward; `input_ids` the matching token ids.
    `[first, last)` selects the real-residue span (drops the leading <cls> and trailing <eos>).
    Returns the mean per-position log-probability; pseudo-perplexity is `exp(-this)`.
    """
    logits = np.asarray(logits, dtype=np.float64)[first:last]
    ids = np.asarray(input_ids, dtype=int)[first:last]
    if len(ids) == 0:
        raise ValueError(f"empty residue span [{first}, {last}) for logits of len {len(logits)}")
    # log_softmax = logits - logsumexp(logits), gathered at the true residue token
    m = logits.max(axis=-1, keepdims=True)
    logZ = m[:, 0] + np.log(np.exp(logits - m).sum(axis=-1))
    true_logit = logits[np.arange(len(ids)), ids]
    return float(np.mean(true_logit - logZ))


def wt_marginal_llr(
    wt_logits: np.ndarray, wt_ids: np.ndarray, design_ids: np.ndarray, first: int, last: int
) -> float:
    """Single-pass wt-marginal LLR of a design vs sfGFP (Meier et al. 2021) — pure numpy.

    One forward on sfGFP scores every design's substitutions: Σ over the colinear span [first, last)
    of ``logp_sfgfp(design_aaᵢ) − logp_sfgfp(wt_aaᵢ)``. The per-position softmax normalizer cancels
    between the two terms, so this reduces to the difference of the *raw* sfGFP logits at the two
    tokens. Positive = the design's residues are likelier than sfGFP's under ESMC; non-substituted
    positions contribute 0. Assumes design and sfGFP are colinear (same length, substitutions only) —
    the caller passes the shared real-residue span and NaNs off-length designs.
    """
    wt_logits = np.asarray(wt_logits, dtype=np.float64)[first:last]
    wt_ids = np.asarray(wt_ids, dtype=int)[first:last]
    design_ids = np.asarray(design_ids, dtype=int)[first:last]
    if len(wt_ids) == 0:
        raise ValueError(f"empty residue span [{first}, {last}) for logits of len {len(wt_logits)}")
    pos = np.arange(len(wt_ids))
    return float(np.sum(wt_logits[pos, design_ids] - wt_logits[pos, wt_ids]))


def ppl_gate(ppl: np.ndarray, sfgfp_ppl: float, margin: float) -> np.ndarray:
    """Bool keep-mask: a design passes iff its pseudo-perplexity ≤ `margin` × sfGFP's.

    Soft foldability gate anchored to sfGFP (the anchor cancels OFS's single-pass bias). `margin`
    is multiplicative over sfGFP (conf `ppl_margin`, default 1.5×). Recharge designs (our thermo
    prior) read slightly higher PPL, so the gate is deliberately loose, not a tight cutoff.
    """
    return np.asarray(ppl, dtype=float) <= float(margin) * float(sfgfp_ppl)


def upside_mask(sources: "pd.Series | list[str]", tokens: tuple[str, ...]) -> np.ndarray:
    """Bool mask marking high-risk upside-slot rows (their `source` contains any of `tokens`).

    These rows bypass the 0.6× brightness floor in `select_brightness` (the floor over-rejects the
    high-mutation sampler slot; viability is carried by the PPL gate + fold_sanity instead). They
    are **not** exempt from the PPL gate — catching sampler junk is exactly what that gate is for.
    """
    s = pd.Series(list(sources), dtype="object").fillna("").astype(str)
    low = tuple(t.lower() for t in tokens)
    return s.str.lower().apply(lambda v: any(t in v for t in low)).to_numpy(dtype=bool)


def _rank_pct(values: np.ndarray) -> np.ndarray:
    """Tie-corrected rank-percentile in [0, 1]; highest value → 1.0, single element → 0.5."""
    from synbio.stages.rank_combine import average_rank  # shared pure helper (no heavy deps)

    v = np.asarray(values, dtype=float)
    n = len(v)
    if n <= 1:
        return np.full(n, 0.5)
    return (average_rank(v) - 1.0) / (n - 1.0)


def select_brightness(
    df: pd.DataFrame,
    b_hat: np.ndarray,
    min_brightness: float,
    top_n: int,
    *,
    exempt: np.ndarray | None = None,
    llr: np.ndarray | None = None,
    llr_weight: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Attach `b_hat`, drop dim designs below the floor, rank by the brightness score, take top-N.

    `min_brightness` is the internal ×WT floor (0.6, stricter than the 0.3 competition cutoff).
    `exempt` is an optional bool mask marking rows that bypass the hard floor — the high-mutation
    upside slot, where docs/experiments §6 shows the absolute 0.6× gate over-rejects and viability
    is instead carried by the dead/alive + fold_sanity checks. Exempt rows are still ranked, never
    floor-killed.

    The top-N *cut* ranks by ``(1 − llr_weight)·pct(B̂) + llr_weight·pct(LLR)`` (PROJECT_PLAN §7-04в:
    B̂ primary + a weak ESMC-6B LLR voice). `llr_weight=0` (default) ⇒ pure B̂ order, identical to the
    previous behavior. Off-length designs whose `llr` is NaN keep their B̂ standing (unbiased). The
    stored `b_hat` column is always the ×WT magnitude — only the cut order uses the blend, so the
    downstream `R = B̂ × σ` magnitude is untouched. Returns kept rows (with `b_hat`) + a count dict.
    """
    if len(b_hat) != len(df):
        raise ValueError(f"b_hat length {len(b_hat)} != candidates length {len(df)}")
    b_hat = np.asarray(b_hat, dtype=float)
    exempt = (
        np.zeros(len(df), dtype=bool) if exempt is None else np.asarray(exempt, dtype=bool)
    )

    out = df.copy()
    out["b_hat"] = b_hat
    score = _rank_pct(b_hat)
    w = float(llr_weight)
    if llr is not None and w > 0.0:
        llr = np.asarray(llr, dtype=float)
        llr_pct = _rank_pct(llr)
        nan = ~np.isfinite(llr)
        llr_pct[nan] = score[nan]  # no LLR (off-length design) → keep B̂ standing, no penalty
        score = (1.0 - w) * score + w * llr_pct
    out["_score"] = score

    passes_floor = b_hat >= float(min_brightness)
    keep_mask = passes_floor | exempt
    n_floored = int((~passes_floor & ~exempt).sum())
    n_exempt_kept = int((~passes_floor & exempt).sum())

    kept = out.loc[keep_mask].sort_values(
        ["_score", "id"], ascending=[False, True], kind="stable"
    ).head(int(top_n)).reset_index(drop=True).drop(columns=["_score"])

    counts = {
        "n_in": int(len(df)),
        "n_floored": n_floored,
        "n_exempt_kept": n_exempt_kept,
        "n_out": int(len(kept)),
    }
    return kept, counts


def _load_maskedlm(model_id: str, dtype: str = "float16"):
    """Load `ESMCForMaskedLM` (HF fork) for a fused logits+hidden forward — module-local.

    Mirrors `synbio.esmc.load_esmc` but keeps the masked-LM head (that loader uses the headless
    `ESMCModel`). Same `device_map="auto"` to dodge the meta-tensor `.to()` failure; n_layers from
    a 1-token forward. Returns (model, tokenizer, n_layers).
    """
    import torch
    from transformers import AutoTokenizer, ESMCForMaskedLM

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = ESMCForMaskedLM.from_pretrained(
        model_id, torch_dtype=getattr(torch, dtype), device_map="auto"
    ).eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        enc = tokenizer(["M"], return_tensors="pt").to(device)
        n_layers = len(model(**enc, output_hidden_states=True).hidden_states) - 1
    logger.info("loaded ESMCForMaskedLM %s: %d layers, dtype=%s", model_id, n_layers, dtype)
    return model, tokenizer, n_layers


def _embed_ppl_llr(
    model, tokenizer, sequences: list[str], layer: int, aromatic_indices: list[int],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fused forward → (emb [N,d] L`layer`-aromatic-pooled fp32, ppl [N] OFS, llr [N] wt-marginal).

    `sequences[0]` is sfGFP (the WT reference for both the PPL anchor and the LLR). One
    `output_hidden_states=True` pass per batch yields the brightness embedding (hidden_states[layer]
    pooled at the chromophore-shell `aromatic_indices`, matching the stage-02 probe's
    `pool="aromatic"`), the OFS pseudo-perplexity logits, and — reusing sfGFP's own logits — the
    wt-marginal LLR of every design vs sfGFP. `llr[0]` (sfGFP vs itself) = 0; off-length designs
    (not colinear with sfGFP) get NaN. Heavy; cluster-only.
    """
    import torch

    from synbio.esmc.extract import pool_indices  # shared pooling helper (read-only reuse)

    # Both reads below assume residue p sits at token p (cls at 0): aromatic `pool_indices` uses
    # absolute token indices, and the PPL/LLR span is [1, n_real-1). That holds ONLY under right-
    # padding — left-padding would shift every non-longest sequence's frame and corrupt all three.
    # The stage-02 probe was trained on fixed-length (unpadded) data, so right-padding matches it.
    tokenizer.padding_side = "right"
    device = next(model.parameters()).device
    embs: list[np.ndarray] = []
    ppls: list[float] = []
    llrs: list[float] = []
    wt_logits: np.ndarray | None = None  # sfGFP logits [L, vocab] + span, captured from row 0
    wt_ids: np.ndarray | None = None
    wt_n_real = 0
    gidx = 0  # global sequence index (0 = sfGFP)
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        enc = tokenizer(batch, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        hidden = out.hidden_states[layer].float().cpu().numpy()  # [B, L, d]
        logits = out.logits.float().cpu().numpy()                # [B, L, vocab]
        ids = enc["input_ids"].cpu().numpy()                     # [B, L]
        mask = enc["attention_mask"].cpu().numpy()               # [B, L]
        embs.append(pool_indices(hidden, aromatic_indices))
        for b in range(len(batch)):
            n_real = int(mask[b].sum())  # [<cls> r1..rLp <eos>] → residue span is [1, n_real-1)
            ppls.append(float(np.exp(-pseudo_log_likelihood(logits[b], ids[b], 1, n_real - 1))))
            if gidx == 0:  # sfGFP: the WT reference for every design's LLR
                wt_logits, wt_ids, wt_n_real = logits[b].copy(), ids[b].copy(), n_real
                llrs.append(0.0)
            elif n_real == wt_n_real:  # colinear (same length) → wt-marginal LLR is defined
                llrs.append(wt_marginal_llr(wt_logits, wt_ids, ids[b], 1, n_real - 1))
            else:  # off-length design: not colinear with sfGFP, LLR undefined
                llrs.append(float("nan"))
            gidx += 1
    return (np.concatenate(embs, axis=0), np.asarray(ppls, dtype=float),
            np.asarray(llrs, dtype=float))


@register_stage(SPEC)
def run(cfg: StageConfig) -> StageResult:
    from synbio.io.artifacts import read_candidates, write_candidates
    from synbio.probes import CalibratedRidge

    p = cfg.params
    out_dir = Path(cfg.stage_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_candidates(cfg.inputs["candidates"]).reset_index(drop=True)

    probe_dir = Path(cfg.inputs["brightness_probe"])
    meta = json.loads((probe_dir / "meta.json").read_text())
    assert_linearization(meta)
    probe = CalibratedRidge.load(probe_dir / "ridge.npz")
    layer = int(meta["layer"])
    aromatic = list(meta["aromatic_indices"])
    anchor = float(meta["sfgfp_anchor"])

    sfgfp_seq = json.loads((Path(cfg.inputs["wt"]) / "wt.json").read_text())["sequence"]

    # Off-scaffold guard: the L24-aromatic probe pools at canonical residue positions (max 223,
    # Met=1). A design shorter than that has no residue there → the aromatic pool would read its
    # <eos>/padding and the probe can't score it. Drop those upfront (also keeps batch pooling
    # valid: every survivor has all aromatic tokens on real residues). Bulk near-WT designs (~238
    # aa) pass; this only catches off-length / heavily-indelled designs the probe can't handle.
    max_arom = max(aromatic)
    n_in = int(len(df))
    on_scaffold = df["sequence"].str.len().to_numpy() >= max_arom
    n_off = int((~on_scaffold).sum())
    df = df.loc[on_scaffold].reset_index(drop=True)

    # One fused forward over sfGFP (PPL + LLR anchor) + every surviving candidate.
    model, tokenizer, _ = _load_maskedlm(p["esmc_6b_model_id"])
    emb, ppl, llr = _embed_ppl_llr(
        model, tokenizer, [sfgfp_seq] + df["sequence"].tolist(), layer, aromatic,
        int(p["batch_size"]),
    )
    sfgfp_ppl, cand_ppl = float(ppl[0]), ppl[1:]
    cand_emb = emb[1:]

    b_hat = brightness_xwt(probe.predict(cand_emb), anchor)
    df["b_hat"] = b_hat
    df["ppl"] = cand_ppl
    df["llr"] = llr[1:]  # wt-marginal ESMC-6B LLR vs sfGFP (weak ranking voice, §7-04в)

    # (a) foldability PPL gate (all candidates; the sampler slot is its main target).
    ppl_keep = ppl_gate(cand_ppl, sfgfp_ppl, float(p["ppl_margin"]))
    n_ppl_dropped = int((~ppl_keep).sum())
    survivors = df.loc[ppl_keep].reset_index(drop=True)

    # (b+c) 0.6× floor + rank top-N by B̂ blended with the weak LLR voice; upside slot bypasses the
    # floor only (not the PPL gate). The stored `b_hat` stays the ×WT magnitude for R = B̂ × σ.
    llr_weight = float(p.get("llr_weight", 0.0))
    exempt = upside_mask(survivors["source"], tuple(p.get("upside_source_tokens", ("sampler",))))
    kept, counts = select_brightness(
        survivors, survivors["b_hat"].to_numpy(), float(p["min_brightness"]),
        int(p["top_n"]), exempt=exempt, llr=survivors["llr"].to_numpy(), llr_weight=llr_weight,
    )

    out_path = out_dir / "candidates_bright.parquet"
    write_candidates(kept, out_path)

    return StageResult(
        outputs={"candidates_bright": "candidates_bright.parquet"},
        decisions=[
            Decision(name="off_scaffold", threshold=f"len >= {max_arom} (aromatic shell)",
                     kept=int(on_scaffold.sum()), dropped=n_off,
                     note="probe can't pool the chromophore shell on shorter designs"),
            Decision(name="ppl_gate", threshold=f"<= {p['ppl_margin']}× sfGFP (={sfgfp_ppl:.2f})",
                     kept=int(ppl_keep.sum()), dropped=n_ppl_dropped,
                     note="single-pass OFS foldability gate"),
            Decision(name="brightness_floor", threshold=f">= {p['min_brightness']}× WT",
                     kept=counts["n_out"], dropped=counts["n_floored"],
                     note=f"{counts['n_exempt_kept']} upside rows bypassed the floor"),
            Decision(name="rank_top_n",
                     threshold=f"top {p['top_n']} by B̂ + {llr_weight:g}×LLR voice",
                     kept=counts["n_out"],
                     note="×WT magnitude preserved; LLR reorders the cut only"),
        ],
        metrics={
            "n_in": n_in, "n_off_scaffold": n_off, "sfgfp_ppl": sfgfp_ppl,
            "n_ppl_dropped": n_ppl_dropped, "n_floored": counts["n_floored"],
            "n_exempt_kept": counts["n_exempt_kept"], "n_out": counts["n_out"],
            "llr_weight": llr_weight,
        },
    )


if __name__ == "__main__":
    cli()
