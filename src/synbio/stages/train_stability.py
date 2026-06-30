"""Stage 02b: sequence-only ΔΔG probe — frozen ESMC-6B penultimate ridge on Megascale (Ч1).

Paper-exact A.1.4.4: predict absolute folding stability dG_ML per sequence; derive ΔΔG by
subtraction. Emits ΔΔG_funnel = dG_pred(wt) - dG_pred(mut) (positive = destabilizing).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from synbio.orchestrator import register_stage
from synbio.orchestrator.stage import Decision, StageConfig, StageResult, StageSpec, cli

if TYPE_CHECKING:
    from synbio.probes import RidgeProbe

logger = logging.getLogger(__name__)

SPEC = StageSpec(
    name="train_probes.stability",
    module="train_stability",
    env="esm",
    inputs=("wt", "gfps_wt", "megascale"),
    outputs=("stability_probe",),
)


def _ddg_funnel(probe: "RidgeProbe", X: np.ndarray, seq2row: dict[str, int],
                pairs: pd.DataFrame) -> np.ndarray:
    """ΔΔG_funnel = dG_pred(wt) - dG_pred(mut), aligned to `pairs` rows (positive = destabilizing)."""
    wt_rows = [seq2row[s] for s in pairs["wt_seq"]]
    mut_rows = [seq2row[s] for s in pairs["mut_seq"]]
    return probe.predict(X[wt_rows]) - probe.predict(X[mut_rows])


def _mean_per_domain_spearman(pairs: pd.DataFrame, ddg_pred: np.ndarray,
                              min_count: int) -> float:
    """Mean over domains of Spearman(ddG_ML, ΔΔG_funnel); 0.0 if no domain qualifies."""
    from synbio.probes import per_group_spearman

    d = per_group_spearman(pairs["ddG_ML"].to_numpy(float), ddg_pred,
                           pairs["WT_name"].to_numpy(), min_count)
    vals = [v for v in d.values() if v is not None]
    return float(np.mean(vals)) if vals else 0.0


def fit_stability_probe(
    seq_table: pd.DataFrame,
    X: np.ndarray,
    pairs: pd.DataFrame,
    params: dict,
) -> tuple["RidgeProbe", dict]:
    """Fit the dG ridge (α tuned on `val`, final fit on `train`); evaluate ΔΔG on `test`.

    Returns (probe predicting absolute dG, report). Pure numpy/pandas — synthetic-embedding testable.
    """
    from synbio.probes import fit_ridge, pearson, spearman

    seq2row = {s: i for i, s in enumerate(seq_table["sequence"])}
    dG = seq_table["dG"].to_numpy(dtype=float)
    split = seq_table["split"].to_numpy()
    train = split == "train"
    val_pairs = pairs[pairs["split"] == "val"]
    test_pairs = pairs[pairs["split"] == "test"]
    min_count = int(params.get("min_domain_pairs", 10))

    best_alpha, best_score, best_probe = None, -np.inf, None
    for alpha in params["alpha_grid"]:
        probe_a = fit_ridge(X[train], dG[train], float(alpha))
        if len(val_pairs):
            score = _mean_per_domain_spearman(val_pairs, _ddg_funnel(probe_a, X, seq2row, val_pairs),
                                              min_count)
        else:
            score = 0.0
        if score > best_score:
            best_alpha, best_score, best_probe = float(alpha), score, probe_a
    if len(val_pairs) == 0:
        logger.warning(
            "alpha selection uninformative: val_pairs is empty, every grid score was 0.0; "
            "defaulted to first grid entry alpha=%s", best_alpha,
        )
    probe = best_probe  # already fit on train with best_alpha

    ddg_test = _ddg_funnel(probe, X, seq2row, test_pairs)
    y_test = test_pairs["ddG_ML"].to_numpy(dtype=float)
    # sign sanity: most-destabilizing decile should project ΔΔG_funnel > 0
    if len(y_test):
        thr = np.quantile(y_test, 0.9)
        destab = y_test >= thr
        sign_frac = float(np.mean(ddg_test[destab] > 0)) if destab.any() else 0.0
    else:
        sign_frac = 0.0

    report = {
        "alpha": best_alpha,
        "per_domain_test_spearman": _mean_per_domain_spearman(test_pairs, ddg_test, min_count),
        "pooled_test_spearman": spearman(y_test, ddg_test) if len(y_test) else 0.0,
        "pooled_test_pearson": pearson(y_test, ddg_test) if len(y_test) else 0.0,
        "sign_frac": sign_frac,
        "n_train": int(train.sum()),
        "n_val": int((split == "val").sum()),
        "n_test": int((split == "test").sum()),
    }
    return probe, report


@register_stage(SPEC)
def run(cfg: StageConfig) -> StageResult:
    from synbio.esmc import EmbeddingCache, embed_sequences, load_esmc
    from synbio.probes import build_gfp_reversion_panel, build_stability_dataset, read_multi_fasta

    p = cfg.params
    out_dir = Path(cfg.stage_dir) / "stability_probe"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(cfg.inputs["megascale"])
    seq_table, pairs = build_stability_dataset(df)

    handle = load_esmc(p["esmc_6b_model_id"])
    stab_layer = int(p["stability_layer"])
    layer = handle.n_layers + 1 + stab_layer if stab_layer < 0 else stab_layer  # -2 -> n_layers-1
    pool = p["pool"]
    cache = EmbeddingCache(p["embed_cache_dir"], model_tag="esmc6b", layer=layer, pool=pool)
    X = embed_sequences(handle, seq_table["sequence"].tolist(), layer, int(p["batch_size"]),
                        cache, pool)

    probe, report = fit_stability_probe(seq_table, X, pairs, p)

    # GFP transfer sanity (non-blocking, config-gated). Both directions guard against a one-sided
    # bias: sfGFP→avGFP reversions should destabilize, avGFP→sfGFP should stabilize.
    gfp_sanity: dict = {}
    if p.get("gfp_sanity", True):
        try:
            sfgfp = json.loads((Path(cfg.inputs["wt"]) / "wt.json").read_text())["sequence"]
            avgfp = read_multi_fasta(cfg.inputs["gfps_wt"])["avGFP"]
            rev = build_gfp_reversion_panel(sfgfp, avgfp)  # sfGFP→avGFP (expect destabilizing)
            fwd = build_gfp_reversion_panel(avgfp, sfgfp)  # avGFP→sfGFP (expect stabilizing)
            emb = embed_sequences(handle, [sfgfp, avgfp] + rev + fwd, layer, int(p["batch_size"]),
                                  cache, pool)
            dg = probe.predict(emb)
            dg_sf, dg_av = float(dg[0]), float(dg[1])
            ddg_rev = dg_sf - dg[2:2 + len(rev)]   # positive = destabilizing
            ddg_fwd = dg_av - dg[2 + len(rev):]    # negative = stabilizing
            gfp_sanity = {
                "n": len(rev), "frac_destabilizing": float(np.mean(ddg_rev > 0)),
                "sum_ddg": float(ddg_rev.sum()),                    # reversion sfGFP→avGFP
                "fwd_frac_stabilizing": float(np.mean(ddg_fwd < 0)),
                "fwd_sum_ddg": float(ddg_fwd.sum()),                # forward avGFP→sfGFP
                "whole_avgfp_to_sfgfp_ddg": dg_av - dg_sf,          # negative = sfGFP more stable
                "dg_pred_sfgfp": dg_sf, "dg_pred_avgfp": dg_av,
            }
        except (KeyError, ValueError) as e:
            logger.warning("GFP transfer sanity skipped: %s", e)
            gfp_sanity = {"skipped": str(e)}
    else:
        gfp_sanity = {"skipped": "disabled"}

    passed = report["per_domain_test_spearman"] >= float(p["spearman_gate"])
    meta = {
        "kind": "ridge_dg_6b", "model_tag": "esmc6b", "layer": layer, "pool": pool,
        "target": "dG_ML (folding stability, higher = more stable)",
        "emit": "ddG_funnel = dG_pred(wt) - dG_pred(mut)", "sign": "positive = destabilizing",
        "feature": "penultimate masked-mean-pool of a single sequence's ESMC-6B embedding",
        "alpha": report["alpha"], "alpha_grid": list(p["alpha_grid"]),
        "per_domain_test_spearman": report["per_domain_test_spearman"],
        "pooled_test_spearman": report["pooled_test_spearman"],
        "pooled_test_pearson": report["pooled_test_pearson"],
        "sign_frac": report["sign_frac"], "passed_gate": passed,
        "spearman_gate": float(p["spearman_gate"]),
        "n_train": report["n_train"], "n_val": report["n_val"], "n_test": report["n_test"],
        "gfp_sanity": gfp_sanity,
    }
    probe.meta.update(meta)
    probe.save(out_dir / "dir.npz")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    note = (f"L{layer}/{pool} per_domain_S={report['per_domain_test_spearman']:.3f} "
            f"pooled_S={report['pooled_test_spearman']:.3f} alpha={report['alpha']}")
    return StageResult(
        outputs={"stability_probe": "stability_probe"},
        decisions=[
            Decision(name="probe", note=note),
            Decision(name="spearman_gate", threshold=f">={p['spearman_gate']}",
                     note="passed" if passed else "below_threshold"),
        ],
        metrics={"per_domain_test_spearman": report["per_domain_test_spearman"],
                 "pooled_test_spearman": report["pooled_test_spearman"],
                 "sign_frac": report["sign_frac"], "passed_gate": passed},
    )


if __name__ == "__main__":
    cli()
