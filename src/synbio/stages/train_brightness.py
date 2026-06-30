"""Stage 02a: brightness probe — frozen CalibratedRidge on L24-aromatic avGFP embeddings (P3)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from synbio.orchestrator import register_stage
from synbio.orchestrator.stage import Decision, StageConfig, StageResult, StageSpec, cli

if TYPE_CHECKING:
    from synbio.probes import CalibratedRidge

SPEC = StageSpec(
    name="train_probes.brightness",
    module="train_brightness",
    env="esm",
    inputs=("wt", "gfps_wt", "sarkisyan_dms"),
    outputs=("brightness_probe",),
)

_LOCAL_POOLS = ("chromo", "pocket", "aromatic")


def fit_brightness_probe(
    emb: np.ndarray,
    table: pd.DataFrame,
    params: dict,
    seed: int,
) -> tuple[CalibratedRidge, dict, bool, np.ndarray]:
    """Pure: split -> CalibratedRidge.fit -> regime_report -> gate (synthetic emb).

    Returns (CalibratedRidge, report, passed, test_idx).
    """
    from synbio.probes import (
        fit_calibrated_ridge,
        mutation_count_stratified_split,
        regime_gate,
        regime_report,
    )

    y = table["y"].to_numpy(dtype=float)
    train_idx, test_idx = mutation_count_stratified_split(
        table, float(params["test_frac"]), seed, bool(params["hamming1_guard"])
    )
    probe = fit_calibrated_ridge(
        emb[train_idx], y[train_idx], float(params["ridge_alpha"]),
        float(params["dead_weight"]), float(params["live_lo"]),
    )
    report = regime_report(y[test_idx], probe.predict(emb[test_idx]))
    passed = regime_gate(report, float(params["bright_spearman_gate"]))
    return probe, report, passed, test_idx


@register_stage(SPEC)
def run(cfg: StageConfig) -> StageResult:
    from synbio.esmc import EmbeddingCache, embed_sequences, load_esmc
    from synbio.esmc.pocket import resolve_indices
    from synbio.probes import build_brightness_dataset, read_multi_fasta

    p = cfg.params
    out_dir = Path(cfg.stage_dir) / "brightness_probe"
    out_dir.mkdir(parents=True, exist_ok=True)

    refs = read_multi_fasta(cfg.inputs["gfps_wt"])
    df = pd.read_csv(cfg.inputs["sarkisyan_dms"])
    backgrounds = list(p["backgrounds"])
    table, dropped, wt_bright = build_brightness_dataset(
        df, refs, backgrounds, int(p["mut_numbering_offset"])
    )
    sfgfp_seq = json.loads((Path(cfg.inputs["wt"]) / "wt.json").read_text())["sequence"]

    handle = load_esmc(p["esmc_6b_model_id"])
    layer = round(float(p["brightness_layer_frac"]) * handle.n_layers)
    pool = p["pool"]
    local_idx = resolve_indices(refs[backgrounds[0]])[pool] if pool in _LOCAL_POOLS else None
    cache = EmbeddingCache(p["embed_cache_dir"], model_tag="esmc6b", layer=layer, pool=pool)
    seqs = table["sequence"].tolist()
    emb = embed_sequences(handle, seqs, layer, int(p["batch_size"]), cache, pool, local_idx)

    probe, report, passed, _ = fit_brightness_probe(emb, table, p, cfg.seed)
    anchor = float(
        probe.predict(embed_sequences(handle, [sfgfp_seq], layer, 1, cache, pool, local_idx))[0]
    )

    meta = {
        "kind": "calibrated_ridge", "layer": layer, "pool": pool,
        "aromatic_indices": list(local_idx) if local_idx is not None else None,
        "backgrounds": backgrounds, "wt_brightness_by_type": wt_bright,
        "sfgfp_anchor": anchor, "linearization": "10**(predict(b) - sfgfp_anchor)",
        "regime": report, "passed_gate": passed,
        "bright_spearman_gate": float(p["bright_spearman_gate"]),
        "dead_weight": float(p["dead_weight"]), "live_lo": float(p["live_lo"]),
        "n_dropped": dropped, "model_tag": "esmc6b",
    }
    probe.meta.update(meta)
    probe.save(out_dir / "ridge.npz")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    note = (f"L{layer}/{pool} bright_S={report['bright']['spearman']:.3f} "
            f"live_S={report['live']['spearman']:.3f}")
    return StageResult(
        outputs={"brightness_probe": "brightness_probe"},
        decisions=[
            Decision(name="probe", note=note),
            Decision(name="bright_spearman_gate", threshold=f">={p['bright_spearman_gate']}",
                     note="passed" if passed else "below_threshold"),
        ],
        metrics={"bright_spearman": report["bright"]["spearman"],
                 "live_spearman": report["live"]["spearman"],
                 "passed_gate": passed, "n_dropped": dropped},
    )


if __name__ == "__main__":
    cli()
