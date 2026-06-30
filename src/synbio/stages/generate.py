"""Stage 03: candidate generation mini-DAG (owned by P5/P2).

Four nodes across three envs: DMS-directed combinatorics + LigandMPNN + ESMC sampler,
then a merge into the carrier candidates.parquet. All near-WT.
"""

from synbio.orchestrator import register_stage
from synbio.orchestrator.stage import Decision, StageConfig, StageResult, StageSpec, cli
from synbio.utils.logging import get_logger

logger = get_logger(__name__)

SPEC_LIGANDMPNN = StageSpec(
    name="generate.ligandmpnn",
    module="generate",
    env="ligandmpnn",
    inputs=("wt", "position_tolerance", "sfgfp_sasa"),
    outputs=("cand_ligandmpnn",),
)
SPEC_COMBINATORIAL = StageSpec(
    name="generate.combinatorial",
    module="generate",
    env="dnatools",
    inputs=(
        "wt", "do_not_mutate", "position_tolerance", "sarkisyan_dms",
        "gfps_wt", "previous_top_sequences", "sfgfp_sasa",
    ),
    outputs=("cand_combinatorial",),
)
SPEC_SAMPLER = StageSpec(
    name="generate.sampler",
    module="generate",
    env="esm",
    inputs=("wt", "position_tolerance", "brightness_probe"),
    outputs=("cand_sampler",),
)
SPEC_MERGE = StageSpec(
    name="generate.merge",
    module="generate",
    env="dnatools",
    inputs=("cand_ligandmpnn", "cand_combinatorial", "cand_sampler", "exclusion_list"),
    outputs=("candidates",),
)


def _invoke(argv: list[str], cwd: str) -> None:
    """Subprocess seam: run LigandMPNN run.py. Monkeypatched in tests."""
    import subprocess

    subprocess.run(argv, cwd=cwd, check=True)


@register_stage(SPEC_LIGANDMPNN)
def run_ligandmpnn(cfg: StageConfig) -> StageResult:
    import json
    import math
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from synbio.generate import (
        PROVENANCE_COLUMNS,
        bucket_of,
        build_argv,
        design_row,
        exposed_positions,
        load_sasa,
        modeled_resids,
        mutations_from_diff,
        parse_fasta,
        pick_subset,
        recharge_bias,
        reconstruct_full,
        redesignable_positions,
        residue_tokens,
        rows_to_frame,
    )
    from synbio.io.artifacts import BASE_COLUMNS
    from synbio.io.constraints import validate_sequence

    p = cfg.params
    stage_dir = Path(cfg.stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    wt_dir = Path(cfg.inputs["wt"])
    wt_seq = json.loads((wt_dir / "wt.json").read_text())["sequence"]
    wt_pdb = wt_dir / "wt.pdb"
    free = set(json.loads(Path(cfg.inputs["position_tolerance"]).read_text())["free_mutate_positions"])
    sasa = load_sasa(cfg.inputs["sfgfp_sasa"])

    logger.info("ligandmpnn: building redesignable set + exposed (recharge) set")
    modeled = modeled_resids(wt_pdb)
    redesign = redesignable_positions(sorted(free), modeled)
    threshold = float(p.get("sasa_exposed_threshold", 0.25))
    exposed_set = exposed_positions(sasa, threshold) & set(redesign)

    # Absolute paths so the call is cwd-independent: run.py adds its own dir to sys.path[0],
    # so its `import model_utils` works regardless of where we launch it from.
    run_script = str((Path(p["ligandmpnn_repo"]) / "run.py").resolve())
    ckpt = p.get("checkpoint", "")
    ckpt = str(Path(ckpt).resolve()) if ckpt else ""

    # Per-design redesign budget: each invocation redesigns a random subset of the tolerant
    # set (size ~redesign_min..redesign_max), keeping designs near-WT for slots 2 & 4.
    # Redesigning all ~179 tolerant positions at T=0.1 overshoots to ~100 mut (cluster-measured),
    # so we cap the per-design designable window instead.
    rng = np.random.default_rng(cfg.seed)
    batch_size = int(p.get("batch_size", 8))
    batches_per_subset = int(p.get("batches_per_subset", 4))
    per_subset = batch_size * batches_per_subset
    n_designs = int(p.get("n_designs", 1000))
    n_subsets = max(1, math.ceil(n_designs / per_subset))
    rmax = min(int(p.get("redesign_max", 65)), len(redesign))
    rmin = min(int(p.get("redesign_min", 26)), rmax)
    recharge_b = float(p.get("recharge_bias", 1.5))

    records = []
    for si in range(n_subsets):
        s = int(rng.integers(rmin, rmax + 1))
        subset = pick_subset(redesign, s, rng)
        exposed_sub = sorted(r for r in subset if r in exposed_set)
        sub_dir = stage_dir / f"subset_{si:03d}"
        sub_dir.mkdir(parents=True, exist_ok=True)
        bias_path = sub_dir / "bias_AA_per_residue.json"
        bias_path.write_text(json.dumps(recharge_bias(exposed_sub, recharge_b)))
        argv = build_argv(
            run_script=run_script, pdb_path=str(wt_pdb), out_folder=str(sub_dir / "out"),
            redesigned_residues=residue_tokens(subset), bias_json_path=str(bias_path),
            omit_aa=str(p.get("omit_aa", "C")), temperature=float(p.get("temperature", 0.1)),
            seed=int(cfg.seed) + si, batch_size=batch_size, number_of_batches=batches_per_subset,
            checkpoint=ckpt, model_type=str(p.get("model_type", "ligand_mpnn")),
            use_atom_context=int(p.get("ligand_mpnn_use_atom_context", 1)),
            use_side_chain_context=int(p.get("ligand_mpnn_use_side_chain_context", 1)),
        )
        _invoke(argv, cwd=None)  # all paths absolute; run from repo root
        for fa in sorted((sub_dir / "out" / "seqs").glob("*.fa")):
            records.extend(parse_fasta(fa))

    logger.info("ligandmpnn: %d subsets x %d designs -> %d raw records (subset size %d-%d)",
                n_subsets, per_subset, len(records), rmin, rmax)
    rows = []
    n_drop = 0
    for i, rec in enumerate(records, start=1):
        full = reconstruct_full(rec.sequence, modeled, wt_seq)
        ok, _ = validate_sequence(full, set(), require_nterm=True)
        if not ok:
            n_drop += 1
            continue
        muts = mutations_from_diff(wt_seq, full)
        rows.append(design_row(
            f"lmpnn_{i:06d}", full, wt_seq, muts, source="ligandmpnn", parent="sfGFP",
            pool_counts={}, bucket=bucket_of(len(muts)),
        ))

    if rows:
        df = rows_to_frame(rows)
    else:
        df = pd.DataFrame(columns=list(BASE_COLUMNS) + list(PROVENANCE_COLUMNS))
    out_rel = "cand_ligandmpnn.parquet"
    df.to_parquet(stage_dir / out_rel, index=False)

    return StageResult(
        outputs={"cand_ligandmpnn": out_rel},
        decisions=[
            Decision(name="redesign_budget", kept=n_subsets,
                     note=f"{n_subsets} subsets, size {rmin}-{rmax} of {len(redesign)} tolerant, "
                          f"{len(exposed_set)} exposed, thr={threshold}"),
            Decision(name="constraint_filter", kept=len(rows), dropped=n_drop,
                     threshold="len 220-250 / M-start / 20AA / N-term"),
        ],
        metrics={
            "n_designs": len(df), "n_in": len(records), "n_out": len(rows),
            "n_redesignable": len(redesign), "n_exposed": len(exposed_set),
            "n_subsets": n_subsets, "redesign_min": rmin, "redesign_max": rmax,
            "mean_hamming": float(df["hamming"].mean()) if rows else 0.0,
        },
    )


@register_stage(SPEC_COMBINATORIAL)
def run_combinatorial(cfg: StageConfig) -> StageResult:
    import json
    from pathlib import Path

    from synbio.generate import (
        BudgetConfig,
        brightness_pools,
        consensus_pool,
        design_row,
        load_sasa,
        read_fasta_records,
        recharge_singles,
        rows_to_frame,
        sample_designs,
    )

    p = cfg.params
    logger.info("loading prep_wt outputs + DMS/winners/SASA for combinatorial generation")
    wt = json.loads((Path(cfg.inputs["wt"]) / "wt.json").read_text())
    wt_seq = wt["sequence"]
    tol = json.loads(Path(cfg.inputs["position_tolerance"]).read_text())
    free = set(tol["free_mutate_positions"])
    dnm = {e["pos"] for e in json.loads(Path(cfg.inputs["do_not_mutate"]).read_text())["positions"]}
    sasa = load_sasa(cfg.inputs["sfgfp_sasa"])
    records = read_fasta_records(cfg.inputs["gfps_wt"])
    if "amacGFP" not in records:
        raise ValueError(
            f"amacGFP record not found in {cfg.inputs['gfps_wt']}; "
            f"available records: {sorted(records)}"
        )
    amac = records["amacGFP"]

    bright = brightness_pools(
        wt_seq, cfg.inputs["sarkisyan_dms"], cfg.inputs["previous_top_sequences"],
        free, dnm, eps=p.get("brightness_eps", 0.5),
        enhancing_margin=p.get("enhancing_margin", 0.3),
    )
    consensus = (
        consensus_pool(wt_seq, amac, free, dnm)
        if p.get("include_consensus", True) else []
    )
    recharge = recharge_singles(
        wt_seq, sasa, free, dnm,
        threshold=p.get("sasa_exposed_threshold", 0.25),
        targets=p.get("recharge_targets", {"K": ["E", "Q"], "R": ["Q"]}),
    )

    kb = p.get("k_bright", {"min": 0, "max": 5})
    kc = p.get("k_consensus", {"min": 0, "max": 4})
    kr = p.get("k_recharge", {"min": 0, "max": 8})
    budget = BudgetConfig(
        n_designs=p.get("n_designs", 800),
        max_mutations=p.get("max_mutations", 16),
        pocket_cap=p.get("pocket_cap", 8),
        k_bright=(int(kb["min"]), int(kb["max"])),
        k_consensus=(int(kc["min"]), int(kc["max"])),
        k_recharge=(int(kr["min"]), int(kr["max"])),
        seed_ladders=p.get("seed_ladders", True),
        low_bucket_max=p.get("low_bucket_max", 8),
        max_attempts_factor=p.get("max_attempts_factor", 50),
    )
    designs = sample_designs(wt_seq, bright, consensus, recharge, budget, cfg.seed)

    rows = [
        design_row(
            f"comb_{i:06d}", d.sequence, wt_seq, list(d.mutations),
            source="combinatorial", parent="sfGFP",
            pool_counts=d.pool_counts, bucket=d.bucket,
        )
        for i, d in enumerate(designs, start=1)
    ]
    df = rows_to_frame(rows)
    stage_dir = Path(cfg.stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    out_rel = "cand_combinatorial.parquet"
    df.to_parquet(stage_dir / out_rel, index=False)

    n_low = int((df["bucket"] == "low").sum())
    return StageResult(
        outputs={"cand_combinatorial": out_rel},
        decisions=[
            Decision(
                name="pools_built",
                note=(
                    f"bright={len(bright.proven)}+{len(bright.neutral_or_better)}, "
                    f"recharge={len(recharge)}, consensus={len(consensus)}"
                ),
            ),
            Decision(
                name="designs_generated",
                kept=len(df),
                note=f"{n_low} low / {len(df) - n_low} mid; deduped, seed={cfg.seed}",
            ),
            Decision(
                name="budget",
                note=f"pocket_cap={budget.pocket_cap}, k_recharge<={budget.k_recharge[1]}, "
                     f"max_mutations={budget.max_mutations}",
            ),
        ],
        metrics={
            "n_designs": len(df),
            "n_low": n_low,
            "n_mid": len(df) - n_low,
            "mean_hamming": float(df["hamming"].mean()),
            "pool_sizes": {
                "proven": len(bright.proven),
                "neutral_or_better": len(bright.neutral_or_better),
                "recharge": len(recharge),
                "consensus": len(consensus),
            },
        },
    )


def _build_logits_fn(model_id: str):
    """Seam: load ESMC-600M MLM and return a logits_fn for the Gibbs loop. Monkeypatched in tests."""
    from synbio.esmc.mlm import load_esmc_mlm, make_logits_fn

    return make_logits_fn(load_esmc_mlm(model_id))


def _score_brightness(seqs, probe_dir, model_id_6b, batch_size, cache_dir):
    """Seam: load ESMC-6B and score sequences with the stage-02 probe. Monkeypatched in tests."""
    from synbio.esmc import load_esmc
    from synbio.probes.brightness_score import score_brightness

    handle = load_esmc(model_id_6b)
    return score_brightness(seqs, probe_dir, handle, batch_size, cache_dir)


@register_stage(SPEC_SAMPLER)
def run_sampler(cfg: StageConfig) -> StageResult:
    import json
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from synbio.generate import (
        PROVENANCE_COLUMNS,
        bucket_of,
        design_row,
        gibbs_sample,
        mutations_from_diff,
        pick_subset,
        rows_to_frame,
        sampling_set,
    )
    from synbio.io.artifacts import BASE_COLUMNS
    from synbio.io.constraints import validate_sequence

    p = cfg.params
    stage_dir = Path(cfg.stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    wt_seq = json.loads((Path(cfg.inputs["wt"]) / "wt.json").read_text())["sequence"]
    free = json.loads(Path(cfg.inputs["position_tolerance"]).read_text())["free_mutate_positions"]
    probe_dir = cfg.inputs["brightness_probe"]
    sampling_positions = sampling_set(free, int(p.get("nterm_freeze", 10)))

    rng = np.random.default_rng(cfg.seed)
    logits_fn = _build_logits_fn(str(p["esmc_600m_model_id"]))
    n_over = int(p.get("n_over", 2000))
    mut_min, mut_max = int(p.get("mut_min", 40)), int(p.get("mut_max", 90))
    n_passes = int(p.get("n_passes", 3))
    temperature = float(p.get("temperature", 1.0))
    exclude_wt = bool(p.get("exclude_wt", True))
    omit_aa = frozenset(str(p.get("omit_aa", "C")))  # no free surface Cys (reducing CFPS, §4.3)

    logger.info("sampler: gibbs over-generation (n_over=%d, k=%d..%d, %d tolerant positions)",
                n_over, mut_min, mut_max, len(sampling_positions))
    seqs: list[str] = []
    for _ in range(n_over):
        k = int(rng.integers(mut_min, mut_max + 1))
        positions = pick_subset(sampling_positions, k, rng)
        s = gibbs_sample(wt_seq, positions, logits_fn, rng,
                         n_passes=n_passes, temperature=temperature, exclude_wt=exclude_wt,
                         omit=omit_aa)
        ok, _ = validate_sequence(s, set(), require_nterm=True)
        if ok:
            seqs.append(s)
    seqs = list(dict.fromkeys(seqs))  # dedup, preserve order

    n_emit = int(p.get("n_emit", 300))
    if seqs:
        b_hat = _score_brightness(
            seqs, probe_dir, str(p["esmc_6b_model_id"]),
            int(p.get("batch_size", 16)), p.get("embed_cache_dir", "workspace/embed_cache"),
        )
        order = list(np.argsort(-np.asarray(b_hat, dtype=float))[:n_emit])
    else:
        b_hat, order = np.array([]), []

    rows = []
    for rank, j in enumerate(order, start=1):
        s = seqs[int(j)]
        muts = mutations_from_diff(wt_seq, s)
        row = design_row(
            f"smp_{rank:06d}", s, wt_seq, muts, source="esmc600m_sampler",
            parent="sfGFP", pool_counts={}, bucket=bucket_of(len(muts)),
        )
        row["b_hat"] = float(b_hat[int(j)])
        rows.append(row)

    if rows:
        df = rows_to_frame(rows)
    else:
        df = pd.DataFrame(columns=list(BASE_COLUMNS) + list(PROVENANCE_COLUMNS) + ["b_hat"])
    out_rel = "cand_sampler.parquet"
    df.to_parquet(stage_dir / out_rel, index=False)

    return StageResult(
        outputs={"cand_sampler": out_rel},
        decisions=[
            Decision(name="over_generate", kept=len(seqs),
                     note=f"{n_over} gibbs draws, k={mut_min}-{mut_max}, T={temperature}"),
            Decision(name="brightness_rerank", kept=len(rows),
                     threshold=f"top {n_emit} by B_hat"),
        ],
        metrics={
            "n_designs": len(df), "n_over": n_over, "n_valid": len(seqs), "n_out": len(rows),
            "mean_hamming": float(df["hamming"].mean()) if rows else 0.0,
        },
    )


@register_stage(SPEC_MERGE)
def run_merge(cfg: StageConfig) -> StageResult:
    from pathlib import Path

    from synbio.generate.merge import merge_fragments, read_exclusion
    from synbio.io.artifacts import read_candidates, write_candidates

    stage_dir = Path(cfg.stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    # priority order: combinatorial (near-WT carrier) -> ligandmpnn -> sampler
    frames = [
        read_candidates(cfg.inputs["cand_combinatorial"]),
        read_candidates(cfg.inputs["cand_ligandmpnn"]),
        read_candidates(cfg.inputs["cand_sampler"]),
    ]
    exclusion = read_exclusion(cfg.inputs["exclusion_list"])
    df, stats = merge_fragments(frames, exclusion)

    out_rel = "candidates.parquet"
    write_candidates(df, stage_dir / out_rel)
    buckets = df["bucket"].value_counts().to_dict() if "bucket" in df.columns else {}

    return StageResult(
        outputs={"candidates": out_rel},
        decisions=[
            Decision(name="dedup", dropped=stats["n_dup"],
                     note="exact sequence dedup, keep-first (combinatorial>ligandmpnn>sampler)"),
            Decision(name="exclusion_filter", kept=stats["n_out"], dropped=stats["n_excluded"],
                     threshold="exact match vs Exclusion_List"),
        ],
        metrics={**stats, "buckets": buckets},
    )


if __name__ == "__main__":
    cli()
