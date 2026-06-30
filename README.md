# SynBio 2026 — GFP "all-around champion" pipeline

A cascade funnel that designs 6 green fluorescent protein (GFP) sequences (220–250 aa)
that stay **bright** in cell-free expression and **thermostable** after 10 min at 72 °C.

The competition score `S_B × S_T` algebraically collapses to `F_final / F_initial_WT` —
absolute post-heat brightness — so the funnel optimizes a **single objective**:
stability-engineer a bright **near-WT sfGFP** scaffold. Brightness mutations come from
the Sarkisyan DMS plus a TGP-style **surface-recharge** layer on DMS-tolerant positions,
keeping the mutation budget small. Brightness is ranked with **calibrated magnitude**
(`B̂`, in ×WT); stability is **rank-only** (out-of-distribution at 72 °C, dominated by
irreversible aggregation, so it is engineered constructively rather than scored).

## Requirements

This targets a **Linux machine with an NVIDIA GPU** (developed on 2× V100S 32 GB) — a SLURM
cluster or baremetal. The upstream tools under `docker/repos/*` are pinned git submodules.

Model weights are **not committed**. Pulled from **HuggingFace** at runtime: **ESMC-6B**,
**ESMC-600M**, **ESMFold2**, and **SPURS**. The non-HuggingFace checkpoints — **ProteinMPNN**,
**LigandMPNN**, and **ThermoMPNN** — go under `weights/` (gitignored).

```bash
git clone --recursive https://github.com/f-normies/synbiochallenges2026-public
# or, if you already cloned without --recursive:
git submodule update --init
```

## Setup & run — containerless native envs (recommended)

The tools run in shared, read-only micromamba envs built once by `env/bootstrap.sh`
(no containers required). See `env/README.md` for details and per-env retries.

```bash
# 1) One-time, on a node with network access (~tens of GB, takes a while):
bash env/bootstrap.sh

# 2) Open an interactive GPU session (SLURM srun; REPO auto-derives from the clone):
bash start_synbio.sh
#    (quick tests: edit --partition=long → --partition=short)

# 3) Inside the session, run the full DAG (orchestrator in the dnatools env):
synrun dnatools python run/pipeline/run_pipeline.py

# A single stage (Hydra override):
synrun dnatools python run/pipeline/run_pipeline.py stages='[prep_wt]'

# Force re-run (otherwise completed stages are skipped on resume):
synrun dnatools python run/pipeline/run_pipeline.py +force=true
```

Outputs: `submission.csv` (6 sequences), the run manifest / decision tree, and per-stage
logs under `workspace/runs/<run_id>/`.

> **Note on paths/scheduler.** `start_synbio.sh` assumes a SLURM cluster with a `gpu`
> constraint. Adjust `--partition`, `--constraint`, and `--gres` for your scheduler.
> `env/synenv.sh` derives all paths from the clone location, so no path editing is needed
> beyond your SLURM flags.

## Run without SLURM (baremetal GPUs)

`synrun` is just a micromamba runner — it has **no SLURM dependency**. On a machine where
GPUs are directly accessible, skip `start_synbio.sh` and source the env yourself:

```bash
# 1) One-time, on a machine with network + GPU (~tens of GB):
bash env/bootstrap.sh

# 2) Source the env — puts `synrun` on PATH; no srun, no SLURM:
#    env/synenv.sh defaults to HuggingFace-offline (compute nodes are usually
#    internet-less). The first run must download ESMC-6B / ESMFold2, so opt back
#    online BEFORE sourcing — the `:=` guards make the first value win:
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
source env/synenv.sh

# 3) Run the full DAG directly (pin to one GPU; ESMC/ESMFold2 use a single card):
CUDA_VISIBLE_DEVICES=0 synrun dnatools python run/pipeline/run_pipeline.py
```

After the weights are cached you can drop the two `HF_*OFFLINE=0` exports (offline is the
default and avoids slow etag checks). `env/synenv.sh` routes per-user writable state to
`$WORKDIR` (default `$HOME/synbio-ws`) and the shared weight cache to `cache/` under the
clone; override with `SYNBIO_WORKDIR` / `SYNBIO_SHARED_CACHE` if needed.

## Alternative — container image

On a cluster where enroot/pyxis (or Docker) works, build the image from `docker/` instead
of the native envs; the `synrun <env> <cmd>` interface is identical either way.

```bash
bash docker/build.sh
bash docker/smoke_test.sh            # verify per-env imports (add --gpu / --esmfold2)
```

## Local development (orchestrator core only, no GPU/containers)

The orchestrator, probe logic, and IO are pure-Python and unit-testable without GPUs or
the heavy envs:

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.10
pip install -e ".[dev]"
```
