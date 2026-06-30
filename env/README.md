# Containerless envs (Aldan3)

enroot on Aldan3 is a pyxis-only plugin, and after a SLURM update any
`--container-image` job dies with `TaskProlog failed status=127` before the command
runs (the cluster's `TaskProlog` now executes *inside* the container namespace, where
`/opt/aldan3` and `loginctl` don't exist). The admin confirmed it's an unfixable
SLURM+enroot bug, so we run the tools **without containers**.

The `synrun <env> <cmd>` interface, `synenv.sh` variables, and the `docker/repos/*`
layout are unchanged — the orchestrator and every stage keep working byte-for-byte,
because they only ever call `synrun`.

## Layout

```
env/
├── bootstrap.sh        # build the 7 envs into the shared root (run once)
├── synenv.sh           # source this; sets MAMBA_ROOT_PREFIX + caches
├── bin/
│   ├── micromamba      # fetched by bootstrap.sh
│   └── synrun          # micromamba run -n <env>, repos on PYTHONPATH
└── mm/                 # MAMBA_ROOT_PREFIX — the 7 envs (read-only, shared)
cache/                  # shared download-once weight cache (HF / torch)
```

**One shared install for the whole team** — no per-user duplicates. The envs under
`env/mm` are read-only (`a+rX`, `go-w`); everyone reads them concurrently. Per-user
*writable* state (tmp, triton/numba/torch compile caches) lives under `$WORKDIR`
(default `$HOME/synbio-ws`). Big model weights download **once** into the shared
`cache/` (HF_HOME / TORCH_HOME), matching the "кэш HF общий" rule.

## One-time setup (one person, node with network)

```bash
bash env/bootstrap.sh                 # all 7 envs; ~tens of GB, takes a while
# optional: give the shared tree to the team unix group
SYNBIO_GROUP=dogma bash env/bootstrap.sh
```

Retry a single env without touching the rest:

```bash
SYNBIO_FORCE=1 bash env/bootstrap.sh spurs
```

## Daily use

```bash
bash start_synbio_NOCONTAINER.sh      # plain GPU srun -> shell with synrun ready
# then, exactly as before:
synrun esm      python -c "import esm; print(esm.__version__)"
synrun dnatools python -m synbio.stages.prep_wt run --cfg …
synrun esm      python run/pipeline/run_pipeline.py …
```

Ad-hoc (already on a GPU node): `source env/synenv.sh` then use `synrun`.

## Notes

- micromamba is a package manager, not a container: plain directories of binaries,
  no enroot/namespaces. GPU works via the node's NVIDIA driver directly; the torch
  wheels (cu121 / cu113 for spurs) carry their own userspace CUDA.
- conda envs aren't relocatable — the shared root is built in place at this path and
  must not be moved. All users see the same `/projects/...` path, so this is a no-op
  constraint here.
- This is **not** the squashfuse rootfs that died mid-job; these are ordinary files,
  so that failure mode is gone. CephFS small-file latency makes the first import on a
  cold node slow; stage a hot env to node-local `/scratch` only if a tight loop needs it.
- `$WORKDIR` per user; override with `SYNBIO_WORKDIR=/scratch/$USER` for node-local speed.
