# Sourceable. Containerless port of docker/synenv.sh.
#
# Selects the SHARED, read-only micromamba root and the SHARED download-once weight
# cache, and routes every write-on-run path either to that shared cache (weights) or
# to a per-user $WORKDIR (tmp + compile caches). Source me, then use
# `synrun <env> <cmd>` exactly as inside the old container. No enroot, no namespaces.

# --- locate the shared install (this file lives at $REPO/env/synenv.sh) -------
_synenv_self="${BASH_SOURCE[0]:-$0}"
SYNBIO_ROOT="$(cd "$(dirname "$_synenv_self")/.." && pwd)"
export SYNBIO_ROOT
unset _synenv_self
_synbio_env="$SYNBIO_ROOT/env"

# --- shared, read-only env root + shared binaries (micromamba, synrun) on PATH -
export MAMBA_ROOT_PREFIX="$_synbio_env/mm"
case ":$PATH:" in
  *":$_synbio_env/bin:"*) : ;;
  *) PATH="$_synbio_env/bin:$PATH" ;;
esac
export PATH

# --- per-user writable workspace: tmp, HOME, and all rebuildable compile caches
# (`:=` guards keep this stable across nested sourcing, e.g. synrun -> micromamba).
: "${WORKDIR:=${SYNBIO_WORKDIR:-$HOME/synbio-ws}}"
export WORKDIR
export HOME="$WORKDIR"
export TMPDIR="$WORKDIR/tmp"
export XDG_CACHE_HOME="$WORKDIR/.cache"
export PIP_CACHE_DIR="$WORKDIR/.cache/pip"
export MPLCONFIGDIR="$WORKDIR/.cache/matplotlib"
export NUMBA_CACHE_DIR="$WORKDIR/.cache/numba"
export TRITON_CACHE_DIR="$WORKDIR/.cache/triton"
export CUDA_CACHE_PATH="$WORKDIR/.cache/nv"
export TORCH_EXTENSIONS_DIR="$WORKDIR/.cache/torch_extensions"
export CONDA_PKGS_DIRS="$WORKDIR/.cache/mamba/pkgs"

# --- SHARED, download-once model cache (team-wide; honors "кэш HF общий") ------
: "${SYNBIO_SHARED_CACHE:=$SYNBIO_ROOT/cache}"
export HF_HOME="$SYNBIO_SHARED_CACHE/huggingface"
# HF_HUB_CACHE is the name huggingface_hub actually resolves (constants.HF_HUB_CACHE);
# HUGGINGFACE_HUB_CACHE is the legacy alias kept for older code. Weights live under here
# as models--org--name (consolidated into this hub dir — see env/README.md). Do NOT set the
# deprecated TRANSFORMERS_CACHE: modern transformers maps it to HF_HOME and appends /hub,
# which silently diverged from where the shards actually were.
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TORCH_HOME="$SYNBIO_SHARED_CACHE/torch"

# Compute nodes have no outbound internet: default to reading already-cached weights
# directly — an online etag check on cached files hangs on a long timeout. Guarded (`:=`)
# so a node WITH network can download by exporting HF_HUB_OFFLINE=0 before sourcing.
: "${HF_HUB_OFFLINE:=1}"; export HF_HUB_OFFLINE
: "${TRANSFORMERS_OFFLINE:=1}"; export TRANSFORMERS_OFFLINE

export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE=offline

mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR" \
         "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$TORCH_EXTENSIONS_DIR" \
         "$CONDA_PKGS_DIRS" "$HUGGINGFACE_HUB_CACHE" "$TORCH_HOME" 2>/dev/null || true

unset _synbio_env
