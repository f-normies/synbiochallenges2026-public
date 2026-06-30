# Sourceable: redirect every write-on-run path to the single writable mount.
# Loaded by both entrypoint.sh and synrun so caches resolve correctly under an
# enroot read-only squashfs rootfs, regardless of how the container is launched.
: "${WORKDIR:=/workspace}"

export HOME="$WORKDIR"
export TMPDIR="$WORKDIR/tmp"
export XDG_CACHE_HOME="$WORKDIR/.cache"
export PIP_CACHE_DIR="$WORKDIR/.cache/pip"
export HF_HOME="$WORKDIR/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
# TRANSFORMERS_CACHE must point at the hub dir (not HF_HOME root), else transformers models
# land in $HF_HOME/ instead of $HF_HOME/hub/ — splitting the cache. (Also deprecated; HF_HOME +
# HUGGINGFACE_HUB_CACHE above already suffice, but set it consistently for older code paths.)
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export TORCH_HOME="$WORKDIR/.cache/torch"
export TORCH_EXTENSIONS_DIR="$WORKDIR/.cache/torch_extensions"
export MPLCONFIGDIR="$WORKDIR/.cache/matplotlib"
export NUMBA_CACHE_DIR="$WORKDIR/.cache/numba"
export TRITON_CACHE_DIR="$WORKDIR/.cache/triton"
export CUDA_CACHE_PATH="$WORKDIR/.cache/nv"
export CONDA_PKGS_DIRS="$WORKDIR/.cache/mamba/pkgs"
export MAMBA_ROOT_PREFIX="/opt/micromamba"
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE=offline

mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$HUGGINGFACE_HUB_CACHE" "$TORCH_HOME" \
         "$TORCH_EXTENSIONS_DIR" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR" \
         "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$CONDA_PKGS_DIRS" 2>/dev/null || true
