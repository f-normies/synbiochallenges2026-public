#!/usr/bin/env bash
# Build the 7 tool envs natively into ONE shared micromamba root — no container.
#
# enroot is pyxis-only on Aldan3 and the TaskProlog-in-namespace bug is unfixable
# (admin confirmed), so we drop containers entirely. micromamba is NOT a container:
# it is a userspace package manager that writes plain directories of binaries — no
# enroot, no namespaces. We reproduce the image's envs (same docker/envs/*.yml +
# the same pip fix-ups the Dockerfile applies) into a shared, read-only prefix that
# the whole team uses; per-user writable state lives under $WORKDIR (see synenv.sh).
#
# Run ONCE, by one person, on a node WITH network (login or CPU node is fine):
#   bash env/bootstrap.sh                     # build all 7 (default)
#   bash env/bootstrap.sh esm dnatools        # build a subset (retry-friendly)
#   SYNBIO_FORCE=1 bash env/bootstrap.sh esm  # rebuild even if already present
#   SYNBIO_GROUP=dogma bash env/bootstrap.sh  # chgrp the shared tree to a team group
#   SYNBIO_NO_CHECK=1 bash env/bootstrap.sh   # skip the CPU import smoke at the end
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"      # .../env
REPO="$(cd "$HERE/.." && pwd)"                            # repo root
REPOS="$REPO/docker/repos"                                # vendored tool sources
ENVS="$REPO/docker/envs/mamba"                            # conda env manifests
MM_ROOT="$HERE/mm"                                        # MAMBA_ROOT_PREFIX (shared)
MM_BIN="$HERE/bin/micromamba"                             # micromamba binary (shared)
SHARED_CACHE="${SYNBIO_SHARED_CACHE:-$REPO/cache}"        # download-once weight cache

CU121="https://download.pytorch.org/whl/cu121"
CU113="https://download.pytorch.org/whl/cu113"

log(){ printf '\n\033[1m== %s ==\033[0m\n' "$*"; }

# ---- micromamba binary (static linux-64), placed on the shared path ----------
ensure_micromamba(){
  [ -x "$MM_BIN" ] && return 0
  log "fetching micromamba -> $MM_BIN"
  mkdir -p "$HERE/bin"
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xj -C "$HERE" bin/micromamba
  chmod +x "$MM_BIN"
}

mm(){ "$MM_BIN" "$@"; }
run(){ local e="$1"; shift; "$MM_BIN" run -n "$e" "$@"; }
have_env(){ [ -d "$MM_ROOT/envs/$1" ]; }

# Returns 0 (and prints) if the env is present and should be skipped; 1 otherwise.
skip_if_present(){
  if have_env "$1"; then
    if [ "${SYNBIO_FORCE:-0}" = "1" ]; then
      log "rebuilding '$1' (SYNBIO_FORCE=1) — removing old env"
      mm env remove -y -n "$1" || true
      return 1
    fi
    log "env '$1' already present — skip (SYNBIO_FORCE=1 to rebuild)"
    return 0
  fi
  return 1
}

# ---- per-env builds: identical recipe to docker/Dockerfile -------------------
build_proteinmpnn(){ skip_if_present proteinmpnn && return 0
  mm create -y -n proteinmpnn -f "$ENVS/proteinmpnn.yml"; }

build_dnatools(){ skip_if_present dnatools && return 0
  mm create -y -n dnatools -f "$ENVS/dnatools.yml"; }

build_esm(){ skip_if_present esm && return 0
  mm create -y -n esm -f "$ENVS/esm.yml"
  run esm pip install -e "$REPOS/esm"; }   # also clones the Biohub transformers fork (needs git+net)

build_ligandmpnn(){ skip_if_present ligandmpnn && return 0
  mm create -y -n ligandmpnn -c conda-forge python=3.10 pip
  run ligandmpnn pip install --extra-index-url "$CU121" torch==2.4.1
  run ligandmpnn pip install -r "$REPOS/LigandMPNN/requirements.txt"
  # ProDy (requirements pins 2.4.1) imports pkg_resources, dropped in setuptools>=81 —
  # same breakage the Dockerfile patches for spurs/evodiff. NB requirements.txt also
  # pins torch==2.2.1, intentionally downgrading the 2.4.1 installed just above.
  run ligandmpnn pip install "setuptools<81"
  # stage 03 generate.ligandmpnn emits candidates.parquet from this env (synbio.io -> pandas)
  run ligandmpnn pip install "pandas>=2.0" "pyarrow>=14"; }

build_evodiff(){ skip_if_present evodiff && return 0
  mm create -y -n evodiff -c conda-forge python=3.10 pip
  run evodiff pip install --extra-index-url "$CU121" torch==2.4.1
  run evodiff pip install -e "$REPOS/evodiff"
  run evodiff pip install "setuptools<81"; }   # evodiff/pretrained.py imports pkg_resources

build_thermompnn(){ skip_if_present thermompnn && return 0
  mm create -y -n thermompnn -f "$REPOS/ThermoMPNN-D/environment.yaml"
  # ThermoMPNN-D (a submodule) hardcodes the authors' cluster path in examples/configs/local.yaml;
  # get_protein_mpnn() builds the ProteinMPNN backbone path from cfg.platform.thermompnn_dir. Point
  # it at the vendored weights in this checkout (vanilla_model_weights/ lives in the repo root).
  sed -i "s#thermompnn_dir:.*#thermompnn_dir: \"$REPOS/ThermoMPNN-D\"#" \
    "$REPOS/ThermoMPNN-D/examples/configs/local.yaml"
  # Replace the self-inconsistent conda CUDA stack (pytorch-cuda=11.7 + cudatoolkit=11.3)
  # with the self-contained cu121 pip wheel — see docker/Dockerfile post-build notes.
  run thermompnn pip install --extra-index-url "$CU121" torch==2.4.1
  # ssm_utils -> train_thermompnn -> pytorch_lightning -> lightning_fabric imports pkg_resources,
  # dropped in setuptools>=81 — same guard as ligandmpnn/evodiff/spurs. (Bare `import thermompnn`
  # doesn't hit this chain, so the smoke missed it; v2_ssm.py does.)
  run thermompnn pip install "setuptools<81"
  # stage 05 emits parquet vote artifacts from this env
  run thermompnn pip install "pandas>=2.0" "pyarrow>=14"; }

build_spurs(){ skip_if_present spurs && return 0
  # Old 2022 stack: torch 1.12/cu113, PL 1.7.3. Wheel metadata violates PEP 440, so
  # this env needs pip<24.1; PL 1.7.3 then needs setuptools<81 (pkg_resources).
  mm create -y -n spurs -c conda-forge python=3.10 "pip<24.1"
  run spurs pip install --extra-index-url "$CU113" torch==1.12.0
  run spurs pip install -r "$REPOS/SPURS/requirements.txt"
  run spurs pip install -e "$REPOS/SPURS"
  run spurs pip install "setuptools<81"
  # stage 05 emits parquet vote artifacts from this env
  run spurs pip install "pandas>=2.0" "pyarrow>=14"; }

# ---- CPU import smoke (mirrors docker/smoke_test.sh imports) ------------------
smoke_one(){
  local e="$1" code="$2"
  printf '  %-12s ' "$e"
  if "$HERE/bin/synrun" "$e" python -c "$code" >/dev/null 2>&1; then
    printf '\033[32mOK\033[0m\n'
  else
    printf '\033[31mFAIL\033[0m\n'; SMOKE_RC=1
  fi
}
smoke(){
  local e
  for e in "$@"; do
    case "$e" in
      proteinmpnn) smoke_one "$e" "import torch, protein_mpnn_utils, pandas, pyarrow" ;;
      ligandmpnn)  smoke_one "$e" "import torch; assert torch.__version__.startswith('2.2'), torch.__version__; import prody" ;;
      thermompnn)  smoke_one "$e" "import torch, thermompnn, pandas, pyarrow, pytorch_lightning" ;;
      spurs)       smoke_one "$e" "import torch, spurs, pandas, pyarrow; assert torch.__version__.startswith('1.12')" ;;
      esm)         smoke_one "$e" "import esm" ;;
      evodiff)     smoke_one "$e" "import torch, evodiff" ;;
      dnatools)    smoke_one "$e" "import dnachisel, RNA, Bio, freesasa" ;;
    esac
  done
}

# ---- main --------------------------------------------------------------------
ensure_micromamba
export MAMBA_ROOT_PREFIX="$MM_ROOT"
export CONDA_PKGS_DIRS="$MM_ROOT/pkgs"     # build-time pkg cache, on the same FS as envs (hardlinks)
mkdir -p "$MM_ROOT"

ALL=(proteinmpnn dnatools esm ligandmpnn evodiff thermompnn spurs)
TARGETS=("$@"); [ "${#TARGETS[@]}" -eq 0 ] && TARGETS=("${ALL[@]}")

for e in "${TARGETS[@]}"; do
  log "building env: $e"
  "build_$e"
done

log "micromamba clean"
mm clean -ay || true

# Shared, download-once weight cache (ESMC-6B / ESMFold2 etc.). setgid so new files
# inherit the directory's group; pass SYNBIO_GROUP to give it to the team group.
log "preparing shared weight cache: $SHARED_CACHE"
mkdir -p "$SHARED_CACHE/huggingface/hub" "$SHARED_CACHE/torch"
chmod -R 2775 "$SHARED_CACHE" || true
if [ -n "${SYNBIO_GROUP:-}" ]; then
  chgrp -R "$SYNBIO_GROUP" "$SHARED_CACHE" "$HERE" || true
fi

# Freeze the envs read-only: everyone may read/execute, nobody but the owner may
# write — corruption-proof and safe under concurrent multi-user `synrun`.
log "freezing envs read-only (a+rX, go-w)"
chmod -R a+rX "$MM_ROOT"
chmod -R go-w "$MM_ROOT"

if [ "${SYNBIO_NO_CHECK:-0}" != "1" ]; then
  log "CPU import smoke"
  SMOKE_RC=0
  smoke "${TARGETS[@]}"
  if [ "$SMOKE_RC" -ne 0 ]; then
    echo; echo "some envs failed the import smoke — see above" >&2
    exit 1
  fi
fi

log "done"
echo "shared root : $MM_ROOT"
echo "weight cache: $SHARED_CACHE"
echo "next        : source $REPO/env/synenv.sh   # then: synrun esm python -c 'import esm'"
