#!/usr/bin/env bash
# Smoke tests for the synbio2026 *micromamba* container (Dockerfile).
#
# This is the per-tool-env counterpart to smoke_test.sh (which targets the other,
# 3-venv image). Here envs are micromamba envs driven by `synrun <env> <cmd>`,
# repos live at /repos, and there is no /opt/venvs or `activate`.
#
# Usage (inside the container):
#   bash smoke_test.sh                 # imports + per-env version checks (CPU, no network)
#   bash smoke_test.sh --gpu           # add nvidia-smi + per-torch-env CUDA matmul
#   bash smoke_test.sh --esmfold2      # add the ESMFold2 fp16 feasibility probe (GPU + HF download!)
#   bash smoke_test.sh --weights       # add weight-file presence checks (requires /data/weights)
#   bash smoke_test.sh --all           # everything
#
# Exit code = number of failed checks.

set -uo pipefail

WANT_GPU=0
WANT_ESMFOLD2=0
WANT_WEIGHTS=0
for arg in "$@"; do
  case "$arg" in
    --gpu)      WANT_GPU=1 ;;
    --esmfold2) WANT_ESMFOLD2=1; WANT_GPU=1 ;;
    --weights)  WANT_WEIGHTS=1 ;;
    --all)      WANT_GPU=1; WANT_ESMFOLD2=1; WANT_WEIGHTS=1 ;;
    -h|--help)  sed -n '2,21p' "$0"; exit 0 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

# Must run inside the image — synrun + micromamba have to be on PATH.
if ! command -v synrun >/dev/null 2>&1 || ! command -v micromamba >/dev/null 2>&1; then
  echo "error: 'synrun'/'micromamba' not found — run this INSIDE the container, not on the host." >&2
  exit 2
fi

PASS=0; FAIL=0
declare -a FAILED_NAMES=()

if [ -t 1 ]; then GREEN=$'\e[32m'; RED=$'\e[31m'; DIM=$'\e[2m'; RST=$'\e[0m'
else GREEN=''; RED=''; DIM=''; RST=''; fi

check() {
  local name="$1"; shift
  printf '  %-58s ' "$name"
  local out
  if out=$("$@" 2>&1); then
    printf '%sOK%s\n' "$GREEN" "$RST"
    [ -n "${VERBOSE:-}" ] && printf '%s%s%s\n' "$DIM" "$out" "$RST"
    PASS=$((PASS+1))
  else
    printf '%sFAIL%s\n' "$RED" "$RST"
    printf '%s%s%s\n' "$DIM" "$out" "$RST"
    FAIL=$((FAIL+1))
    FAILED_NAMES+=("$name")
  fi
}

section() { printf '\n=== %s ===\n' "$1"; }

# ----- Image layout sanity --------------------------------------------------
section "Image layout"
for e in proteinmpnn ligandmpnn thermompnn spurs esm evodiff dnatools; do
  check "env $e present" test -d "/opt/micromamba/envs/$e"
done
for r in ProteinMPNN LigandMPNN ThermoMPNN-D SPURS evodiff esm; do
  check "repo /repos/$r present" test -d "/repos/$r"
done
check "synrun launcher present"     test -x /usr/local/bin/synrun
check "synenv.sh present"           test -f /usr/local/bin/synenv.sh
check "entrypoint present"          test -x /usr/local/bin/entrypoint.sh

# ----- proteinmpnn env ------------------------------------------------------
section "proteinmpnn (ProteinMPNN + ProteinMPNN-ddG)"
check "import protein_mpnn_utils + torch" \
  synrun proteinmpnn python -c "import protein_mpnn_utils, torch; print('torch', torch.__version__)"
check "protein_mpnn_run.py present" test -f /repos/ProteinMPNN/protein_mpnn_run.py

# ----- ligandmpnn env -------------------------------------------------------
section "ligandmpnn (torch 2.4.1+cu121)"
check "torch 2.4.1 + cuda 12.1" \
  synrun ligandmpnn python -c "
import torch
assert torch.__version__.startswith('2.4.1'), torch.__version__
assert torch.version.cuda == '12.1', torch.version.cuda
print(torch.__version__, 'cuda', torch.version.cuda)"
check "LigandMPNN run.py present" test -f /repos/LigandMPNN/run.py

# ----- thermompnn env -------------------------------------------------------
section "thermompnn (ThermoMPNN-D)"
check "import thermompnn + torch" \
  synrun thermompnn python -c "import thermompnn, torch; print('torch', torch.__version__)"

# ----- spurs env (old stack, fully isolated) --------------------------------
section "spurs (torch 1.12.0+cu113, PL 1.7.3)"
check "torch 1.12 + cuda 11.3" \
  synrun spurs python -c "
import torch
assert torch.__version__.startswith('1.12'), torch.__version__
assert torch.version.cuda == '11.3', torch.version.cuda
print(torch.__version__, 'cuda', torch.version.cuda)"
check "pytorch-lightning 1.7.3" \
  synrun spurs python -c "import pytorch_lightning as pl; assert pl.__version__=='1.7.3', pl.__version__; print(pl.__version__)"
check "import spurs" \
  synrun spurs python -c "import spurs; print('spurs ok')"

# ----- esm env (Biohub ESM SDK v3.3.0) --------------------------------------
section "esm (Biohub SDK v3.3.0 — ESMC 600M/6B + ESMFold2)"
check "python is 3.12" \
  synrun esm python -c "import sys; assert sys.version_info[:2]==(3,12), sys.version_info"
check "torch 2.4.1 + cuda 12.1" \
  synrun esm python -c "
import torch
assert torch.__version__.startswith('2.4.1'), torch.__version__
assert torch.version.cuda == '12.1', torch.version.cuda"
check "import esm + version" \
  synrun esm python -c "import esm; print('esm', esm.__version__)"
check "ESMFold2 import path resolves" \
  synrun esm python -c "from esm.models.esmfold2 import ESMFold2InputBuilder, ProteinInput, StructurePredictionInput; print('esmfold2 ok')"
check "transformers.models.esmfold2 (Biohub fork)" \
  synrun esm python -c "import transformers.models.esmfold2; import transformers; print('transformers', transformers.__version__)"
check "esmc_client (Forge/Biohub API SDK)" \
  synrun esm python -c "from esm.sdk import esmc_client; print('sdk ok')"

# ----- evodiff env ----------------------------------------------------------
section "evodiff (torch 2.4.1+cu121)"
check "import evodiff + torch 2.4.1" \
  synrun evodiff python -c "
import evodiff, torch
assert torch.__version__.startswith('2.4.1'), torch.__version__
print('evodiff ok', torch.__version__)"

# ----- dnatools env (CPU DNA-side gate) -------------------------------------
section "dnatools (CPU; DNA gate + aggregation)"
check "dnachisel + ViennaRNA(RNA) + Bio + freesasa" \
  synrun dnatools python -c "
import dnachisel, RNA, Bio, freesasa
print('dnachisel', dnachisel.__version__)"
check "numpy<2 + pandas + scipy" \
  synrun dnatools python -c "
import numpy, pandas, scipy
assert numpy.__version__.startswith('1.'), numpy.__version__
print('numpy', numpy.__version__)"
check "foldseek on PATH (in env)" \
  synrun dnatools bash -c "command -v foldseek"

# ----- GPU runtime (--gpu) --------------------------------------------------
if [ "$WANT_GPU" = 1 ]; then
  section "GPU runtime (--gpu)"
  check "nvidia-smi runs" nvidia-smi
  for e in proteinmpnn ligandmpnn thermompnn spurs esm evodiff; do
    check "$e: torch.cuda.is_available()" \
      synrun "$e" python -c "import torch; assert torch.cuda.is_available(), 'no CUDA device'; print(torch.cuda.get_device_name(0))"
    check "$e: 4096x4096 fp16 matmul on cuda:0" \
      synrun "$e" python -c "
import torch
x = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)
y = x @ x.T
torch.cuda.synchronize()
assert torch.isfinite(y).all()
print('mean', float(y.float().abs().mean()))"
  done
fi

# ----- ESMFold2 fp16 feasibility probe (--esmfold2) -------------------------
# The Week-1 gate: can the 6B-backbone + diffusion model fold a GFP-length seq
# on a V100S 32 GB at fp16 (sm_70 has no hardware bf16)? Downloads biohub/ESMFold2
# from HuggingFace on first run (large) — needs network + a writable HF cache.
if [ "$WANT_ESMFOLD2" = 1 ]; then
  section "ESMFold2 fp16 feasibility (--esmfold2; downloads weights)"
  check "biohub/ESMFold2 folds sfGFP under fp16; peak VRAM reported" \
    synrun esm python -c "
import torch
from esm.models.esmfold2 import ESMFold2InputBuilder, ProteinInput, StructurePredictionInput
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
SFGFP = ('MSKGEELFTGVVPILVELGDGVNGHKFSVRGEGEQDATNGKLTLKFICTTGKLPVPWPTLVTTLTYG'
         'VQCFSRYPDHMKRHDFFKSAMPQGYVQERTISFKDDGTYKTRADEVKFEQDTLVNRIELKGIDFKEDG'
         'NILGHKLEYNFSNHNVYITADQKNGIKANFKIRHNIVEDGQVQLADHYQQNTPIGDGPVLLPDNHYL'
         'STQSVLSKDPNEKRDHMVLLFVTAAGITHGMDELYK')
torch.cuda.reset_peak_memory_stats()
model = ESMFold2Model.from_pretrained('biohub/ESMFold2', torch_dtype=torch.float16).cuda().eval()
spi = StructurePredictionInput(sequences=[ProteinInput(id='A', sequence=SFGFP)])
with torch.no_grad():
    ESMFold2InputBuilder().fold(model, spi, num_loops=3, num_sampling_steps=20, num_diffusion_samples=1, seed=0)
peak = torch.cuda.max_memory_allocated()/1e9
print('ESMFOLD2_OK peak=%.1fGB' % peak)
assert peak < 31.0, 'fits but uncomfortably close to 32GB: %.1fGB' % peak"
fi

# ----- Weight-file presence (--weights, requires /data/weights) -------------
# Adjust paths as weights are staged. ESMC/ESMFold2 pull from HF at runtime into
# the HF cache (HF_HOME), so they are NOT expected under /data/weights.
if [ "$WANT_WEIGHTS" = 1 ]; then
  section "Weights (--weights, requires /data/weights mount)"
  check "/data/weights mounted" test -d /data/weights
  check "ProteinMPNN v_48_020.pt present" test -f /data/weights/proteinmpnn/v_48_020.pt
  check "ThermoMPNN-D checkpoint present"  test -f /data/weights/thermompnn/thermoMPNN_default.pt
fi

# ----- Summary --------------------------------------------------------------
section "Summary"
TOTAL=$((PASS+FAIL))
printf "%d / %d checks passed\n" "$PASS" "$TOTAL"
if [ "$FAIL" -gt 0 ]; then
  printf "%sFailures:%s\n" "$RED" "$RST"
  for n in "${FAILED_NAMES[@]}"; do printf "  - %s\n" "$n"; done
fi

exit "$FAIL"
