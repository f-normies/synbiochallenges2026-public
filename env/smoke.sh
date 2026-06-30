#!/usr/bin/env bash
# Containerless smoke for the shared native envs — the counterpart to
# docker/smoke_test.sh, but with NATIVE paths (env/mm, docker/repos) and
# reality-based version pins. Drives every env through `synrun <env> <cmd>` exactly
# as the pipeline does, and imports what the stages actually import.
#
#   bash env/smoke.sh             # CPU: imports + versions + repo entrypoints (no net)
#   bash env/smoke.sh --gpu       # + nvidia-smi and a per-torch-env fp16 matmul
#   bash env/smoke.sh --esmfold2  # + ESMFold2 fp16 fold probe (GPU + HF download!)
#   bash env/smoke.sh --all
#
# Run the CPU pass on a login node; run --gpu inside start_synbio_NOCONTAINER.sh.
# Exit code = number of failed checks.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../env
REPO="$(cd "$HERE/.." && pwd)"
SYNRUN="$HERE/bin/synrun"
REPOS="$REPO/docker/repos"

WANT_GPU=0; WANT_ESMFOLD2=0
for a in "$@"; do case "$a" in
  --gpu) WANT_GPU=1 ;;
  --esmfold2) WANT_ESMFOLD2=1; WANT_GPU=1 ;;
  --all) WANT_GPU=1; WANT_ESMFOLD2=1 ;;
  -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
  *) echo "unknown arg: $a" >&2; exit 2 ;;
esac; done

[ -x "$SYNRUN" ] || { echo "error: $SYNRUN not found — run env/bootstrap.sh first" >&2; exit 2; }

PASS=0; FAIL=0; declare -a FAILED=()
if [ -t 1 ]; then G=$'\e[32m'; R=$'\e[31m'; D=$'\e[2m'; Z=$'\e[0m'; else G=''; R=''; D=''; Z=''; fi
check(){ local n="$1"; shift; printf '  %-56s ' "$n"; local o
  if o=$("$@" 2>&1); then printf '%sOK%s\n' "$G" "$Z"; [ -n "${VERBOSE:-}" ] && printf '%s%s%s\n' "$D" "$o" "$Z"; PASS=$((PASS+1))
  else printf '%sFAIL%s\n' "$R" "$Z"; printf '%s%s%s\n' "$D" "$o" "$Z"; FAIL=$((FAIL+1)); FAILED+=("$n"); fi; }
sec(){ printf '\n=== %s ===\n' "$1"; }
sr(){ "$SYNRUN" "$@"; }

# ----- layout (native paths) -------------------------------------------------
sec "Layout (native)"
for e in proteinmpnn ligandmpnn thermompnn spurs esm evodiff dnatools; do
  check "env $e present" test -d "$REPO/env/mm/envs/$e"
done
for r in ProteinMPNN LigandMPNN ThermoMPNN-D SPURS evodiff esm; do
  check "repo docker/repos/$r present" test -d "$REPOS/$r"
done
check "synrun present"  test -x "$SYNRUN"
check "synenv present"  test -f "$HERE/synenv.sh"

# ----- proteinmpnn -----------------------------------------------------------
sec "proteinmpnn"
check "import protein_mpnn_utils + torch 2.4.1" \
  sr proteinmpnn python -c "import torch, protein_mpnn_utils, pandas, pyarrow; assert torch.__version__.startswith('2.4.1'), torch.__version__"
check "protein_mpnn_run.py present" test -f "$REPOS/ProteinMPNN/protein_mpnn_run.py"

# ----- ligandmpnn (torch 2.2.1 per its requirements; ProDy = the prior break) -
sec "ligandmpnn"
check "torch 2.2.1 + cuda 12.1" \
  sr ligandmpnn python -c "import torch; assert torch.__version__.startswith('2.2.1'), torch.__version__; assert torch.version.cuda=='12.1', torch.version.cuda"
check "import prody (pkg_resources/setuptools<81 guard)" \
  sr ligandmpnn python -c "import prody, ml_collections; print(prody.__version__)"
check "run.py present" test -f "$REPOS/LigandMPNN/run.py"

# ----- thermompnn (conda stack swapped to cu121 torch 2.4.1) ------------------
sec "thermompnn"
check "import thermompnn + pytorch_lightning + torch 2.4.1" \
  sr thermompnn python -c "import thermompnn, torch, pandas, pyarrow, pytorch_lightning; assert torch.__version__.startswith('2.4.1'), torch.__version__"

# ----- spurs (old cu113 stack) -----------------------------------------------
sec "spurs"
check "torch 1.12 + cuda 11.3" \
  sr spurs python -c "import torch; assert torch.__version__.startswith('1.12'), torch.__version__; assert torch.version.cuda=='11.3', torch.version.cuda"
check "pytorch-lightning 1.7.3" \
  sr spurs python -c "import pytorch_lightning as pl; assert pl.__version__=='1.7.3', pl.__version__"
check "import spurs + pkg_resources guard" \
  sr spurs python -c "import spurs, pkg_resources, pandas, pyarrow; print('spurs ok')"

# ----- esm (real stage-02 import is `from transformers import ESMCModel`) -----
sec "esm"
check "python 3.12 + torch 2.4.1 + cuda 12.1" \
  sr esm python -c "import sys, torch; assert sys.version_info[:2]==(3,12), sys.version_info; assert torch.__version__.startswith('2.4.1'), torch.__version__; assert torch.version.cuda=='12.1'"
check "import esm + version" \
  sr esm python -c "import esm; print(esm.__version__)"
check "from transformers import AutoTokenizer, ESMCModel (HF fork; stage 02)" \
  sr esm python -c "from transformers import AutoTokenizer, ESMCModel; import transformers; print(transformers.__version__)"
check "ESMFold2 path + transformers.models.esmfold2 (stage 07)" \
  sr esm python -c "from esm.models.esmfold2 import ESMFold2InputBuilder; import transformers.models.esmfold2; print('esmfold2 ok')"
check "peft present (LoRA tooling)" \
  sr esm python -c "import peft; print(peft.__version__)"

# ----- evodiff ---------------------------------------------------------------
sec "evodiff"
check "import evodiff + torch 2.4.1 + pkg_resources guard" \
  sr evodiff python -c "import evodiff, torch, pkg_resources; assert torch.__version__.startswith('2.4.1'), torch.__version__"

# ----- dnatools (CPU; DNA gate + orchestrator) -------------------------------
sec "dnatools"
check "dnachisel + ViennaRNA(RNA) + Bio + freesasa" \
  sr dnatools python -c "import dnachisel, RNA, Bio, freesasa; print(dnachisel.__version__)"
check "numpy<2 + pandas + scipy + pyarrow + hydra" \
  sr dnatools python -c "import numpy, pandas, scipy, pyarrow, hydra; assert numpy.__version__.startswith('1.'), numpy.__version__"
check "foldseek on PATH (in env)" \
  sr dnatools bash -c "command -v foldseek"

# ----- pipeline wiring: synbio imports under its orchestrator env -------------
sec "pipeline wiring"
check "import synbio (orchestrator) under dnatools, PYTHONPATH=src" \
  env PYTHONPATH="$REPO/src" sr dnatools python -c "import synbio, synbio.orchestrator; print('synbio ok')"

# ----- GPU runtime (--gpu) ---------------------------------------------------
if [ "$WANT_GPU" = 1 ]; then
  sec "GPU runtime (--gpu)"
  check "nvidia-smi runs" nvidia-smi
  for e in proteinmpnn ligandmpnn thermompnn spurs esm evodiff; do
    check "$e: cuda available + 4096 fp16 matmul" \
      sr "$e" python -c "
import torch
assert torch.cuda.is_available(), 'no CUDA device'
x = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)
y = x @ x.T; torch.cuda.synchronize()
assert torch.isfinite(y).all()
print(torch.cuda.get_device_name(0))"
  done
fi

# ----- ESMFold2 fp16 fold probe (--esmfold2; downloads weights) --------------
if [ "$WANT_ESMFOLD2" = 1 ]; then
  sec "ESMFold2 fp16 fold (downloads biohub/ESMFold2)"
  check "folds sfGFP under fp16; peak VRAM < 31GB" \
    sr esm python -c "
import torch
from esm.models.esmfold2 import ESMFold2InputBuilder, ProteinInput, StructurePredictionInput
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
S=('MSKGEELFTGVVPILVELGDGVNGHKFSVRGEGEQDATNGKLTLKFICTTGKLPVPWPTLVTTLTYG'
   'VQCFSRYPDHMKRHDFFKSAMPQGYVQERTISFKDDGTYKTRADEVKFEQDTLVNRIELKGIDFKEDG'
   'NILGHKLEYNFSNHNVYITADQKNGIKANFKIRHNIVEDGQVQLADHYQQNTPIGDGPVLLPDNHYL'
   'STQSVLSKDPNEKRDHMVLLFVTAAGITHGMDELYK')
torch.cuda.reset_peak_memory_stats()
m=ESMFold2Model.from_pretrained('biohub/ESMFold2', torch_dtype=torch.float16).cuda().eval()
spi=StructurePredictionInput(sequences=[ProteinInput(id='A', sequence=S)])
with torch.no_grad():
    ESMFold2InputBuilder().fold(m, spi, num_loops=3, num_sampling_steps=20, num_diffusion_samples=1, seed=0)
p=torch.cuda.max_memory_allocated()/1e9
print('peak=%.1fGB'%p); assert p<31.0, p"
fi

# ----- summary ---------------------------------------------------------------
sec "Summary"
printf "%d / %d checks passed\n" "$PASS" "$((PASS+FAIL))"
if [ "$FAIL" -gt 0 ]; then printf "%sFailures:%s\n" "$R" "$Z"; for n in "${FAILED[@]}"; do printf '  - %s\n' "$n"; done; fi
exit "$FAIL"
