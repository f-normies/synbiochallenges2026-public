#!/usr/bin/env bash
# Download the non-HuggingFace model weights into weights/ (gitignored).
#
# HuggingFace models (ESMC-6B, ESMC-600M, ESMFold2, SPURS) are pulled at runtime by
# their stages — NOT here. This script only fetches the checkpoints the pipeline expects
# under weights/:
#
#   weights/ligandmpnn/ligandmpnn_v_32_0{05,10,20,30}_25.pt   <- files.ipd.uw.edu
#   weights/proteinmpnn/proteinmpnn_v_48_0{02,10,20,30}.pt     <- files.ipd.uw.edu
#   weights/thermompnn/thermoMPNN_default.pt                   <- Kuhlman-Lab/ThermoMPNN
#   weights/thermompnn-d/{ThermoMPNN-D-ens,ThermoMPNN-ens}{1,2,3}.ckpt
#       <- copied from the PINNED ThermoMPNN-D submodule (reproducible; run
#          `git submodule update --init docker/repos/ThermoMPNN-D` first)
#
# Idempotent: files that already exist are skipped. Re-run any time.
# Usage: bash scripts/download_weights.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
W="$REPO/weights"
IPD="https://files.ipd.uw.edu/pub/ligandmpnn"

# --- pick a downloader -------------------------------------------------------
if command -v curl >/dev/null 2>&1; then
  _dl() { curl -fL --retry 3 --create-dirs -o "$2" "$1"; }
elif command -v wget >/dev/null 2>&1; then
  _dl() { mkdir -p "$(dirname "$2")"; wget -q -O "$2" "$1"; }
else
  echo "ERROR: need either curl or wget on PATH." >&2; exit 1
fi

fetch() {  # url dest  (skip if present; clean up partial on failure)
  local url="$1" dest="$2"
  if [ -s "$dest" ]; then echo "  skip (exists): ${dest#"$REPO"/}"; return 0; fi
  echo "  get:           ${dest#"$REPO"/}"
  _dl "$url" "$dest" || { rm -f "$dest"; echo "ERROR: failed to download $url" >&2; exit 1; }
}

echo "[1/4] LigandMPNN  -> weights/ligandmpnn/"
for n in 005 010 020 030; do
  fetch "$IPD/ligandmpnn_v_32_${n}_25.pt" "$W/ligandmpnn/ligandmpnn_v_32_${n}_25.pt"
done

echo "[2/4] ProteinMPNN -> weights/proteinmpnn/"
for n in 002 010 020 030; do
  fetch "$IPD/proteinmpnn_v_48_${n}.pt" "$W/proteinmpnn/proteinmpnn_v_48_${n}.pt"
done

echo "[3/4] ThermoMPNN  -> weights/thermompnn/"
fetch "https://github.com/Kuhlman-Lab/ThermoMPNN/raw/refs/heads/main/models/thermoMPNN_default.pt" \
      "$W/thermompnn/thermoMPNN_default.pt"

echo "[4/4] ThermoMPNN-D ensemble -> weights/thermompnn-d/  (copy from pinned submodule)"
TD_SRC="$REPO/docker/repos/ThermoMPNN-D/model_weights"
if [ ! -s "$TD_SRC/ThermoMPNN-D-ens1.ckpt" ]; then
  echo "ERROR: $TD_SRC not populated." >&2
  echo "       Run: git submodule update --init docker/repos/ThermoMPNN-D" >&2
  exit 1
fi
mkdir -p "$W/thermompnn-d"
for f in ThermoMPNN-D-ens1 ThermoMPNN-D-ens2 ThermoMPNN-D-ens3 \
         ThermoMPNN-ens1   ThermoMPNN-ens2   ThermoMPNN-ens3; do
  dest="$W/thermompnn-d/$f.ckpt"
  if [ -s "$dest" ]; then echo "  skip (exists): weights/thermompnn-d/$f.ckpt"; continue; fi
  echo "  copy:          weights/thermompnn-d/$f.ckpt"
  cp "$TD_SRC/$f.ckpt" "$dest"
done

echo
echo "Done. Non-HF weights are under $W/"
echo "(HuggingFace models — ESMC-6B, ESMC-600M, ESMFold2, SPURS — download at runtime.)"
