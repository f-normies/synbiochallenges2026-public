#!/bin/bash
# Containerless interactive GPU session (SLURM).
#
# Why: a plain srun with NO --container-image runs the host TaskProlog and drops
# you into a normal shell on the GPU node, where you use the SHARED native envs
# via `synrun`. (On a cluster where enroot/pyxis works, you can instead build the
# image from docker/ — see README.)
#
# Prereqs: submodules checked out (`git submodule update --init`) and
#          `bash env/bootstrap.sh` has been run once (shared micromamba root
#          exists under env/mm).
# Usage:   bash start_synbio.sh
#          (quick tests: swap --partition=long for --partition=short)
#
# REPO is auto-derived from this script's location, so it works from any clone.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

srun --job-name=synbio2026 \
     --cpus-per-task=16 \
     --mem=64gb \
     --partition=long \
     --constraint=gpu \
     --gres=gpu:1 \
     --pty bash -c "source '$REPO/env/synenv.sh'; export PYTHONPATH='$REPO/src'\${PYTHONPATH:+:\$PYTHONPATH}; cd '$REPO'; exec bash"
