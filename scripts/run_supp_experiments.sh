#!/usr/bin/env bash
# Queued supplementary experiments, run sequentially after the T=8 span job
# (pid tracked externally) finishes, to avoid GPU contention on a single
# 16GB card. Each job reuses run_temporal_protocol.py with the exact config
# used for the main paper's corresponding numbers.
set -e
set -o pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=. FORGET_CLS=60
PY=/home/kavyansh/anaconda3/envs/myenv/bin/python
LOG=results/seg/supp_experiments.log
mkdir -p results/seg
: > "$LOG"

wait_for_pid() {
  while kill -0 "$1" 2>/dev/null; do sleep 30; done
}

if [ -n "$1" ]; then
  echo "waiting for pid $1 to finish..." | tee -a "$LOG"
  wait_for_pid "$1"
fi

echo "=== [1/3] PACE, conv3d architecture, T=16 (extends architecture-robustness to PACE) $(date) ===" | tee -a "$LOG"
CLIP_LEN=16 FRAME_STRIDE=2 MIN_VIDEO_LEN=31 N_TRIALS=2 STEPS=500 EFF_CLIPS=4 \
N_EVAL=800 N_TRAIN=4000 TEMPORAL_KIND=conv3d WITH_FRAME=1 \
MODELS=ADVR-P OUT_JSON=results/seg/zoo_conv3d_T16_s2_PACE.json \
$PY -u scripts/run_temporal_protocol.py 2>&1 | tee -a "$LOG"

echo "=== [2/3] PACE, convgru architecture, T=16 $(date) ===" | tee -a "$LOG"
CLIP_LEN=16 FRAME_STRIDE=2 MIN_VIDEO_LEN=31 N_TRIALS=2 STEPS=500 EFF_CLIPS=4 \
N_EVAL=800 N_TRAIN=4000 TEMPORAL_KIND=convgru WITH_FRAME=1 \
MODELS=ADVR-P OUT_JSON=results/seg/zoo_convgru_T16_s2_PACE.json \
$PY -u scripts/run_temporal_protocol.py 2>&1 | tee -a "$LOG"

echo "=== [3/3] PACE, mamba, T=16, N=3 replication (tighter trial spread on headline number) $(date) ===" | tee -a "$LOG"
CLIP_LEN=16 FRAME_STRIDE=2 MIN_VIDEO_LEN=31 N_TRIALS=3 STEPS=500 EFF_CLIPS=4 \
N_EVAL=800 N_TRAIN=4000 TEMPORAL_KIND=mamba \
MODELS=ADVR-P OUT_JSON=results/seg/zoo_mamba_T16_s2_PACE_N3.json \
$PY -u scripts/run_temporal_protocol.py 2>&1 | tee -a "$LOG"

echo "########## SUPP EXPERIMENTS DONE $(date) ##########" | tee -a "$LOG"
