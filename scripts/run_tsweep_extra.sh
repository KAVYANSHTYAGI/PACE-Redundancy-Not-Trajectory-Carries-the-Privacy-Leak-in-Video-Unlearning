#!/usr/bin/env bash
set -e
set -o pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=. FORGET_CLS=60
PY=/home/kavyansh/anaconda3/envs/myenv/bin/python
LOG=results/seg/tsweep_extra.log
: > "$LOG"

for T in 2 4; do
  echo "=== T=$T $(date) ===" | tee -a "$LOG"
  CLIP_LEN=$T FRAME_STRIDE=2 MIN_VIDEO_LEN=31 N_TRIALS=2 STEPS=500 EFF_CLIPS=4 \
  N_EVAL=800 N_TRAIN=4000 TEMPORAL_KIND=mamba \
  MODELS=oracle_FLOOR,original,ADVR-T,ADVR-P \
  OUT_JSON=results/seg/zoo_mamba_T${T}_s2_person.json \
  $PY -u scripts/run_temporal_protocol.py 2>&1 | tee -a "$LOG"
done
echo "########## T-SWEEP EXTRA DONE $(date) ##########" | tee -a "$LOG"
