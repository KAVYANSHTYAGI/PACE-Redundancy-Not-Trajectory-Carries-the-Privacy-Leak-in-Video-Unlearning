#!/usr/bin/env bash
# Train ADVR-P (pooling-discriminator + two-sided L_tc, EXPERIMENTS.md sec 9)
# and evaluate it against the established reference set under the same
# strongest-attacker protocol used throughout (mamba, T=16, stride=2).
#
# Config matches ADVR-T's exact training run (epochs=4, lam_tc=0.5,
# temporal_kind=conv3d, bs=2, clip_len=3) plus --lam_pool 1.0, so any
# difference from ADVR-T is attributable to the method change, not the recipe.
#
# Usage: bash scripts/run_advrp.sh
set -e
set -o pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=. PYTORCH_ALLOC_CONF=expandable_segments:True
PY=/home/kavyansh/anaconda3/envs/myenv/bin/python
mkdir -p results/seg
LOG=results/seg/advrp.log; : > "$LOG"

echo "=========== train ADVR-P  $(date '+%F %T') ===========" | tee -a "$LOG"
[ -f checkpoints/unlearned_ADVR-P.pth ] || \
  $PY scripts/test_candidate.py --name ADVR-P --epochs 4 --clip_len 3 --bs 2 \
      --lam_r 1.0 --lam_adv 1.0 --lam_tc 0.5 --lam_pool 1.0 \
      --save_as ADVR-P --no_eval 2>&1 | tee -a "$LOG"

echo "=========== eval vs oracle_FLOOR/original/ADVR/ADVR-T under mamba, T=16/stride=2  $(date '+%F %T') ===========" | tee -a "$LOG"
MODELS="oracle_FLOOR,original,ADVR,ADVR-T,ADVR-P" \
  N_TRIALS=2 STEPS=500 EFF_CLIPS=4 PHYS_BS=1 N_EVAL=800 N_TRAIN=4000 \
  MIN_VIDEO_LEN=31 CLIP_LEN=16 FRAME_STRIDE=2 TEMPORAL_KIND=mamba \
  OUT_JSON=results/seg/zoo_mamba_T16_s2_advrp.json \
  $PY scripts/run_temporal_protocol.py 2>&1 | tee -a "$LOG"

echo "########## ADVR-P RUN DONE  $(date '+%F %T') ##########" | tee -a "$LOG"
