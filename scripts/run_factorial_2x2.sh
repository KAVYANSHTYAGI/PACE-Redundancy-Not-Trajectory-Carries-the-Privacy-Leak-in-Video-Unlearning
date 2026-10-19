#!/usr/bin/env bash
# Complete the 2x2 factorial (D_T fooling x pooling) requested for the
# supplementary. Corners A (PACE-T: lam_adv=1,lam_pool=0, one-sided TC via
# a different candidate function) and D (PACE: lam_adv=1,lam_pool=1) already
# exist as trained checkpoints (unlearned_ADVR-T.pth, unlearned_ADVR-P.pth).
# This trains the two missing corners using the exact recipe of
# scripts/run_advrp.sh (epochs=4, clip_len=3, bs=2, lam_r=1.0, lam_tc=0.5),
# varying only lam_adv/lam_pool, then evaluates all four under the main
# protocol (mamba, T=16, stride=2, N=2).
set -e
set -o pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=. PYTORCH_ALLOC_CONF=expandable_segments:True
PY=/home/kavyansh/anaconda3/envs/myenv/bin/python
LOG=results/seg/factorial_2x2.log
: > "$LOG"

echo "=== [1/3] train PACE-P (pooled-only: lam_adv=0, lam_pool=1) $(date) ===" | tee -a "$LOG"
[ -f checkpoints/unlearned_PACE-P-only.pth ] || \
  $PY scripts/test_candidate.py --name ADVR-P --epochs 4 --clip_len 3 --bs 2 \
      --lam_r 1.0 --lam_adv 0.0 --lam_tc 0.5 --lam_pool 1.0 \
      --save_as PACE-P-only --no_eval 2>&1 | tee -a "$LOG"

echo "=== [2/3] train PACE-TS (two-sided-only: lam_adv=1, lam_pool=0) $(date) ===" | tee -a "$LOG"
[ -f checkpoints/unlearned_PACE-TS-only.pth ] || \
  $PY scripts/test_candidate.py --name ADVR-P --epochs 4 --clip_len 3 --bs 2 \
      --lam_r 1.0 --lam_adv 1.0 --lam_tc 0.5 --lam_pool 0.0 \
      --save_as PACE-TS-only --no_eval 2>&1 | tee -a "$LOG"

echo "=== [3/3] eval all four 2x2 corners under main protocol $(date) ===" | tee -a "$LOG"
MODELS="ADVR-T,PACE-P-only,PACE-TS-only,ADVR-P" \
  N_TRIALS=2 STEPS=500 EFF_CLIPS=4 PHYS_BS=1 N_EVAL=800 N_TRAIN=4000 \
  MIN_VIDEO_LEN=31 CLIP_LEN=16 FRAME_STRIDE=2 TEMPORAL_KIND=mamba \
  OUT_JSON=results/seg/zoo_mamba_T16_s2_factorial2x2.json \
  $PY scripts/run_temporal_protocol.py 2>&1 | tee -a "$LOG"

echo "########## 2X2 FACTORIAL DONE $(date) ##########" | tee -a "$LOG"
