#!/usr/bin/env bash
# VKITTI2 (2nd dataset) pipeline: 8 baselines + TAE + PACE, forget class=Car.
# Reference models (oracle/original) already trained separately.
set -e
set -o pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=. PYTORCH_ALLOC_CONF=expandable_segments:True FORGET_CLS=1 NUM_CLASSES=14
PY=/home/kavyansh/anaconda3/envs/myenv/bin/python
mkdir -p results/seg checkpoints
LOG=results/seg/vkitti_pipeline.log; : > "$LOG"

echo "=========== [1/3] 8 baselines  $(date '+%F %T') ===========" | tee -a "$LOG"
$PY scripts/run_baselines_vkitti.py --bs 2 2>&1 | tee -a "$LOG"

echo "=========== [2/3] TAE (vkitti, epoch-scaled)  $(date '+%F %T') ===========" | tee -a "$LOG"
[ -f checkpoints/unlearned_TAE_vkitti.pth ] || \
  $PY scripts/test_candidate_vkitti.py --name ADVR-T --epochs 24 --clip_len 3 --bs 2 \
      --lam_r 1.0 --lam_adv 1.0 --lam_tc 0.5 --save_as TAE_vkitti 2>&1 | tee -a "$LOG"

echo "=========== [3/3] PACE (vkitti, epoch-scaled)  $(date '+%F %T') ===========" | tee -a "$LOG"
[ -f checkpoints/unlearned_PACE_vkitti.pth ] || \
  $PY scripts/test_candidate_vkitti.py --name ADVR-P --epochs 24 --clip_len 3 --bs 2 \
      --lam_r 1.0 --lam_adv 1.0 --lam_tc 0.5 --lam_pool 1.0 --save_as PACE_vkitti 2>&1 | tee -a "$LOG"

echo "########## VKITTI PIPELINE DONE  $(date '+%F %T') ##########" | tee -a "$LOG"
