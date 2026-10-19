#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=. PYTORCH_ALLOC_CONF=expandable_segments:True
PY=/home/kavyansh/anaconda3/envs/myenv/bin/python
mkdir -p results/seg checkpoints
LOG=results/seg/vkitti_refs.log; : > "$LOG"

echo "=========== [1/2] original_vkitti  $(date '+%F %T') ===========" | tee -a "$LOG"
$PY scripts/train_seg_vkitti.py --mode original --epochs 30 --bs 2 2>&1 | tee -a "$LOG"

echo "=========== [2/2] oracle_vkitti  $(date '+%F %T') ===========" | tee -a "$LOG"
$PY scripts/train_seg_vkitti.py --mode oracle --epochs 30 --bs 2 2>&1 | tee -a "$LOG"

echo "########## VKITTI REFS DONE  $(date '+%F %T') ##########" | tee -a "$LOG"
