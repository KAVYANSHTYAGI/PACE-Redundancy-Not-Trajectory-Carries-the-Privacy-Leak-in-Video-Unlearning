#!/usr/bin/env bash
# Establish the B0 baseline pair (original + oracle) sequentially -- one GPU,
# one model at a time. Resumable in spirit with project1's convention: skips a
# mode if its checkpoint already exists, so re-running after an interruption
# only does the missing half.
set -e
cd "$(dirname "$0")/.."
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONPATH=.
EPOCHS=${EPOCHS:-4}
BS=${BS:-6}
mkdir -p results/seg checkpoints

run_mode () {
  local MODE=$1
  local CKPT="checkpoints/vspw_${MODE}_deeplabv3r50_cls60.pth"
  if [ -f "$CKPT" ]; then
    echo "[run_baseline] $MODE already done ($CKPT exists), skipping"
    return 0
  fi
  echo "[run_baseline] training $MODE ($EPOCHS epochs, bs$BS)"
  /home/kavyansh/anaconda3/envs/myenv/bin/python scripts/train_seg.py \
    --mode "$MODE" --epochs "$EPOCHS" --bs "$BS" --size 512 \
    > "results/seg/vspw_train_${MODE}.log" 2>&1
  echo "[run_baseline] $MODE done"
}

run_mode original
run_mode oracle
echo "[run_baseline] BOTH DONE"
