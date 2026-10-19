#!/usr/bin/env bash
# Car-class generalization test (does the redundancy-leak finding and its fix
# hold for a class other than person?). FORGET_CLS=48 (car, raw label 49)
# throughout. See EXPERIMENTS.md for the full plan this implements.
#
# Stages, in order (each stage's checkpoint gates the next):
#   1. oracle_car, original_car reference models (train_seg.py)
#   2. 8 published baselines (run_baselines.py)
#   3. ADVR-T, ADVR-P (test_candidate.py) -- epoch-scaled vs the person recipe.
#      Measured directly (not guessed): car has 6077 train clip-units at
#      clip_len=3/stride=3 vs person's 45976 -- a 7.57x ratio. epochs=30 here
#      (4 * 7.57, rounded) so both classes see roughly the same total optimizer
#      steps; a straight epochs=4 on car would train on ~1/7.6 the exposure.
#   4. corrected mamba/T=16/stride=2 protocol against the full car-class set
#
# Usage: bash scripts/run_car_pipeline.sh
set -e
set -o pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=. PYTORCH_ALLOC_CONF=expandable_segments:True FORGET_CLS=48
PY=/home/kavyansh/anaconda3/envs/myenv/bin/python
mkdir -p results/seg checkpoints
LOG=results/seg/car_pipeline.log; : > "$LOG"

echo "=========== [1/4] oracle_car / original_car  $(date '+%F %T') ===========" | tee -a "$LOG"
# bs=6 not the script's default bs=8: EXPERIMENTS.md sec 6.2 already documents
# bs=8 OOMing on this 16GB card at 512-res DeepLabV3-R50 (confirmed again here
# on the first launch attempt) -- bs=6 is the established, working setting.
[ -f checkpoints/vspw_original_deeplabv3r50_cls48.pth ] || \
  $PY scripts/train_seg.py --mode original --epochs 24 --bs 6 2>&1 | tee -a "$LOG"
[ -f checkpoints/vspw_oracle_deeplabv3r50_cls48.pth ] || \
  $PY scripts/train_seg.py --mode oracle --epochs 24 --bs 6 2>&1 | tee -a "$LOG"

echo "=========== [2/4] 8 baselines  $(date '+%F %T') ===========" | tee -a "$LOG"
$PY scripts/run_baselines.py 2>&1 | tee -a "$LOG"

echo "=========== [3/4] ADVR-T (car, epoch-scaled)  $(date '+%F %T') ===========" | tee -a "$LOG"
[ -f checkpoints/unlearned_ADVR-T_car.pth ] || \
  $PY scripts/test_candidate.py --name ADVR-T --epochs 30 --clip_len 3 --bs 2 \
      --lam_r 1.0 --lam_adv 1.0 --lam_tc 0.5 \
      --save_as ADVR-T_car --no_eval 2>&1 | tee -a "$LOG"

echo "=========== [3/4] ADVR-P (car, epoch-scaled)  $(date '+%F %T') ===========" | tee -a "$LOG"
[ -f checkpoints/unlearned_ADVR-P_car.pth ] || \
  $PY scripts/test_candidate.py --name ADVR-P --epochs 30 --clip_len 3 --bs 2 \
      --lam_r 1.0 --lam_adv 1.0 --lam_tc 0.5 --lam_pool 1.0 \
      --save_as ADVR-P_car --no_eval 2>&1 | tee -a "$LOG"

echo "=========== [4/4] corrected protocol, mamba/T=16/stride=2  $(date '+%F %T') ===========" | tee -a "$LOG"
MODELS="oracle_FLOOR,original,bl_FT_s1_cls48,bl_NegGrad_s1_cls48,bl_NegGrad+_s1_cls48,bl_RL_s1_cls48,bl_SalUn_s1_cls48,bl_SSD_s1_cls48,bl_SCRUB_s1_cls48,bl_DELETE_s1_cls48,ADVR-T_car,ADVR-P_car" \
  N_TRIALS=2 STEPS=500 EFF_CLIPS=4 PHYS_BS=1 N_EVAL=800 N_TRAIN=4000 \
  MIN_VIDEO_LEN=31 CLIP_LEN=16 FRAME_STRIDE=2 TEMPORAL_KIND=mamba \
  OUT_JSON=results/seg/zoo_mamba_T16_s2_car.json \
  $PY scripts/run_temporal_protocol.py 2>&1 | tee -a "$LOG"

echo "########## CAR PIPELINE DONE  $(date '+%F %T') ##########" | tee -a "$LOG"
