#!/usr/bin/env bash
# Full project3 pipeline, resumable (each stage skips if its output exists).
# Assumes the reference models (vspw_original / vspw_oracle) are already trained
# by run_baseline.sh -- this script does everything downstream of them.
#
#   ssh 5090  (or run locally)
#   cd project3 && bash scripts/run_all.sh
#
# Stages:
#   1. baselines        -> checkpoints/unlearned_bl_*.pth
#   2. ADVR   (per-frame) -> checkpoints/unlearned_ADVR.pth
#   3. ADVR-T (temporal)  -> checkpoints/unlearned_ADVR-T.pth
#   4. benchmark        -> results/seg/benchmark.json   (utility + per-frame leak)
#   5. temporal attack  -> results/seg/temporal_attack.json  (the headline)
set -e
set -o pipefail
cd "$(dirname "$0")/.."
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONPATH=.
PY=/home/kavyansh/anaconda3/envs/myenv/bin/python
mkdir -p results/seg
LOG=results/seg/run_all.log; : > "$LOG"

need_refs () {
  for r in original oracle; do
    [ -f "checkpoints/vspw_${r}_deeplabv3r50_cls60.pth" ] || {
      echo "[run_all] MISSING reference vspw_${r} -- run scripts/run_baseline.sh first" | tee -a "$LOG"
      exit 1; }
  done
}
need_refs

echo "=========== 1. baselines  $(date '+%F %T') ===========" | tee -a "$LOG"
$PY scripts/run_baselines.py 2>&1 | tee -a "$LOG"

echo "=========== 2. ADVR (per-frame)  $(date '+%F %T') ===========" | tee -a "$LOG"
[ -f checkpoints/unlearned_ADVR.pth ] || \
  $PY scripts/test_candidate.py --name ADVR --epochs 4 --lam_r 1.0 --lam_adv 1.0 \
      --save_as ADVR --no_eval 2>&1 | tee -a "$LOG"

echo "=========== 3. ADVR-T (temporal)  $(date '+%F %T') ===========" | tee -a "$LOG"
# bs=2 clips * clip_len=3 frames = 6 frames/batch, matching ADVR's per-batch frame
# count -- bs=6 (default) OOMed the 16GB card (18 frames/batch through 2 models).
[ -f checkpoints/unlearned_ADVR-T.pth ] || \
  $PY scripts/test_candidate.py --name ADVR-T --epochs 4 --clip_len 3 --bs 2 \
      --lam_r 1.0 --lam_adv 1.0 --lam_tc 0.5 --save_as ADVR-T --no_eval 2>&1 | tee -a "$LOG"

echo "=========== 4. benchmark (utility + per-frame leak)  $(date '+%F %T') ===========" | tee -a "$LOG"
$PY scripts/run_benchmark.py 2>&1 | tee -a "$LOG"

echo "=========== 5. temporal attack (headline)  $(date '+%F %T') ===========" | tee -a "$LOG"
UNLEARNED="ADVR,ADVR-T,bl_SSD_s1,bl_SalUn_s1,bl_DELETE_s1" \
  $PY scripts/run_temporal_attack.py 2>&1 | tee -a "$LOG"

echo "############ PIPELINE DONE  $(date '+%F %T') ############" | tee -a "$LOG"
