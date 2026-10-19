#!/usr/bin/env bash
# Corrected leakage-vs-window-length sweep + time-shuffle synergy control.
#
# Supersedes sweep_clip_len.sh, whose results are NOT usable: it varied the
# evaluation content with T (36 distinct val videos at T=2 vs 238 at T=16) and
# shrank the attacker's budget with T (4000 clips at T=2 vs 500 at T=16). Both
# push leakage down as T grows for every model, which is why the retrain oracle
# floor -- a quantity independent of T -- appeared to drift 17.7 -> 0.0.
#
# Here every T sees the SAME anchors, the SAME number of optimizer steps, and
# the SAME effective batch in clips, and is scored threshold-free (AP).
#
# Usage: bash scripts/sweep_protocol.sh
set -e
set -o pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=. PYTORCH_ALLOC_CONF=expandable_segments:True
PY=/home/kavyansh/anaconda3/envs/myenv/bin/python
mkdir -p results/seg
LOG=results/seg/sweep_protocol.log; : > "$LOG"

MODELS="oracle_FLOOR,original,ADVR,ADVR-T,bl_SalUn_s1"
export MODELS N_TRIALS=2 STEPS=500 EFF_CLIPS=4 N_EVAL=1000 N_TRAIN=6000

for T in 3 2 5 8 16; do            # T=3 first: it is the trained-against horizon
  WF=0; [ "$T" = "3" ] && WF=1     # per-frame surface is T-independent, run once
  echo "=========== CLIP_LEN=$T  $(date '+%F %T') ===========" | tee -a "$LOG"
  CLIP_LEN=$T WITH_FRAME=$WF $PY scripts/run_temporal_protocol.py 2>&1 | tee -a "$LOG"
done

echo "############ PROTOCOL SWEEP DONE  $(date '+%F %T') ############" | tee -a "$LOG"
