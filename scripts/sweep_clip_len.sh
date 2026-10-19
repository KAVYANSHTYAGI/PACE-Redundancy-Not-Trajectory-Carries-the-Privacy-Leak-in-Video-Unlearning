#!/usr/bin/env bash
# Leakage-vs-window-length scaling sweep. ADVR-T was trained against a 3-frame
# temporal discriminator (--clip_len 3); the headline temporal_attack.json
# result also evaluated at CLIP_LEN=3 (the default). Before trusting "ADVR-T
# defeats the temporal attacker" as a general claim, we need to know whether
# that holds at window lengths ADVR-T never trained against -- otherwise the
# result could just be "it learned to fool the exact judge it played against".
#
# We only rerun the TEMPORAL surface (SURFACES=temporal) at each window length;
# the frame-attacker numbers don't depend on CLIP_LEN and are already in
# results/seg/temporal_attack.json (the CLIP_LEN=3 run, which also has both
# surfaces and is the reference point for this sweep at T=3).
#
# Usage: bash scripts/sweep_clip_len.sh
set -e
set -o pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=. PYTORCH_ALLOC_CONF=expandable_segments:True
PY=/home/kavyansh/anaconda3/envs/myenv/bin/python
mkdir -p results/seg
LOG=results/seg/sweep_clip_len.log; : > "$LOG"
UNLEARNED="ADVR,ADVR-T,bl_SSD_s1,bl_SalUn_s1,bl_DELETE_s1"

# clip_len -> batch size (bigger windows need smaller batches to fit the 16GB card)
declare -A BS=( [2]=8 [5]=4 [8]=2 [16]=1 )

for T in 2 5 8 16; do
  echo "=========== CLIP_LEN=$T (bs=${BS[$T]})  $(date '+%F %T') ===========" | tee -a "$LOG"
  CLIP_LEN=$T BS_CLIP=${BS[$T]} SURFACES=temporal UNLEARNED="$UNLEARNED" \
    OUT_JSON="results/seg/temporal_attack_T${T}.json" \
    $PY scripts/run_temporal_attack.py 2>&1 | tee -a "$LOG"
done

echo "############ SWEEP DONE  $(date '+%F %T') ############" | tee -a "$LOG"
