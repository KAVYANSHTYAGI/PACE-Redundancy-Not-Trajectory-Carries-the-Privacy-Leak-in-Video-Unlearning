#!/usr/bin/env bash
# ADVR-TX lam_ord ablation (EXPERIMENTS.md sec 8, item 2a).
#
# WHY. Phase B (sec 5.5) showed ADVR-T's magnitude-capping L_tc already
# suppresses Delta_redundancy substantially (2.4-5x smaller than ADVR's under
# every attacker), even though it was designed against ORDER, which sec 5.2
# showed doesn't measurably leak here. ADVR-TX adds a second adversary
# (OrderDiscriminator, weight lam_ord) that specifically targets order. If
# order isn't the leak, that adversary may be pure overhead: extra compute,
# extra optimization difficulty, zero marginal privacy benefit. This ablation
# trains ADVR-TX twice, identical in every respect except lam_ord, and compares
# their Delta_redundancy/synergy under the SAME strongest-attacker protocol
# used for the sec 5.5 comparison (mamba, T=16, stride=2).
#
# Both arms use ADVR-T's exact training config (epochs=4, lam_tc=0.5,
# temporal_kind=conv3d, bs=2, clip_len=3 -- confirmed from
# checkpoints/unlearned_ADVR-T.pth's saved config) so lam_ord is the only
# thing that differs between ADVR-TX0 and ADVR-TX.
#
# Usage: bash scripts/run_advrtx_ablation.sh
set -e
set -o pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=. PYTORCH_ALLOC_CONF=expandable_segments:True
PY=/home/kavyansh/anaconda3/envs/myenv/bin/python
mkdir -p results/seg
LOG=results/seg/advrtx_ablation.log; : > "$LOG"

echo "=========== train ADVR-TX0 (lam_ord=0)  $(date '+%F %T') ===========" | tee -a "$LOG"
[ -f checkpoints/unlearned_ADVR-TX0.pth ] || \
  $PY scripts/test_candidate.py --name ADVR-TX --epochs 4 --clip_len 3 --bs 2 \
      --lam_r 1.0 --lam_adv 1.0 --lam_tc 0.5 --lam_ord 0.0 \
      --save_as ADVR-TX0 --no_eval 2>&1 | tee -a "$LOG"

echo "=========== train ADVR-TX (lam_ord=1.0)  $(date '+%F %T') ===========" | tee -a "$LOG"
[ -f checkpoints/unlearned_ADVR-TX.pth ] || \
  $PY scripts/test_candidate.py --name ADVR-TX --epochs 4 --clip_len 3 --bs 2 \
      --lam_r 1.0 --lam_adv 1.0 --lam_tc 0.5 --lam_ord 1.0 \
      --save_as ADVR-TX --no_eval 2>&1 | tee -a "$LOG"

echo "=========== eval both under mamba, T=16/stride=2  $(date '+%F %T') ===========" | tee -a "$LOG"
MODELS="oracle_FLOOR,original,ADVR,ADVR-T,ADVR-TX0,ADVR-TX" \
  N_TRIALS=2 STEPS=500 EFF_CLIPS=4 PHYS_BS=1 N_EVAL=800 N_TRAIN=4000 \
  MIN_VIDEO_LEN=31 CLIP_LEN=16 FRAME_STRIDE=2 TEMPORAL_KIND=mamba \
  OUT_JSON=results/seg/zoo_mamba_T16_s2_ablation.json \
  $PY scripts/run_temporal_protocol.py 2>&1 | tee -a "$LOG"

echo "########## ABLATION DONE  $(date '+%F %T') ##########" | tee -a "$LOG"
