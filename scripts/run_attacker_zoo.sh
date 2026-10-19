#!/usr/bin/env bash
# Attacker zoo + temporal-span sweep: does trajectory SYNERGY exist at all?
#
# WHY. The corrected protocol's first rows showed synergy = AP(ordered) -
# AP(time-shuffled) of +0.00 on the UNDEFENDED model at T=3. If that holds
# everywhere, "temporal silhouette leakage" is not synergy at all -- it is
# order-free redundancy pooling, and the thesis has to be restated. Before
# accepting that, two things have to be ruled out, because both would suppress
# synergy for reasons that have nothing to do with the model:
#
#   ATTACKER REACH. TemporalSilhouetteNet's temporal receptive field is +/-3
#   frames (measured by perturbation). At T=16 it cannot integrate more than 7
#   of the 16 frames, so it is structurally unable to find long-range
#   trajectory structure. VideoMambaAttacker (selective SSM, official mamba_ssm
#   CUDA kernel) reaches the whole clip -- with 0.08M params vs conv3d's 0.44M,
#   so any difference is reach, not capacity.
#
#   TEMPORAL SPAN. 16 CONSECUTIVE VSPW frames span ~1.1s, in which a walking
#   person barely displaces; there is almost no trajectory to recover regardless
#   of attacker. frame_stride=2 spans ~2.1s at identical T and compute.
#   min_video_len is pinned to 31 in BOTH arms so the anchor set -- and hence the
#   evaluated content -- is identical across strides (cf. POSITIONING.md 8.2).
#
# Phase A is the decisive, cheap test: strongest attacker x longest span on the
# undefended model, where trajectory information is maximally available. If
# synergy is ~0 THERE, it is ~0 everywhere and the reframing is forced.
# Phase B runs the full zoo only if it is worth running.
#
# Usage: bash scripts/run_attacker_zoo.sh [A|B|AB]   (default AB)
set -e
set -o pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=. PYTORCH_ALLOC_CONF=expandable_segments:True
PY=/home/kavyansh/anaconda3/envs/myenv/bin/python
mkdir -p results/seg
LOG=results/seg/attacker_zoo.log
PHASE="${1:-AB}"

# identical for every arm so the comparison is clean
export N_TRIALS=2 STEPS=500 EFF_CLIPS=4 PHYS_BS=1 N_EVAL=800 N_TRAIN=4000 \
       MIN_VIDEO_LEN=31 CLIP_LEN=16

if [[ "$PHASE" == *A* ]]; then
  echo "########## PHASE A: does synergy exist?  $(date '+%F %T') ##########" | tee -a "$LOG"
  for S in 1 2; do
    echo "===== mamba | T=16 stride=$S =====" | tee -a "$LOG"
    TEMPORAL_KIND=mamba FRAME_STRIDE=$S MODELS="oracle_FLOOR,original" \
      OUT_JSON="results/seg/zoo_mamba_T16_s${S}.json" \
      $PY scripts/run_temporal_protocol.py 2>&1 | tee -a "$LOG"
  done
fi

if [[ "$PHASE" == *B* ]]; then
  echo "########## PHASE B: full zoo @ 2.1s span  $(date '+%F %T') ##########" | tee -a "$LOG"
  for K in conv3d convgru mamba; do
    echo "===== $K | T=16 stride=2 =====" | tee -a "$LOG"
    TEMPORAL_KIND=$K FRAME_STRIDE=2 MODELS="oracle_FLOOR,original,ADVR,ADVR-T" \
      OUT_JSON="results/seg/zoo_${K}_T16_s2.json" \
      $PY scripts/run_temporal_protocol.py 2>&1 | tee -a "$LOG"
  done
fi

echo "########## ZOO DONE  $(date '+%F %T') ##########" | tee -a "$LOG"
