#!/usr/bin/env bash
# Evaluate the learned permutation-invariant DeepSets attacker against the
# existing checkpoints (no new generator training needed). Tests whether
# PACE's suppression is specific to a fixed-mean-of-raw-logits adversary
# (PoolDiscriminator) or survives a strictly more expressive order-invariant
# attacker (learned per-frame embedding, then mean-pool, then decode).
set -e
set -o pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=. FORGET_CLS=60
PY=/home/kavyansh/anaconda3/envs/myenv/bin/python
LOG=results/seg/deepsets_eval.log
: > "$LOG"

CLIP_LEN=16 FRAME_STRIDE=2 MIN_VIDEO_LEN=31 N_TRIALS=2 STEPS=500 EFF_CLIPS=4 \
N_EVAL=800 N_TRAIN=4000 TEMPORAL_KIND=deepsets \
MODELS=oracle_FLOOR,original,ADVR,ADVR-T,ADVR-P \
OUT_JSON=results/seg/zoo_deepsets_T16_s2_person.json \
$PY -u scripts/run_temporal_protocol.py 2>&1 | tee -a "$LOG"

echo "########## DEEPSETS EVAL DONE $(date) ##########" | tee -a "$LOG"
