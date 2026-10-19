#!/usr/bin/env bash
# Retry helper: VSPW's Google Drive files hit a shared per-file daily quota
# ("Too many users have viewed or downloaded this file recently") that is not
# under our control and can take hours to clear. Re-run this script to retry;
# gdown --continue resumes any partial download rather than restarting.
set -e
cd "$(dirname "$0")/.."
mkdir -p data
cd data
LOG=../results/vspw_download.log
mkdir -p ../results
echo "[$(date)] attempting VSPW full-resolution download" | tee -a "$LOG"
/home/kavyansh/anaconda3/envs/myenv/bin/gdown --continue \
  "14yHWsGneoa1pVdULFk7cah3t-THl7yEz" -O VSPW_full.zip 2>&1 | tee -a "$LOG"
if [ -f VSPW_full.zip ] && [ "$(stat -c%s VSPW_full.zip 2>/dev/null || echo 0)" -gt 40000000000 ]; then
  echo "[$(date)] download appears complete, extracting" | tee -a "$LOG"
  unzip -q VSPW_full.zip
  echo "[$(date)] extraction done" | tee -a "$LOG"
  PYTHONPATH=.. /home/kavyansh/anaconda3/envs/myenv/bin/python ../seg/data.py 2>&1 | tee -a "$LOG"
else
  echo "[$(date)] download not yet complete or failed -- retry later" | tee -a "$LOG"
fi
