#!/bin/bash
# Waits for MARL 10M to finish and runs temporal suite (memory bake-off).
set -u
BASE=/mnt/c/Users/Acer/Downloads/MLE
LOG=$BASE/jax_port/marl_10m.log
TMP_LOG=$BASE/jax_port/temporal_grade.log
PY=/root/procgen-jax/bin/python
export PYTHONPATH=$BASE
# waits for final marker from MARL script
until grep -q "ALL COMPLETED" "$LOG" 2>/dev/null; do sleep 60; done
echo "=== MARL 10M completed; starting temporal suite $(date) ===" >> "$TMP_LOG"
cd "$BASE/jax_port"
$PY run_grade.py --suite temporal --seeds 42 43 44 45 46 \
  --timesteps 100000 --eval-full \
  --master "$BASE/jax_port/results_grade/master_temporal.json" \
  >> "$TMP_LOG" 2>&1
echo "=== temporal completed $(date) ===" >> "$TMP_LOG"
