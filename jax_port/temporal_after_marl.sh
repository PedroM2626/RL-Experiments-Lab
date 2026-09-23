#!/bin/bash
# Waits for MARL 10M to finish and runs the temporal suite (memory bake-off).
# Override REPO (checkout root) and PY (venv python) as needed.
set -u
BASE="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
PY="${PY:-/root/procgen-jax/bin/python}"
export PYTHONPATH="$BASE"
LOG="$BASE/jax_port/marl_10m.log"
TMP_LOG="$BASE/jax_port/temporal_grade.log"
# waits for final marker from MARL script
until grep -q "ALL COMPLETED" "$LOG" 2>/dev/null; do sleep 60; done
echo "=== MARL 10M completed; starting temporal suite $(date) ===" >> "$TMP_LOG"
# Invoked as a module from the checkout root: an earlier version did `cd jax_port`
# first, which made run_grade.py write its cells into jax_port/jax_port/results_grade.
"$PY" -m jax_port.run_grade --suite temporal --seeds 42 43 44 45 46 \
  --timesteps 100000 --eval-full \
  --master "$BASE/jax_port/results_grade/master_temporal.json" \
  >> "$TMP_LOG" 2>&1
echo "=== temporal completed $(date) ===" >> "$TMP_LOG"
