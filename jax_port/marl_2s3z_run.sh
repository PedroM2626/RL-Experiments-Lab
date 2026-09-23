#!/bin/bash
# MARL positive control: QMIX-rec 10M on 2s3z (JaxMARL paper solves).
# Override REPO (checkout root), PY (venv python) or SEEDS as needed.
set -u
BASE="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
PY="${PY:-/root/procgen-jax/bin/python}"
export PYTHONPATH="$BASE"
SEEDS="${SEEDS:-42 43 44}"
LOG="$BASE/jax_port/marl_2s3z.log"
OUT="$BASE/jax_port/results_grade/marl"
mkdir -p "$OUT"
for seed in $SEEDS; do
  echo "=== QMIX-rec 2s3z seed $seed 10M $(date) ===" >> "$LOG"
  $PY -m jax_port.marl.train_ql --algo qmix --recurrent --map 2s3z \
      --timesteps 10000000 --seed "$seed" \
      --out "$OUT/qmix_rec__2s3z__seed${seed}__10M.json" >> "$LOG" 2>&1
done
echo "=== 2s3z COMPLETED $(date) ===" >> "$LOG"