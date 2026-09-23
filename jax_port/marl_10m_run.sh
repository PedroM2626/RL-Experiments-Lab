#!/bin/bash
# Recurrent MARL 10M: IPPO then QMIX, sequential (1 GPU process at a time).
# Run from inside WSL. Override REPO (checkout root) and/or PY (venv python) if
# they are not in the default place; the hyperparameters below are the ones that
# produced the cells documented in README section 15.4.4.
set -u
BASE="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
PY="${PY:-/root/procgen-jax/bin/python}"
export PYTHONPATH="$BASE"
LR="${LR:-4e-3}"
ENT="${ENT:-0.0}"
SEEDS="${SEEDS:-42 43 44}"
LOG="$BASE/jax_port/marl_10m.log"
OUT="$BASE/jax_port/results_grade/marl"
mkdir -p "$OUT"
for seed in $SEEDS; do
  echo "=== IPPO-rec seed $seed 10M $(date) ===" >> "$LOG"
  $PY -m jax_port.marl.ppo_marl --algo ippo --recurrent --lr "$LR" --ent "$ENT" \
      --map 3m --timesteps 10000000 --seed "$seed" \
      --out "$OUT/ippo_rec__3m__seed${seed}__10M.json" >> "$LOG" 2>&1
  echo "=== QMIX-rec seed $seed 10M $(date) ===" >> "$LOG"
  $PY -m jax_port.marl.train_ql --algo qmix --recurrent \
      --map 3m --timesteps 10000000 --seed "$seed" \
      --out "$OUT/qmix_rec__3m__seed${seed}__10M.json" >> "$LOG" 2>&1
done
echo "=== ALL COMPLETED $(date) ===" >> "$LOG"
