#!/bin/bash
# MARL 10M recorrente: IPPO depois QMIX, sequencial (1 proc GPU por vez).
set -u
export PYTHONPATH=/mnt/c/Users/Acer/Downloads/MLE
PY=/root/procgen-jax/bin/python
BASE=/mnt/c/Users/Acer/Downloads/MLE
LOG=$BASE/jax_port/marl_10m.log
OUT=$BASE/jax_port/results_grade/marl
mkdir -p "$OUT"
for seed in 42 43 44; do
  echo "=== IPPO-rec seed $seed 10M $(date) ===" >> "$LOG"
  $PY -m jax_port.marl.ppo_marl --algo ippo --recurrent --lr 4e-3 --ent 0.0 \
      --map 3m --timesteps 10000000 --seed "$seed" \
      --out "$OUT/ippo_rec__3m__seed${seed}__10M.json" >> "$LOG" 2>&1
  echo "=== QMIX-rec seed $seed 10M $(date) ===" >> "$LOG"
  $PY -m jax_port.marl.train_ql --algo qmix --recurrent \
      --map 3m --timesteps 10000000 --seed "$seed" \
      --out "$OUT/qmix_rec__3m__seed${seed}__10M.json" >> "$LOG" 2>&1
done
echo "=== TODOS CONCLUIDOS $(date) ===" >> "$LOG"
