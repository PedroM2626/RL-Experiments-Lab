#!/bin/bash
# MARL positive control: QMIX-rec 10M em 2s3z (o paper JaxMARL resolve).
set -u
export PYTHONPATH=/mnt/c/Users/Acer/Downloads/MLE
PY=/root/procgen-jax/bin/python
BASE=/mnt/c/Users/Acer/Downloads/MLE
LOG=$BASE/jax_port/marl_2s3z.log
OUT=$BASE/jax_port/results_grade/marl
mkdir -p "$OUT"
for seed in 42 43 44; do
  echo "=== QMIX-rec 2s3z seed $seed 10M $(date) ===" >> "$LOG"
  $PY -m jax_port.marl.train_ql --algo qmix --recurrent --map 2s3z \
      --timesteps 10000000 --seed "$seed" \
      --out "$OUT/qmix_rec__2s3z__seed${seed}__10M.json" >> "$LOG" 2>&1
done
echo "=== 2s3z CONCLUIDO $(date) ===" >> "$LOG"