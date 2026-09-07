#!/bin/bash
# Espera o MARL 10M terminar e roda a suite temporal (bake-off de memoria).
set -u
BASE=/mnt/c/Users/Acer/Downloads/MLE
LOG=$BASE/jax_port/marl_10m.log
TMP_LOG=$BASE/jax_port/temporal_grade.log
PY=/root/procgen-jax/bin/python
export PYTHONPATH=$BASE
# espera o marcador final do script MARL
until grep -q "TODOS CONCLUIDOS" "$LOG" 2>/dev/null; do sleep 60; done
echo "=== MARL 10M concluido; iniciando suite temporal $(date) ===" >> "$TMP_LOG"
cd "$BASE/jax_port"
$PY run_grade.py --suite temporal --seeds 42 43 44 45 46 \
  --timesteps 100000 --eval-full \
  --master "$BASE/jax_port/results_grade/master_temporal.json" \
  >> "$TMP_LOG" 2>&1
echo "=== temporal concluida $(date) ===" >> "$TMP_LOG"
