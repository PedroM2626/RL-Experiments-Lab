#!/bin/bash
# Re-measure the SMAX Q-learning cells of README section 15.4 under the corrected
# Bellman target, per-environment sequential buffer and terminal masking (fixed 21/09/2026).
#
# Why these cells: section 15.4 reports a 0.0 win-rate for every Q-learning arm and section
# 15.4.4 explains it as a property of symmetric 3v3 combat ("partial-credit gradients form a
# local optimum"). Those runs were produced before three fixes:
#   - the QMIX mixer target omitted r + gamma*(1-d)*Q_tot, i.e. the mixer loss had no
#     external reward signal at all,
#   - the recurrent replay buffer interleaved parallel environments, so a sampled sequence
#     was not a contiguous trajectory,
#   - the bootstrap ignored terminations, so terminal states were bootstrapped.
# VDN is affected by the terminal mask only; QMIX by all three. IPPO/MAPPO/MAPoCA/CTE/TarMAC
# are on-policy or paradigm trainers that never touch this buffer, mixer or mask, so their
# published cells stay valid and are not re-run here.
#
# Hyperparameters are deliberately identical to marl_10m_run.sh and run_grade.py's marl suite
# (no --tau, train_ql's own lr default), so the corrected code is the only difference against
# the published cells.
#
# Run from inside WSL. Overrides: REPO, PY, MAPS, SEEDS, TIMESTEPS_FLAT, TIMESTEPS_REC, OUT.
set -u
BASE="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
PY="${PY:-/root/procgen-jax/bin/python}"
export PYTHONPATH="$BASE"
MAPS="${MAPS:-3m}"
SEEDS="${SEEDS:-42 43 44}"
TIMESTEPS_FLAT="${TIMESTEPS_FLAT:-1000000}"
TIMESTEPS_REC="${TIMESTEPS_REC:-10000000}"
OUT="${OUT:-$BASE/jax_port/results_grade/marl_remeasure}"
LOG="$OUT/marl_remeasure.log"
mkdir -p "$OUT"

run_cell() { # $1=algo $2=recurrent|flat $3=timesteps $4=map $5=seed
  local algo="$1" mode="$2" steps="$3" map="$4" seed="$5"
  local tag="$algo" cell flags=""
  [ "$mode" = "recurrent" ] && { tag="$algo-recurrent"; flags="--recurrent"; }
  cell="$OUT/${tag}__${map}__seed${seed}__${steps}.json"
  if [ -f "$cell" ]; then
    echo "=== skip $tag $map seed $seed (already measured) $(date) ===" >> "$LOG"
    return
  fi
  echo "=== $tag $map seed $seed $steps $(date) ===" >> "$LOG"
  # shellcheck disable=SC2086
  $PY -m jax_port.marl.train_ql --algo "$algo" $flags \
      --map "$map" --timesteps "$steps" --seed "$seed" --out "$cell" >> "$LOG" 2>&1 \
    || echo "=== FAILED $tag $map seed $seed (see $LOG) ===" >> "$LOG"
}

echo "=== QMIX/VDN re-measure start $(date) maps=[$MAPS] seeds=[$SEEDS] flat=$TIMESTEPS_FLAT rec=$TIMESTEPS_REC ===" >> "$LOG"
for map in $MAPS; do
  for seed in $SEEDS; do
    run_cell vdn  flat      "$TIMESTEPS_FLAT" "$map" "$seed"
    run_cell qmix flat      "$TIMESTEPS_FLAT" "$map" "$seed"
    run_cell qmix recurrent "$TIMESTEPS_REC"  "$map" "$seed"
  done
done
echo "=== QMIX/VDN re-measure done $(date) ===" >> "$LOG"
$PY -m jax_port.marl_remeasure_report --cells_dir "$OUT" \
    --out "$BASE/jax_port/marl_remeasure_summary.json" 2>&1 | tee -a "$LOG"
