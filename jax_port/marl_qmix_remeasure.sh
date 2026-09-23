#!/bin/bash
# Re-measure the recurrent QMIX cells of README section 15.4.4 under the corrected
# Bellman target, per-environment sequential buffer and terminal masking (fixed 21/09/2026).
#
# Why these cells: section 15.4.4 explains a 0.0 win-rate as a property of symmetric 3v3
# combat ("partial-credit gradients form a local optimum"). That diagnosis was read off runs
# whose QMIX mixer never received r + gamma*(1-d)*Q_tot_target, i.e. runs with no external
# reward signal in the mixer loss at all — the failure it explains may have been the bug.
# Only the Q-learning arms are affected: IPPO/ippo-recurrent are on-policy actor-critic and
# never touch the replay buffer, the mixer or the bootstrap mask, so their published cells
# stay valid and are not re-run here.
#
# Hyperparameters are deliberately identical to marl_10m_run.sh (no --tau, train_ql's own
# lr default) so the only difference against the published cells is the corrected code.
#
# Run from inside WSL. Overrides: REPO, PY, MAPS, SEEDS, TIMESTEPS, OUT.
set -u
BASE="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
PY="${PY:-/root/procgen-jax/bin/python}"
export PYTHONPATH="$BASE"
MAPS="${MAPS:-3m}"
SEEDS="${SEEDS:-42 43 44}"
TIMESTEPS="${TIMESTEPS:-10000000}"
OUT="${OUT:-$BASE/jax_port/results_grade/marl_remeasure}"
LOG="$OUT/marl_remeasure.log"
mkdir -p "$OUT"
echo "=== QMIX-rec re-measure start $(date) maps=[$MAPS] seeds=[$SEEDS] steps=$TIMESTEPS ===" >> "$LOG"
for map in $MAPS; do
  for seed in $SEEDS; do
    cell="$OUT/qmix_rec__${map}__seed${seed}__${TIMESTEPS}.json"
    if [ -f "$cell" ]; then
      echo "=== skip $map seed $seed (already measured) $(date) ===" >> "$LOG"
      continue
    fi
    echo "=== QMIX-rec $map seed $seed $TIMESTEPS $(date) ===" >> "$LOG"
    $PY -m jax_port.marl.train_ql --algo qmix --recurrent \
        --map "$map" --timesteps "$TIMESTEPS" --seed "$seed" \
        --out "$cell" >> "$LOG" 2>&1 \
      || echo "=== FAILED $map seed $seed (see $LOG) ===" >> "$LOG"
  done
done
echo "=== QMIX-rec re-measure done $(date) ===" >> "$LOG"
$PY -m jax_port.marl_remeasure_report --cells_dir "$OUT" \
    --out "$BASE/jax_port/marl_remeasure_summary.json" 2>&1 | tee -a "$LOG"
