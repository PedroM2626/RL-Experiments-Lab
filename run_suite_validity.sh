#!/bin/bash
# Run the 16-architecture validity audit (README 3.6.2, section 6 item 9) over several games.
#
#   bash run_suite_validity.sh starpilot coinrun
#   GAMES="bossfight dodgeball" WORKERS=2 bash run_suite_validity.sh
#
# Shape of the run, and why:
#  * one PROCESS PER ARM -- a long-lived shard deadlocks in cffi once it has built enough
#    Procgen envs (gym3 Libenv.__del__ closing during accessor construction: the process sits
#    at 0% CPU forever, no exception). Section 5 documents the incident; audit_suite_validity
#    now also collects at every arm boundary.
#  * WORKERS concurrent processes share this GPU without slowing each other much (measured
#    142-200 FPS at 3 workers, ~10 min per 100k-step cell, ~8 min on bossfight).
#  * TIMEOUT kills a hung cell so it cannot stall the pool; the arm is then simply absent from
#    the shards and a re-run of this script measures it again (each cell is written before the
#    next starts, and an existing shard is skipped).
#  * each game is fused into its own artifact at the end, so a bossfight row can never be
#    mistaken for a dodgeball row of the same arm (audit_suite_validity.merge asserts it).
set -u
cd "$(dirname "$0")" || exit 1

GAMES="${*:-${GAMES:-starpilot bossfight dodgeball}}"
WORKERS="${WORKERS:-3}"
TIMEOUT="${TIMEOUT:-3600}"
SHARDS="${SHARDS:-logs_audit/shards}"
mkdir -p "$SHARDS"

# The arm list comes from the audit module, which builds it from compare_suite and
# compare_new_archs themselves -- this script cannot drift from the published configurations.
ARMS=$(py -3.10 -c "import audit_suite_validity as A; print(' '.join(n for n, *_ in A.ARMS))")
echo "games: $GAMES | arms: $(echo $ARMS | wc -w) | workers: $WORKERS"

cell() { # game arm
  local g=$1 a=$2 f="$SHARDS/g-$g-$a.json"
  [ -e "$f" ] && { echo "skip $g/$a (already measured)"; return 0; }
  timeout "$TIMEOUT" py -3.10 audit_suite_validity.py --game "$g" --timesteps "${TIMESTEPS:-100000}" \
      --seed "${SEED:-42}" --configs "$a" --out "$f" > "$SHARDS/g-$g-$a.log" 2>&1
  echo "$(date -u +%H:%M:%SZ) $g/$a rc=$?"
}
export -f cell SHARDS TIMEOUT

for g in $GAMES; do
  for a in $ARMS; do echo "$g $a"; done
done | xargs -P "$WORKERS" -n 2 bash -c 'cell "$0" "$1"'

for g in $GAMES; do
  py -3.10 audit_suite_validity.py --merge "$SHARDS/g-$g-*.json" \
      --out "results/suite_validity_$g.json" || echo "MERGE FAILED for $g"
done
