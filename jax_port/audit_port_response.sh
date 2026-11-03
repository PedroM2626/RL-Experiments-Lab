#!/usr/bin/env bash
# Port-grid response audit (01/11/2026): re-train the port's `main` suite -- 16 configs x
# bossfight/starpilot/dodgeball x seeds 42-46, 100k steps, the grid's own 64 x 128 recipe and
# --eval-full protocol -- with the cross-game response measure attached to every cell.
# Folded by jax_port/summarize_port_response.py into jax_port/port_response_grid.json.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/root/procgen-jax/bin/python}"
OUT="${OUT:-/root/rg_response}"
export PYTHONPATH="$REPO" XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.45}"
"$PY" -u -m jax_port.run_grade --suite main --seeds 42 43 44 45 46 --timesteps 100000 \
  --eval-full --response --out-dir "$OUT" --master "$OUT/master.json"
