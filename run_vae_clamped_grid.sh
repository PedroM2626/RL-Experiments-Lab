#!/usr/bin/env bash
# The vae_clamped audit arm (01/11/2026): 3 games x seeds 42-46 at the 16x256 recipe, one
# process per cell (gym3 finalizers, see audit_suite_validity.run_arm), then merged per game.
set -u
cd "$(dirname "$0")"
for game in starpilot bossfight dodgeball; do
  for seed in 42 43 44 45 46; do
    py -3.10 -u audit_suite_validity.py --game "$game" --seed "$seed" --n-envs 16 --n-steps 256 \
      --configs vae_clamped --out "logs_audit/shards_vaeclamped/${game}_${seed}.json"
  done
  py -3.10 audit_suite_validity.py --merge "logs_audit/shards_vaeclamped/${game}_*.json" --key-by-seed \
    --out "results/suite_validity_nenv16_vae_clamped_${game}.json"
done
