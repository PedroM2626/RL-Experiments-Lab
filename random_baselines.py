"""Empirical random-policy baselines for Procgen, used as the lower anchor of the
canonical (Agarwal et al., 2021) normalization in run_rliable_eval.py.

Protocol is the same unseen-level evaluation used for trained models:
num_levels=0, distribution_mode='easy', eval seed = --seed + 1000 (re_eval_100.py:20).

Saves results/random_baselines.json incrementally, one game at a time.
"""

import argparse
import json
import os
import time

import numpy as np

from procgen_wrapper import make_procgen_env

DEFAULT_GAMES = ["bossfight", "starpilot", "dodgeball", "maze", "heist"]


def measure_game(game, episodes, seed, max_steps=None):
    env = make_procgen_env(game, num_levels=0, distribution_mode="easy", seed=seed + 1000)
    rng = np.random.default_rng(seed)
    returns = []
    try:
        for _ in range(episodes):
            obs, _ = env.reset()
            total, steps = 0.0, 0
            done = False
            while not done:
                obs, reward, terminated, truncated, _ = env.step(
                    int(rng.integers(env.action_space.n)))
                total += float(reward)
                steps += 1
                done = terminated or truncated or (max_steps is not None and steps >= max_steps)
            returns.append(total)
    finally:
        env.close()
    arr = np.asarray(returns, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "n_episodes": int(arr.size),
        "episodes": [round(float(r), 4) for r in returns],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", nargs="+", default=DEFAULT_GAMES)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=None,
                        help="safety cap per episode (Procgen already enforces its own limit)")
    parser.add_argument("--out", type=str, default="results/random_baselines.json")
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(base, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    results = {}
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        # only reuse entries that carry the full protocol record
        results = {k: v for k, v in existing.items() if "episodes" in v}

    results["_protocol"] = {
        "produced_by": "random_baselines.py",
        "eval_seed": args.seed + 1000,
        "distribution_mode": "easy",
        "num_levels": 0,
        "episodes_per_game": args.episodes,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    for game in args.games:
        if game in results:
            print(f"{game}: already measured (mean={results[game]['mean']:.3f}), skipping")
            continue
        t0 = time.time()
        results[game] = measure_game(game, args.episodes, args.seed, args.max_steps)
        print(f"{game}: {results[game]['mean']:.3f} ± {results[game]['std']:.3f} "
              f"over {results[game]['n_episodes']} eps ({time.time() - t0:.0f}s)", flush=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
