"""Multi-seed expansion of the Caveflyer stack=1 POMDP occlusion benchmark (README section 15.4.8).

Runs 5 seeds (42-46) across 4 architectures:
  - classic (feedforward NatureCNN)
  - recurrent_lstm (step carry)
  - regularized_recurrent_lstm (LayerNorm + weight decay)
  - recurrent_s5 (S5 state space model)
Computes mean across seeds, sample std, Student's t 95% CI, and pairwise Cohen's d vs classic.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from jax_port.bench_temporal_stack1 import run_arm


MODELS = ["classic", "recurrent_lstm", "regularized_recurrent_lstm", "recurrent_s5"]
SEEDS = [42, 43, 44, 45, 46]


def cohens_d(x, y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pool_var = (((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)
    if pool_var <= 1e-12:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / np.sqrt(pool_var))


def stats_df4(vals):
    n = len(vals)
    m = float(np.mean(vals))
    s = float(np.std(vals, ddof=1)) if n > 1 else 0.0
    t_crit = 2.776 if n == 5 else 2.0  # t_crit for df=4, 95%
    half = t_crit * s / math.sqrt(n) if n > 1 else 0.0
    return m, s, [round(m - half, 3), round(m + half, 3)]


def main():
    parser = argparse.ArgumentParser(description="Multi-seed Caveflyer benchmark")
    parser.add_argument("--timesteps", type=int, default=57344)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--eval-eps", type=int, default=20)
    parser.add_argument("--output", type=str, default=os.path.join(BASE, "results", "caveflyer_multiseed_bench.json"))
    args = parser.parse_args()

    # Create dummy namespace expected by run_arm
    class NS:
        pass

    ns = NS()
    ns.game = "caveflyer"
    ns.timesteps = args.timesteps
    ns.distribution = "easy"
    ns.num_envs = 64
    ns.rollout = 128
    ns.minibatch = 1024
    ns.eval_envs = 16
    ns.eval_eps = args.eval_eps

    raw_results = {}
    if os.path.exists(args.output):
        try:
            with open(args.output, "r", encoding="utf-8") as f:
                saved = json.load(f)
                raw_results = saved.get("per_seed", {})
        except Exception:
            pass

    for m in args.models:
        raw_results.setdefault(m, {})
        for s in args.seeds:
            if str(s) in raw_results[m]:
                print(f"Skip {m} seed {s} (already measured)")
                continue
            print(f"Running {m} seed {s}...")
            r = run_arm(m, ns, seed=s)
            raw_results[m][str(s)] = r

    summary = {}
    classic_scores = [raw_results["classic"][str(s)]["eval_unseen_mean"] for s in args.seeds]
    for m in args.models:
        scores = [raw_results[m][str(s)]["eval_unseen_mean"] for s in args.seeds]
        mean, std, ci95 = stats_df4(scores)
        d = cohens_d(scores, classic_scores) if m != "classic" else 0.0
        summary[m] = {
            "mean": round(mean, 3),
            "std": round(std, 3),
            "ci95": ci95,
            "cohens_d_vs_classic": round(d, 3),
            "per_seed": {s: raw_results[m][str(s)]["eval_unseen_mean"] for s in args.seeds},
        }

    payload = {
        "_provenance": {
            "game": "caveflyer",
            "stack": 1,
            "timesteps": args.timesteps,
            "seeds": args.seeds,
            "eval_eps_per_seed": args.eval_eps,
        },
        "summary": summary,
        "per_seed": raw_results,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 75)
    print(f"CAVEFLYER MULTI-SEED (n={len(args.seeds)} SEEDS) BENCHMARK SUMMARY")
    print("=" * 75)
    print(f"{'Model':<28} {'Mean±Std':>12} {'95% CI':>18} {'Cohen d vs Classic':>20}")
    print("-" * 75)
    for m, s in summary.items():
        ci_str = f"[{s['ci95'][0]:.2f}, {s['ci95'][1]:.2f}]"
        m_std = f"{s['mean']:.2f}±{s['std']:.2f}"
        d_str = f"{s['cohens_d_vs_classic']:+.2f}" if m != "classic" else "— (baseline)"
        print(f"{m:<28} {m_std:>12} {ci_str:>18} {d_str:>20}")
    print("=" * 75)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
