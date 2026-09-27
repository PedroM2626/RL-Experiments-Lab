"""Summarize the scaled-bonus (normalized) exploration benchmark.

The published section 3.6 arms inject ~1e-7 per step at beta=0.01, which is seven orders of
magnitude below Procgen's 0/1 extrinsic reward: that grid cannot separate curiosity from PPO
and README reports it as a null by scale. This benchmark is the arm WITH the scale fixed
(models/bonuses.py divides the intrinsic error by its trailing-window mean), so the question
"does curiosity help maze/heist?" is finally testable.

It is a different benchmark, not a repair, so nothing here overwrites the frozen 24/09 files:

  * the three bonus arms are trained into logs_maze_heist_norm/ by
    ``py -3.10 run_exploration_remeasure.py --normalized``;
  * the `ppo` control is REUSED from results/exploration_remeasure.json, because normalizing
    a bonus that does not exist changes nothing -- the control arm is byte-identical code and
    seeds, so re-training it would spend 10 cells to measure the same number twice. The reuse
    is recorded in _provenance so the table can be read without opening this file;
  * the 100-episode protocol comes from results/eval100_normalized.json (re_eval_100.py over
    the norm checkpoints), which is optional until it has been run.

Each cell also carries the wrapper's own bookkeeping, so the table states what the arm
actually injected rather than what its hyperparameters claimed.

Usage:
    py -3.10 summarize_exploration_normalized.py
"""
import argparse
import glob
import json
import math
import os

BASE = os.path.dirname(os.path.abspath(__file__))
NORM_LOGS = os.path.join(BASE, "logs_maze_heist_norm")
FROZEN = os.path.join(BASE, "results", "exploration_remeasure.json")
EVAL100 = os.path.join(BASE, "results", "eval100_normalized.json")
OUT = os.path.join(BASE, "results", "exploration_normalized.json")
ARMS = ("ppo", "icm", "rnd", "ngu")
# Student t 0.975 quantile per sample count, for the same interval as the frozen file.
T975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 8: 2.365, 10: 2.262}


def ci95(values):
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return {"mean": round(mean, 3), "std": 0.0, "n": n, "ci95": [round(mean, 3)] * 2}
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = math.sqrt(var)
    t = T975.get(n, 1.96)
    half = t * std / math.sqrt(n)
    return {"mean": round(mean, 3), "std": round(std, 3), "n": n,
            "ci95": [round(mean - half, 3), round(mean + half, 3)]}


def norm_cells():
    """{(config, seed): cell} from every normalized run directory, later runs winning."""
    out = {}
    for path in sorted(glob.glob(os.path.join(NORM_LOGS, "maze_heist_*", "comparison_results.json"))):
        with open(path, encoding="utf-8") as f:
            j = json.load(f)
        run = os.path.basename(os.path.dirname(path))
        for config, cells in j.items():
            if config.startswith("_"):
                continue
            for c in cells:
                if c.get("mean_reward") is not None:
                    out[(config, c["seed"])] = dict(c, source_run=run)
    return out


def control_cells(frozen, seeds):
    """The ppo arms of the frozen 24/09 re-measurement, same code path as a normalized run."""
    out = {}
    for game in ("maze", "heist"):
        for c in frozen["per_seed"][f"{game}_ppo"]["cells"]:
            if c["seed"] in seeds:
                out[(f"{game}_ppo", c["seed"])] = {
                    "mean_reward": c["mean_reward_10eps"], "std_reward": c["std_reward_10eps"],
                    "bonus": None, "source_run": frozen["per_seed"][f"{game}_ppo"]["source_run"],
                }
    return out


def load_eval100():
    if not os.path.exists(EVAL100):
        return {}
    with open(EVAL100, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    eval100 = load_eval100()
    with open(FROZEN, encoding="utf-8") as f:
        frozen = json.load(f)

    measured = norm_cells()
    seeds = sorted({s for _c, s in measured})
    if not seeds:
        raise SystemExit(f"no normalized cells in {NORM_LOGS} yet -- run "
                         f"`py -3.10 run_exploration_remeasure.py --normalized` first")
    cells = control_cells(frozen, set(seeds))
    cells.update(measured)

    configs = [f"{g}_{a}" for g in ("maze", "heist") for a in ARMS]
    per_seed, stats10, stats100, scale = {}, {}, {}, {}
    for config in configs:
        got = sorted((c for (k, _s), c in cells.items() if k == config), key=lambda c: c["seed"])
        if not got:
            continue
        per_seed[config] = {
            "source_run": got[-1]["source_run"],
            "cells": [{"seed": c["seed"], "mean_reward_10eps": c["mean_reward"],
                       "std_reward_10eps": c.get("std_reward"), "bonus": c.get("bonus")}
                      for c in got],
        }
        stats10[config] = ci95([c["mean_reward"] for c in got])
        hundred = {str(c["seed"]): eval100[f"{config}_seed{c['seed']}"]["stoch_unseen"]
                   for c in got
                   if f"{config}_seed{c['seed']}" in eval100
                   and "stoch_unseen" in eval100.get(f"{config}_seed{c['seed']}", {})}
        if hundred:
            stats100[config] = ci95(list(hundred.values()))
            stats100[config]["per_seed"] = hundred
        bonuses = [c["bonus"] for c in got if c.get("bonus")]
        if bonuses:
            scale[config] = {
                "injected_per_step_mean": round(sum(b["effective_bonus"] for b in bonuses)
                                                / len(bonuses), 8),
                "injected_per_step_max": round(max(b["effective_max"] for b in bonuses), 8),
                "raw_intrinsic_mean": round(sum(b["intrinsic_mean"] for b in bonuses)
                                            / len(bonuses), 10),
                "normalized": sorted({bool(b["normalized"]) for b in bonuses}),
                "beta": sorted({b["beta"] for b in bonuses}),
                "n_cells": len(bonuses),
            }

    payload = {
        "_provenance": {
            "produced_by": "summarize_exploration_normalized.py",
            "benchmark": ("scaled-bonus arms: intrinsic error divided by its trailing-window "
                          "mean before beta is applied (models/bonuses.py, normalize=True)"),
            "code_state": "models/bonuses.py after the divisive-rescaling change of 27/09/2026; "
                          "compare_maze_heist.py --normalize; each cell stores the wrapper's "
                          "injected-magnitude bookkeeping",
            "protocol_10eps": "200 train levels, 100k steps, easy, 10 eps stoch on unseen "
                              "levels (num_levels=0, eval seed +1000)",
            "protocol_100eps": "results/eval100_normalized.json (re_eval_100.py over the norm "
                               "checkpoints) when present",
            "control": {
                "source": "results/exploration_remeasure.json, ppo arms of the 24/09/2026 grid",
                "why_reused": "the ppo arm has no intrinsic bonus, so nothing about the "
                              "normalization change touches it: same code, same seeds, same "
                              "protocol. Re-running it would be 10 cells spent to reproduce a "
                              "number that is already frozen.",
                "seeds": sorted({s for (k, s) in cells if k.endswith("_ppo")}),
            },
            "seeds": seeds,
            "cells": sum(len(v["cells"]) for v in per_seed.values()),
        },
        "per_seed": per_seed,
        "statistics_10eps": stats10,
        "statistics_100eps": stats100,
        "injected_scale": scale,
        "vs_unnormalized": {
            c: {"unnormalized_mean_10eps": frozen["statistics_10eps"][c]["mean"],
                "unnormalized_ci95": frozen["statistics_10eps"][c]["ci95"],
                "normalized_mean_10eps": stats10[c]["mean"],
                "normalized_ci95": stats10[c]["ci95"]}
            for c in stats10
            if not c.endswith("_ppo") and c in frozen.get("statistics_10eps", {})
        },
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {os.path.relpath(args.out, BASE).replace(os.sep, '/')}: "
          f"{payload['_provenance']['cells']} cells, seeds {seeds}")
    print(f"{'config':14s} {'n':>2s} {'mean@10':>8s} {'ci95':>18s}  injected/step")
    for c in configs:
        if c not in stats10:
            continue
        s = stats10[c]
        inj = (scale.get(c) or {}).get("injected_per_step_mean")
        print(f"{c:14s} {s['n']:2d} {s['mean']:8.2f}  [{s['ci95'][0]:6.2f},{s['ci95'][1]:6.2f}]"
              + (f"  {inj:.3e}" if inj else ""))
    if not stats100:
        print("100-ep protocol: not run yet (results/eval100_normalized.json absent)")


if __name__ == "__main__":
    main()
