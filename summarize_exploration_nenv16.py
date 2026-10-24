"""Summarize the scaled-bonus exploration arms under the RESPONSIVE recipe (16 envs).

Roadmap item 8 closed as "measured, not interpretable": the scaled-bonus arms of
3.6.1 beat the PPO control, but every policy behind that table was one of the
state-independent marginals of 3.6.2 -- so what the bonus moved could not be
curiosity about the screen. This benchmark re-asks the question at the recipe that
keeps policies responsive (compare_maze_heist.py --normalize --n-envs 16, the
16x256 recipe of 3.6.7): the three bonus arms AND the ppo control are trained
fresh at the same recipe, and every cell carries the response_guard row, so the
table can say not only who scores but who reads.

The control is NOT reused this time: the recipe changed, so the 1-env ppo numbers
are a different arm definition (recorded in _provenance for the reader who wants
both tables side by side).

Usage:
    py -3.10 summarize_exploration_nenv16.py           # folds whatever cells exist
"""
import argparse
import glob
import json
import math
import os

BASE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(BASE, "logs_maze_heist_norm16")
OUT = os.path.join(BASE, "results", "exploration_nenv16.json")
ARMS = ("ppo", "icm", "rnd", "ngu")
GAMES = ("maze", "heist")
T975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 8: 2.365, 10: 2.262}


def ci95(values):
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return {"mean": round(mean, 3), "std": 0.0, "n": n,
                "ci95": [round(mean, 3)] * 2}
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = math.sqrt(var)
    t = T975.get(n, 1.96)
    half = t * std / math.sqrt(n)
    return {"mean": round(mean, 3), "std": round(std, 3), "n": n,
            "ci95": [round(mean - half, 3), round(mean + half, 3)]}


def cells():
    """{(config, seed): cell} across the nenv16 run dirs, later runs winning."""
    out = {}
    pattern = os.path.join(LOGS, "maze_heist_*nenv16*", "comparison_results.json")
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as f:
            j = json.load(f)
        run = os.path.basename(os.path.dirname(path))
        for config, cs in j.items():
            if config.startswith("_"):
                continue
            for c in cs:
                if c.get("mean_reward") is not None:
                    out[(config, c["seed"])] = dict(c, source_run=run)
    return out


def welch_p(a, b):
    """Welch t-test p-value; scipy is present transitively (scikit-learn 1.5.2)."""
    from scipy import stats
    import numpy as np
    _t, p = stats.ttest_ind(np.asarray(a, float), np.asarray(b, float),
                            equal_var=False)
    return round(float(p), 5)


def hedges_g(a, b):
    import numpy as np
    na, nb = len(a), len(b)
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    s = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    d = (np.mean(a) - np.mean(b)) / s if s > 0 else 0.0
    return round(float(d * (1 - 3 / (4 * (na + nb) - 9))), 3)


def response_account(cs):
    """How many of an arm's cells read their observation (the 0.1 scorecard bar)."""
    rows = [c["response"] for c in cs if c.get("response") and "error" not in c["response"]]
    return {
        "cells_measured": len(rows),
        "responsive": sum(1 for r in rows if r.get("responsive")),
        "tv_mean": round(sum(r["max_pairwise_tv"] for r in rows) / len(rows), 6)
        if rows else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    got = cells()
    seeds = sorted({s for (_c, s) in got})
    if not got:
        raise SystemExit(f"no nenv16 cells under {LOGS} yet")

    per_seed, stats_, tests, resp, scale = {}, {}, {}, {}, {}
    for game in GAMES:
        ctrl = [c for (k, s), c in got.items() if k == f"{game}_ppo"]
        ctrl_rets = sorted(c["mean_reward"] for c in ctrl)
        for arm in ARMS:
            config = f"{game}_{arm}"
            cs = sorted((c for (k, _s), c in got.items() if k == config),
                         key=lambda c: c["seed"])
            if not cs:
                continue
            per_seed[config] = {
                "source_runs": sorted({c["source_run"] for c in cs}),
                "cells": [{"seed": c["seed"], "mean_reward_100eps": c["mean_reward"],
                           "std_reward": c.get("std_reward"), "seconds": c.get("seconds"),
                           "bonus": c.get("bonus"), "response": c.get("response")}
                          for c in cs],
            }
            stats_[config] = ci95([c["mean_reward"] for c in cs])
            resp[config] = response_account(cs)
            bonuses = [c["bonus"] for c in cs if c.get("bonus")]
            if bonuses:
                scale[config] = {
                    "injected_per_step_mean": round(
                        sum(b["effective_bonus"] for b in bonuses) / len(bonuses), 8),
                    "injected_per_step_max": round(
                        max(b["effective_max"] for b in bonuses), 8),
                    "n_envs_per_wrapper": sorted({b.get("n_envs", 1) for b in bonuses}),
                }
            if arm != "ppo" and len(ctrl) == len(cs) and ctrl:
                rets = [c["mean_reward"] for c in cs]
                tests[config] = {
                    "vs": f"{game}_ppo (same recipe, fresh)",
                    "delta_mean": round(sum(rets) / len(rets)
                                         - sum(ctrl_rets) / len(ctrl_rets), 3),
                    "welch_p": welch_p(rets, ctrl_rets),
                    "hedges_g": hedges_g(rets, ctrl_rets),
                }

    total = sum(len(v["cells"]) for v in per_seed.values())
    payload = {
        "_provenance": {
            "produced_by": "summarize_exploration_nenv16.py",
            "benchmark": "scaled-bonus arms AND fresh ppo control at the responsive "
                         "recipe: compare_maze_heist.py --normalize --n-envs 16 "
                         "(16 parallel envs, per-rank level streams, n_steps=256, "
                         "per-update batch 4096 -- README 3.6.7's grid recipe)",
            "question": "does the scaled curiosity bonus separate from PPO among "
                        "policies that actually read their observation",
            "protocol": "200 train levels, 100k steps, easy, 100 eps stoch on "
                        "unseen levels (num_levels=0, eval seed +1000)",
            "control": "trained fresh: the recipe changed, so the 1-env ppo rows of "
                       "3.6.1/exploration_remeasure.json are a different arm and are "
                       "NOT reused here",
            "response_measure": "response_guard.py over diagnose_policy_saturation."
                                "cross_game_response (five games' initial frames)",
            "seeds_present": seeds,
            "cells": total,
        },
        "per_seed": per_seed,
        "statistics_100eps": stats_,
        "vs_control": tests,
        "responsiveness": resp,
        "injected_scale": scale,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote results/exploration_nenv16.json: {total} cells, seeds {seeds}")
    print(f"{'config':12s} {'n':>2s} {'mean':>7s} {'ci95':>16s}  {'resp':>6s}  tv_mean")
    for game in GAMES:
        for arm in ARMS:
            c = f"{game}_{arm}"
            if c not in stats_:
                continue
            s, r = stats_[c], resp[c]
            print(f"{c:12s} {s['n']:2d} {s['mean']:7.2f}  "
                  f"[{s['ci95'][0]:5.2f},{s['ci95'][1]:5.2f}]  "
                  f"{r['responsive']:2d}/{r['cells_measured']:<2d}  {r['tv_mean']}")
        for arm in ("icm", "rnd", "ngu"):
            t = tests.get(f"{game}_{arm}")
            if t:
                print(f"  {game}_{arm} vs ppo: d={t['delta_mean']:+.2f}  "
                      f"p={t['welch_p']}  g={t['hedges_g']}")


if __name__ == "__main__":
    main()
