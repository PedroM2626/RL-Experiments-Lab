"""Fold the section 3.6 / 3.12 exploration re-measurement into the committed evidence.

Roadmap item 7: the rnd/ngu arms of the maze/heist benchmark were run through wrappers whose
network geometry was wrong, so every step raised and the error was swallowed — the published
'ppo == rnd == ngu' ties were one PPO run trained three times, not three null results. The
grid was re-run with the corrected models/bonuses.py, which produces new numbers for all four
arms. This script turns those run directories into committed data, because the previous
version of this evidence lived only in git-ignored logs_* directories and was lost.

Reads   logs_maze_heist/maze_heist_*/comparison_results.json   (10 eps, §3.6 protocol)
        results/eval100_remeasure.json                         (100 eps, §3.12 protocol)
        results/scorecard.json, results/eval100_results.json   (the values being superseded)
        results/random_baselines.json                          (the §18.1 anchors)
Writes  results/exploration_remeasure.json
        results/eval100_results.json                            only with --apply

Usage:
    py -3.10 merge_exploration_remeasure.py                 # report only
    py -3.10 merge_exploration_remeasure.py --apply         # also update eval100_results.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(BASE, "logs_maze_heist")
RESULTS = os.path.join(BASE, "results")
T_CRIT_95_DF4 = 2.776
CONFIGS = [f"{g}_{a}" for g in ("maze", "heist") for a in ("ppo", "icm", "rnd", "ngu")]
BONUS_ARMS = ("icm", "rnd", "ngu")


def load_runs(logs_dir=LOGS, require_complete=True):
    """{config: {seed: cell}} merged from every run directory, newest file winning."""
    cells, sources, conflicts = {}, {}, []
    for path in sorted(glob.glob(os.path.join(logs_dir, "maze_heist_*", "comparison_results.json")),
                       key=os.path.getmtime):
        with open(path, encoding="utf-8") as f:
            j = json.load(f)
        for k, v in j.items():
            if k.startswith("_"):
                continue
            for c in v:
                if c.get("mean_reward") is None:
                    conflicts.append(f"{k} seed {c['seed']}: failed run ({c.get('error')})")
                    continue
                prev = cells.get((k, c["seed"]))
                if prev is not None and abs(prev["mean_reward"] - c["mean_reward"]) > 1e-9:
                    conflicts.append(f"{k} seed {c['seed']}: measured twice, "
                                     f"keeping {c['mean_reward']:.3f} over {prev['mean_reward']:.3f}")
                cells[(k, c["seed"])] = dict(c, _from=os.path.basename(os.path.dirname(path)))
                sources[k] = os.path.basename(os.path.dirname(path))
    out = {}
    for (k, seed), c in cells.items():
        out.setdefault(k, {})[seed] = c
    missing = [f"{k} seed {s}" for k in CONFIGS for s in (42, 43, 44, 45, 46)
               if s not in out.get(k, {})]
    if missing and require_complete:
        print("ERROR: the grid is incomplete; missing or failed cells:")
        for m in missing[:20]:
            print(f"  {m}")
        if len(missing) > 20:
            print(f"  ... {len(missing) - 20} more")
        for c in conflicts:
            print(f"  {c}")
        print("Re-run the sweep with --resume, or pass --allow-partial to inspect "
              "incomplete data (it must not be published).")
        sys.exit(1)
    return out, sources, conflicts


def verify_bonus_fired(cells):
    """An arm whose wrapper never added a bonus is an unlabelled PPO run, not a null result."""
    bad = []
    for k, per_seed in cells.items():
        arm = k.split("_", 1)[1]
        if arm not in BONUS_ARMS:
            continue
        for seed, c in per_seed.items():
            b = c.get("bonus") or {}
            if not b.get("bonus_applied"):
                bad.append(f"{k} seed {seed}: no intrinsic bonus recorded")
            elif b.get("bonus_applied") / max(1, b.get("steps", 1)) < 0.9:
                bad.append(f"{k} seed {seed}: bonus applied on only "
                           f"{b['bonus_applied']}/{b['steps']} steps")
    if bad:
        print("ERROR: exploration arms that did not explore:")
        for b in bad:
            print(f"  {b}")
        sys.exit(1)


def stats_of(vals):
    v = np.array(vals, float)
    m, s = float(v.mean()), float(v.std(ddof=1))
    half = T_CRIT_95_DF4 * s / np.sqrt(len(v))
    return {"mean": round(m, 3), "std": round(s, 3), "n": len(v),
            "ci95": [round(m - half, 3), round(m + half, 3)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true",
                        help="write the new 100-eps values into results/eval100_results.json")
    parser.add_argument("--eval100", default=os.path.join(RESULTS, "eval100_remeasure.json"))
    parser.add_argument("--out", default=os.path.join(RESULTS, "exploration_remeasure.json"))
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    cells, sources, conflicts = load_runs(require_complete=not args.allow_partial)
    verify_bonus_fired(cells)

    with open(os.path.join(RESULTS, "scorecard.json"), encoding="utf-8") as f:
        old_scorecard = json.load(f).get("ci_effect_size", {})
    with open(os.path.join(RESULTS, "eval100_results.json"), encoding="utf-8") as f:
        eval100 = json.load(f)
    with open(os.path.join(RESULTS, "random_baselines.json"), encoding="utf-8") as f:
        anchors = json.load(f)
    new_eval100 = {}
    if os.path.exists(args.eval100):
        with open(args.eval100, encoding="utf-8") as f:
            new_eval100 = json.load(f)
    else:
        print(f"NOTE: no 100-eps re-measurement at {args.eval100}; the 10-eps columns still stand")

    per_seed, summary, superseded = {}, {}, {}
    for k in CONFIGS:
        got = cells.get(k, {})
        seeds = sorted(got)
        per_seed[k] = {"source_run": sources.get(k),
                       "cells": [{"seed": s, "mean_reward_10eps": got[s]["mean_reward"],
                                  "std_reward_10eps": got[s].get("std_reward"),
                                  "seconds": got[s].get("seconds"),
                                  "bonus": got[s].get("bonus")} for s in seeds]}
        if len(seeds) >= 2:
            summary[k] = stats_of([got[s]["mean_reward"] for s in seeds])
        superseded.setdefault("scorecard_10eps", {})[k] = old_scorecard.get(k)
    for k in [x for x in eval100 if x.split("_")[0] in ("maze", "heist")]:
        superseded.setdefault("eval100_per_model", {})[k] = eval100[k]

    # 100-eps aggregation per config, and the comparison against the measured random anchor
    eval100_by_config = {}
    for k in CONFIGS:
        vals = {int(m.split("_seed")[1]): new_eval100[m]["stoch_unseen"]
                for m in new_eval100 if m.startswith(k + "_seed") and "stoch_unseen" in new_eval100[m]}
        if vals:
            agg = stats_of(list(vals.values()))
            agg["per_seed"] = {str(s): vals[s] for s in sorted(vals)}
            agg["det_unseen_mean"] = round(float(np.mean(
                [new_eval100[f"{k}_seed{s}"]["det_unseen"] for s in vals])), 3)
            eval100_by_config[k] = agg

    random_anchors = {g: {"mean": anchors[g]["mean"], "std": anchors[g]["std"],
                          "n_episodes": anchors[g]["n_episodes"]}
                      for g in ("maze", "heist") if g in anchors}
    verdicts = {}
    for k, s in summary.items():
        game = k.split("_")[0]
        anchor = random_anchors.get(game, {}).get("mean")
        if anchor is None:
            continue
        verdicts[k] = {"vs_random": round(s["mean"] - anchor, 3),
                       "below_random": bool(s["ci95"][1] < anchor),
                       "above_random": bool(s["ci95"][0] > anchor)}

    payload = {
        "_provenance": {
            "produced_by": "merge_exploration_remeasure.py",
            "replaces": "the maze/heist arms of README sections 3.6 and 3.12, measured with "
                        "RND/NGU wrappers that raised on every step (see section 3.6 note)",
            "code_state": "models/bonuses.py after the 23/09/2026 geometry fix; each cell "
                          "stores the wrapper's own bonus bookkeeping",
            "protocol_10eps": "200 train levels, 5 seeds 42-46, 10 eps stoch on unseen levels "
                              "(num_levels=0, eval seed +1000), 100k steps, easy",
            "protocol_100eps": "results/eval100_remeasure.json: 100 eps stoch + 100 det on "
                               "unseen levels, 15 eps on train levels (re_eval_100.py)",
            "conflicts": conflicts,
        },
        "per_seed": per_seed,
        "statistics_10eps": summary,
        "statistics_100eps": eval100_by_config,
        "random_anchors_unseen": random_anchors,
        "vs_random": verdicts,
        "superseded": superseded,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"{'config':14s} {'10eps mean+-std':>18s} {'95% CI':>18s} "
          f"{'100eps mean':>12s} {'vs random':>10s}")
    for k in CONFIGS:
        s = summary.get(k)
        e = eval100_by_config.get(k)
        v = verdicts.get(k)
        ms = f"{s['mean']:.2f} ± {s['std']:.2f}" if s else "-"
        ci = f"[{s['ci95'][0]:+.2f}, {s['ci95'][1]:+.2f}]" if s else "-"
        e100 = f"{e['mean']:.2f}" if e else "-"
        if v:
            tag = "below" if v["below_random"] else ("above" if v["above_random"] else "n.s.")
            vsr = f"{v['vs_random']:+.2f} {tag}"
        else:
            vsr = "-"
        print(f"{k:14s} {ms:>18s} {ci:>18s} {e100:>12s} {vsr:>10s}")
    for game, a in random_anchors.items():
        print(f"random anchor {game}: {a['mean']:.2f} ± {a['std']:.2f} ({a['n_episodes']} eps)")
    print(f"Wrote {args.out}")

    if args.apply:
        if not eval100_by_config:
            print("ERROR: --apply requested but no 100-eps re-measurement was found "
                  f"at {args.eval100}")
            sys.exit(1)
        updated = 0
        for k in CONFIGS:
            for seed, val in eval100_by_config[k]["per_seed"].items():
                key = f"{k}_seed{seed}"
                src = new_eval100[key]
                eval100[key] = {"stoch_unseen": src["stoch_unseen"], "det_unseen": src["det_unseen"],
                                "stoch_train": src["stoch_train"], "gen_gap": src["gen_gap"],
                                "n_unseen": src["n_unseen"], "n_train": src["n_train"],
                                "remeasured": "23/09/2026, see results/exploration_remeasure.json"}
                updated += 1
        with open(os.path.join(RESULTS, "eval100_results.json"), "w", encoding="utf-8") as f:
            json.dump(eval100, f, indent=2)
        print(f"Updated {updated} models in results/eval100_results.json "
              f"(previous values archived under 'superseded' in {os.path.basename(args.out)})")


if __name__ == "__main__":
    main()
