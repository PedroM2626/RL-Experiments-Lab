"""Fold the n_envs=16 grid shards into the artifact README 3.6.7 publishes.

Input: the 15 per-(game, seed) shards (logs_audit/shards_nenv16_grid/), each holding the
16 arms' audit rows at one cell spec; audit_suite_validity.py --merge fuses each game's
five shards into results/suite_validity_nenv16_<game>.json with rows named arm@seed.

This script reads those three merged files and produces
results/suite_validity_nenv16_grid.json: per arm and game the five seed returns (the
archived §3.7 tables published per-seed means, so the re-run has to carry the same shape),
per arm the responsiveness accounting over all 15 cells, the error cells, and the §3.7-style
global mean ranking. Nothing here trains anything; --check re-folds and compares.

Usage:
    py -3.10 summarize_grid_nenv16.py            # writes the folded artifact
    py -3.10 summarize_grid_nenv16.py --check   # fails if the artifact is stale
"""
import argparse
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results")
GAMES = ("starpilot", "bossfight", "dodgeball")
SEEDS = (42, 43, 44, 45, 46)


def load_merged(game):
    with open(os.path.join(RESULTS, f"suite_validity_nenv16_{game}.json"), encoding="utf-8") as f:
        return json.load(f)


def fold():
    per_game, resp, err = {}, {}, {}
    for game in GAMES:
        merged = load_merged(game)
        rows = merged["arms"]
        arms = {k.split("@")[0] for k in rows}
        per_game[game] = {}
        for arm in sorted(arms):
            rs, tvs, am, sat, errors, seconds = [], [], [], [], [], []
            for seed in SEEDS:
                r = rows[f"{arm}@{seed}"]
                if "error" in r:
                    errors.append(seed)
                    continue
                rs.append(r["return_10eps"])
                tvs.append(r["max_pairwise_tv"])
                am.append(r["distinct_argmax_across_games"])
                sat.append(r["first_tanh_saturated_fraction"])
                seconds.append(r["seconds"])
            mean = sum(rs) / len(rs) if rs else None
            per_game[game][arm] = {
                "returns": rs, "mean": round(mean, 3) if rs else None,
                "std": round((sum((x - mean) ** 2 for x in rs) / len(rs)) ** 0.5, 3) if len(rs) > 1 else None,
                "tvs": tvs, "argmax_counts": am, "saturated_firsts": sat,
                "error_seeds": errors, "seconds_total": round(sum(seconds), 1),
            }
            a = resp.setdefault(arm, {"tvs": [], "argmax_over_1": 0, "cells": 0, "error_cells": 0})
            a["tvs"] += tvs
            a["argmax_over_1"] += sum(1 for x in am if x > 1)
            a["cells"] += len(tvs)
            a["error_cells"] += len(errors)
            if errors:
                err.setdefault(game, {})[arm] = errors
    for arm, a in resp.items():
        a["tv_mean"] = round(sum(a["tvs"]) / len(a["tvs"]), 6)
        a["tv_min"] = round(min(a["tvs"]), 6)
        a["cells_tv_over_0p1"] = sum(1 for t in a["tvs"] if t >= 0.1)
        a.pop("tvs")
    global_mean = {arm: round(sum(per_game[g][arm]["mean"] for g in GAMES) / 3, 3)
                   for arm in per_game[GAMES[0]]
                   if all(per_game[g][arm]["mean"] is not None for g in GAMES)}
    hours = sum(per_game[g][a]["seconds_total"] for g in GAMES for a in per_game[g]) / 3600
    return {
        "_provenance": {
            "produced_by": "summarize_grid_nenv16.py",
            "question": "what do the 16 architectures score, and read, under the recipe that keeps policies responsive",
            "recipe": "n_envs=16 x n_steps=256, 100k steps, seeds 42-46, compare_*.train_one via audit_suite_validity.py",
            "starpilot_s42": "adopted from the 3.6.6 pilot (results/suite_validity_starpilot_nenv16.json)",
            "training_hours_total": round(hours, 2),
        },
        "games": per_game,
        "responsiveness": resp,
        "error_cells": err,
        # lists, not tuples: json round-trips tuples into lists and --check compares against
    # what was written, so the artifact must be its own canonical form
    "global_mean_ranking": [[a, m] for a, m in sorted(global_mean.items(), key=lambda kv: -kv[1])],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    payload = fold()
    out = os.path.join(RESULTS, "suite_validity_nenv16_grid.json")
    if args.check:
        with open(out, encoding="utf-8") as f:
            committed = json.load(f)
        if committed != payload:
            print("the grid artifact is stale: rerun summarize_grid_nenv16.py")
            return 1
        print("the grid artifact matches the merged files")
        return 0
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    print("global mean ranking (n_envs=16, 5 seeds, 3 games):")
    for arm, m in payload["global_mean_ranking"]:
        r = payload["responsiveness"][arm]
        print(f"  {arm:15s} {m:5.2f}   TV mean {r['tv_mean']:.3f}  "
              f"cells>=0.1: {r['cells_tv_over_0p1']}/{r['cells']}  argmax>1: {r['argmax_over_1']}")
    print(f"\nerror cells: {payload['error_cells'] or 'none'}")
    print(f"training hours total: {payload['_provenance']['training_hours_total']}")
    print(f"saved {os.path.relpath(out, BASE)}")


if __name__ == "__main__":
    raise SystemExit(main())
