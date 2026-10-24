"""Fold the mlp_scaled grid shards into the artifact README 3.6.12 cites.

Input: the 15 per-(game, seed) shards (logs_audit/shards_mlpscaled/), each holding the
mlp_scaled audit row at one cell spec; audit_suite_validity.py --merge fuses each game's
five shards into results/suite_validity_nenv16_mlp_scaled_<game>.json with rows named
arm@seed. mlp_scaled is the 3.6.12 input-scale lever run as a full grid arm: the same
256-D vector observation as mlp_vector, scaled to [0,1] float32 by the wrapper
(procgen_wrapper.ProcgenVectorWrapper(scale=True)) -- the raw-uint8 arm stays in the
published grid artifact untouched, so the two are comparable cell by cell.

This script reads those three merged files and produces
results/suite_validity_nenv16_mlp_scaled.json in the same shape as the 3.6.7 grid's
summarize_grid_nenv16.py fold: per game the five seed returns, the responsiveness
accounting over all 15 cells, and the comparison against the raw mlp_vector row of the
committed grid artifact. Nothing here trains anything; --check re-folds and compares.

Usage:
    py -3.10 summarize_mlpscaled_grid.py            # writes the folded artifact
    py -3.10 summarize_mlpscaled_grid.py --check   # fails if the artifact is stale
"""
import argparse
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results")
GAMES = ("starpilot", "bossfight", "dodgeball")
SEEDS = (42, 43, 44, 45, 46)


def load_merged(game):
    with open(os.path.join(RESULTS, f"suite_validity_nenv16_mlp_scaled_{game}.json"),
              encoding="utf-8") as f:
        return json.load(f)


def load_raw_mlp_vector():
    """The published grid's raw-uint8 arm, for the cell-by-cell comparison."""
    with open(os.path.join(RESULTS, "suite_validity_nenv16_grid.json"), encoding="utf-8") as f:
        grid = json.load(f)
    return {g: grid["games"][g]["mlp_vector"] for g in GAMES},            grid["responsiveness"]["mlp_vector"]


def fold():
    per_game, resp = {}, {"tvs": [], "argmax_over_1": 0, "cells": 0, "error_cells": 0}
    err = {}
    for game in GAMES:
        rows = load_merged(game)["arms"]
        rs, tvs, am, sat, errors, seconds = [], [], [], [], [], []
        for seed in SEEDS:
            r = rows[f"mlp_scaled@{seed}"]
            if "error" in r:
                errors.append(seed)
                continue
            rs.append(r["return_10eps"])
            tvs.append(r["max_pairwise_tv"])
            am.append(r["distinct_argmax_across_games"])
            sat.append(r["first_tanh_saturated_fraction"])
            seconds.append(r["seconds"])
        mean = sum(rs) / len(rs) if rs else None
        per_game[game] = {
            "returns": rs, "mean": round(mean, 3) if rs else None,
            "std": round((sum((x - mean) ** 2 for x in rs) / len(rs)) ** 0.5, 3) if len(rs) > 1 else None,
            "tvs": tvs, "argmax_counts": am, "saturated_firsts": sat,
            "error_seeds": errors, "seconds_total": round(sum(seconds), 1),
        }
        resp["tvs"] += tvs
        resp["argmax_over_1"] += sum(1 for x in am if x > 1)
        resp["cells"] += len(tvs)
        resp["error_cells"] += len(errors)
        if errors:
            err[game] = errors
    resp["tv_mean"] = round(sum(resp["tvs"]) / len(resp["tvs"]), 6)
    resp["tv_min"] = round(min(resp["tvs"]), 6)
    resp["cells_tv_over_0p1"] = sum(1 for t in resp["tvs"] if t >= 0.1)
    resp.pop("tvs")
    raw_games, raw_resp = load_raw_mlp_vector()
    hours = sum(per_game[g]["seconds_total"] for g in GAMES) / 3600
    return {
        "_provenance": {
            "produced_by": "summarize_mlpscaled_grid.py",
            "question": "does the 3.6.12 input-scale rescue hold as a full audit arm "
                        "(the 16x256 recipe, 5 seeds, the cross-game response measure)",
            "recipe": "n_envs=16 x n_steps=256, 100k steps, seeds 42-46, "
                      "compare_suite.train_one(vector=True, vector_scale=True) via audit_suite_validity.py",
            "arm": "mlp_scaled: the same 256-D vector observation as the published "
                   "mlp_vector, scaled to [0,1] float32 (ProcgenVectorWrapper(scale=True)); "
                   "the raw-uint8 arm is opt-out legacy, see procgen_wrapper.py",
            "training_hours_total": round(hours, 2),
        },
        "games": per_game,
        "responsiveness": resp,
        "error_cells": err,
        "vs_raw_mlp_vector": {
            "raw_responsive_cells": raw_resp["cells_tv_over_0p1"],
            "scaled_responsive_cells": resp["cells_tv_over_0p1"],
            "raw_tv_mean": raw_resp["tv_mean"],
            "scaled_tv_mean": resp["tv_mean"],
            "raw_returns": {g: raw_games[g]["mean"] for g in GAMES},
            "scaled_returns": {g: per_game[g]["mean"] for g in GAMES},
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    payload = fold()
    out = os.path.join(RESULTS, "suite_validity_nenv16_mlp_scaled.json")
    if args.check:
        with open(out, encoding="utf-8") as f:
            committed = json.load(f)
        if committed != payload:
            print("the mlp_scaled artifact is stale: rerun summarize_mlpscaled_grid.py")
            return 1
        print("the mlp_scaled artifact matches the merged files")
        return 0
    with open(out, "w") as f:
        json.dump(payload, f, indent=1)
    cmp_ = payload["vs_raw_mlp_vector"]
    print("mlp_scaled (the 3.6.12 rescue as an audit arm) vs the raw mlp_vector row:")
    print(f"  responsive cells: {cmp_['scaled_responsive_cells']}/15 vs {cmp_['raw_responsive_cells']}/15")
    print(f"  cross-game TV mean: {cmp_['scaled_tv_mean']} vs {cmp_['raw_tv_mean']}")
    for g in GAMES:
        pg = payload["games"][g]
        print(f"  {g:10s} returns {pg['mean']} (raw {cmp_['raw_returns'][g]})  "
              f"responsive cells {sum(1 for t in pg['tvs'] if t >= 0.1)}/5")
    print(f"\nerror cells: {payload['error_cells'] or 'none'}")
    print(f"training hours total: {payload['_provenance']['training_hours_total']}")
    print(f"saved {os.path.relpath(out, BASE)}")


if __name__ == "__main__":
    raise SystemExit(main())
