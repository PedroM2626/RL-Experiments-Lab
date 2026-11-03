"""Fold the port-grid response audit (01/11/2026) into its committed artifacts.

jax_port/audit_port_response.sh re-trains the port's `main` suite -- 16 configs x
bossfight/starpilot/dodgeball x seeds 42-46, 100k steps, the grid's own recipe (64 envs x
128 steps, minibatch 1024) and --eval-full protocol -- with the cross-game response measure
(jax_port/response.py, the port of README 3.6.2's instrument) attached to every cell.

Two artifacts, so the fold is checkable without the raw run directory (which lives in the
WSL filesystem and is not committed):
  jax_port/port_response_cells.json  one row per cell, extracted from the raw JSONs (--raw)
  jax_port/port_response_grid.json   the fold: per (game, cfg) returns and TVs, the
                                     responsiveness accounting per config, and the
                                     comparison with the archived grid (cells_summary.json)

Usage:
    python jax_port/summarize_port_response.py --raw /root/rg_response/main   # extract + fold
    python jax_port/summarize_port_response.py                                # re-fold
    python jax_port/summarize_port_response.py --check                        # fail if stale
"""
import argparse
import glob
import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CELLS = os.path.join(BASE, "jax_port", "port_response_cells.json")
OUT = os.path.join(BASE, "jax_port", "port_response_grid.json")
ARCHIVED = os.path.join(BASE, "jax_port", "cells_summary.json")
GAMES = ("bossfight", "starpilot", "dodgeball")
SEEDS = (42, 43, 44, 45, 46)
RESPONSIVE_TV = 0.1


def extract(raw_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "*__seed*__100k.json"))):
        cfg = os.path.basename(path).split("__")[0]
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        r = d["cross_game_response"]
        rows.append({"cfg": cfg, "game": d["game"], "seed": d["seed"], "timesteps": d["timesteps"],
                     "eval_unseen": d["eval_unseen"]["mean"], "eval_eps": d["eval_unseen"]["eps"],
                     "max_pairwise_tv": r["max_pairwise_tv"],
                     "distinct_argmax_across_games": r["distinct_argmax_across_games"],
                     "v_std_across_games": r["v_std_across_games"], "sps": d["sps"]})
    return rows


def _mean(xs):
    return round(sum(xs) / len(xs), 6) if xs else None


def fold(rows):
    with open(ARCHIVED, encoding="utf-8") as f:
        archived = {(c["cfg"], c["game"], c["seed"]): c["eval_unseen"]
                    for c in json.load(f)["main"]}
    cfgs = sorted({r["cfg"] for r in rows})
    by = {(r["cfg"], r["game"], r["seed"]): r for r in rows}
    games, resp = {}, {}
    for game in GAMES:
        games[game] = {}
        for cfg in cfgs:
            cells = [by[(cfg, game, s)] for s in SEEDS if (cfg, game, s) in by]
            rets = [c["eval_unseen"] for c in cells]
            old = [archived[(cfg, game, c["seed"])] for c in cells]
            games[game][cfg] = {
                "seeds": [c["seed"] for c in cells], "returns": rets, "mean": _mean(rets),
                "tvs": [c["max_pairwise_tv"] for c in cells],
                "argmax_counts": [c["distinct_argmax_across_games"] for c in cells],
                "archived_returns": old, "archived_mean": _mean(old)}
    for cfg in cfgs:
        tvs = [t for g in GAMES for t in games[g][cfg]["tvs"]]
        resp[cfg] = {"cells": len(tvs), "cells_tv_over_0p1": sum(1 for t in tvs if t >= RESPONSIVE_TV),
                     "tv_mean": _mean(tvs)}
    ranking = sorted(cfgs, key=lambda c: (-resp[c]["cells_tv_over_0p1"], -resp[c]["tv_mean"]))
    all_rets = [(r["eval_unseen"], archived[(r["cfg"], r["game"], r["seed"])]) for r in rows]
    n_cells = len(rows)
    return {
        "_provenance": {
            "produced_by": "jax_port/summarize_port_response.py",
            "question": "does the port's grid -- trained at 64 envs x 128 steps, a recipe far above "
                        "the SB3 coverage cliff of README 3.6.5 -- read its observation, and does "
                        "its response separate the front ends the way the SB3 scorecard (0.1) does",
            "recipe": "jax_port/audit_port_response.sh: run_grade --suite main --eval-full --response, "
                      "64 envs x 128 steps, minibatch 1024, 100k steps (106496 logged), seeds 42-46",
            "response_measure": "jax_port/response.py (max pairwise TV over the initial frames of "
                                "maze/heist/bossfight/starpilot/dodgeball; responsive at TV >= 0.1)",
            "reproduction_note": "the cells are re-trained, not the archived weights (the archived grid "
                                 "saved none); GPU training is not bitwise deterministic, so returns "
                                 "are compared with the archived grid as distributions, not cell by cell",
            "cells": n_cells,
        },
        "games": games,
        "responsiveness": resp,
        "response_ranking": ranking,
        "vs_archived": {
            "mean_return_rerun": _mean([a for a, _ in all_rets]),
            "mean_return_archived": _mean([b for _, b in all_rets]),
            "mean_abs_cell_delta": _mean([abs(a - b) for a, b in all_rets]),
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", default=None, help="raw run directory to extract the cell rows from")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.raw:
        rows = extract(args.raw)
        with open(CELLS, "w", encoding="utf-8") as f:
            json.dump({"_provenance": {"extracted_from": "jax_port/audit_port_response.sh run directory"},
                       "cells": rows}, f, indent=1)
    with open(CELLS, encoding="utf-8") as f:
        rows = json.load(f)["cells"]
    out = fold(rows)
    if args.check:
        with open(OUT, encoding="utf-8") as f:
            assert json.load(f) == out, "stale: rerun jax_port/summarize_port_response.py"
        print("OK: port_response_grid.json is current")
        return
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"{out['_provenance']['cells']} cells -> {os.path.relpath(OUT, BASE)}")
    for cfg in out["response_ranking"]:
        r = out["responsiveness"][cfg]
        print(f"  {cfg:15s} {r['cells_tv_over_0p1']:2d}/{r['cells']}  TV {r['tv_mean']}")


if __name__ == "__main__":
    main()
