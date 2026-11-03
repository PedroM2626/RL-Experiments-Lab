"""Fold the vae_clamped audit arm (01/11/2026) into the artifact the diagnostic arc cites.

The published `vae` arm diverges to NaN on bossfight/seed 42 under both recipes (README
3.6.2, 3.6.7). diagnose_vae_nan.py names the mechanism -- an unbounded logvar head with no
KL term runs away until exp(0.5 * logvar) overflows float32 -- and the fix is the opt-in
`vae_clamped` arm: the same extractor with logvar bounded to [-10, 10]
(VAEExtractor(logvar_clamp=10.0), audit_suite_validity.py). run_vae_clamped_grid.sh trains
it at the scorecard's own protocol (16x256, 100k, seeds 42-46, 3 games) and merges each
game's five shards into results/suite_validity_nenv16_vae_clamped_<game>.json.

This script folds those three files, plus the two diagnose_vae_nan.py runs of the
diverging cell, into results/vae_nan_fix.json, next to the published vae row of the 3.6.7
grid. --check re-folds and compares.
"""
import argparse
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results")
OUT = os.path.join(RESULTS, "vae_nan_fix.json")
GAMES = ("starpilot", "bossfight", "dodgeball")
SEEDS = (42, 43, 44, 45, 46)


def _load(name):
    with open(os.path.join(RESULTS, name), encoding="utf-8") as f:
        return json.load(f)


def fold():
    grid = _load("suite_validity_nenv16_grid.json")
    per_game, tvs, errors = {}, [], []
    for game in GAMES:
        rows = _load(f"suite_validity_nenv16_vae_clamped_{game}.json")["arms"]
        cells = [rows[f"vae_clamped@{s}"] for s in SEEDS]
        bad = [s for s, r in zip(SEEDS, cells) if "error" in r]
        ok = [r for r in cells if "error" not in r]
        errors += [f"{game}@{s}" for s in bad]
        rets = [r["return_10eps"] for r in ok]
        game_tvs = [r["max_pairwise_tv"] for r in ok]
        tvs += game_tvs
        pub = grid["games"][game]["vae"]
        per_game[game] = {
            "returns": rets, "mean": round(sum(rets) / len(rets), 3) if rets else None,
            "tvs": game_tvs, "feat_max": [r["feat_max"] for r in ok],
            "error_seeds": bad,
            "published_vae": {"returns": pub["returns"], "mean": pub["mean"], "tvs": pub["tvs"],
                              "error_seeds": pub["error_seeds"]}}
    pub_resp = grid["responsiveness"]["vae"]
    diag = {}
    for tag, name in (("published", "vae_nan_bossfight_s42.json"), ("clamped", "vae_clamped_bossfight_s42.json")):
        d = _load(name)
        lv = [r["logvar_max"] for r in d["curve"] if isinstance(r["logvar_max"], (int, float))]
        diag[tag] = {"steps_reached": d["steps_reached"], "error": d["error"],
                     "logvar_max_overall": max(lv) if lv else None,
                     "nonfinite_rows": sum(1 for r in d["curve"] if r["nonfinite"])}
    return {
        "_provenance": {
            "produced_by": "summarize_vae_clamped.py",
            "question": "what makes the vae arm diverge to NaN, and does bounding logvar fix it",
            "recipe": "16 envs x 256 steps, 100k, seeds 42-46, audit_suite_validity.py --configs vae_clamped "
                      "(run_vae_clamped_grid.sh)",
            "fix": "VAEExtractor(logvar_clamp=10.0): logvar bounded to [-10, 10], so the sampled std is at "
                   "most e^5; the published arm (logvar_clamp=None) is unchanged",
        },
        "diagnosis_bossfight_seed42": diag,
        "games": per_game,
        "responsiveness": {"cells": len(tvs), "cells_tv_over_0p1": sum(1 for t in tvs if t >= 0.1),
                           "tv_mean": round(sum(tvs) / len(tvs), 6) if tvs else None,
                           "error_cells": errors},
        "published_vae_responsiveness": pub_resp,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    out = fold()
    if args.check:
        assert _load("vae_nan_fix.json") == out, "stale: rerun summarize_vae_clamped.py"
        print("OK: results/vae_nan_fix.json is current")
        return
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    r = out["responsiveness"]
    print(f"vae_clamped: {r['cells_tv_over_0p1']}/{r['cells']} responsive, TV {r['tv_mean']}, "
          f"errors {r['error_cells']}")


if __name__ == "__main__":
    main()
