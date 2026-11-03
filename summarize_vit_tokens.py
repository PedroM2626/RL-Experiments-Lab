"""Fold the ViT token-pipeline experiment (01/11/2026) into results/vit_token_pipeline.json.

README 3.6.10-3.6.11 left vit/bossfight as the one cell that never ignites: its front end
maps different screens to nearly the same vector, and neither single-stage fix tried
(LayerNorm at the output, max over tokens) changed that. This experiment changes the whole
token pipeline at once -- diagnose_policy_saturation.ViTTokensExtractor: 8x8 patches,
pre-LN without dropout, a CLS readout through a final LayerNorm -- then ablates it one stage
at a time, and finally runs it as a full audit arm (16x256, 3 games x seeds 42-46).

Inputs (all committed): the dense probes results/probe_{arch}_bossfight_seed{s}.json for
vit_tokens (42-44) and its three ablations (42-43), the earlier ViT probes of 3.6.9/3.6.11
for comparison, and results/suite_validity_nenv16_vit_tokens_<game>.json from
run_vit_tokens_grid.sh. --check re-folds and compares.
"""
import argparse
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results")
OUT = os.path.join(RESULTS, "vit_token_pipeline.json")
ALIVE = 5e-5  # the dense probe's alive threshold (README 3.6.4)
GAMES = ("starpilot", "bossfight", "dodgeball")
SEEDS = (42, 43, 44, 45, 46)
PROBES = {
    "vit": {42: "probe_nonreaders_vit_bossfight.json"},
    "vit_ln": {42: "probe_vitlayernorm_bossfight.json", 43: "probe_vit_ln_bossfight_seed43.json",
               44: "probe_vit_ln_bossfight_seed44.json"},
    "vit_maxpool": {42: "probe_vitmaxpool_bossfight.json", 43: "probe_vit_maxpool_bossfight_seed43.json",
                    44: "probe_vit_maxpool_bossfight_seed44.json"},
    "vit_tokens": {s: f"probe_vit_tokens_bossfight_seed{s}.json" for s in (42, 43, 44)},
    "vit_patch8": {s: f"probe_vit_patch8_bossfight_seed{s}.json" for s in (42, 43)},
    "vit_prenorm": {s: f"probe_vit_prenorm_bossfight_seed{s}.json" for s in (42, 43)},
    "vit_cls": {s: f"probe_vit_cls_bossfight_seed{s}.json" for s in (42, 43)},
}
REFERENCE = "probe_nonreaders_classic_bossfight.json"


def _load(name):
    with open(os.path.join(RESULTS, name), encoding="utf-8") as f:
        return json.load(f)


def probe_row(name):
    d = _load(name)
    tv = [r["cross_frame_tv_mean"] for r in d["curve"]]
    return {"within_game_max": max(tv), "within_game_final": tv[-1],
            "alive_probes": sum(1 for t in tv if t >= ALIVE), "probes": len(tv),
            "survives": tv[-1] >= ALIVE,
            "cross_game_tv": d["cross_game"]["max_pairwise_tv"],
            "distinct_argmax_across_games": d["cross_game"]["distinct_argmax_across_games"]}


def fold():
    probes = {arch: {str(s): probe_row(f) for s, f in files.items()} for arch, files in PROBES.items()}
    summary = {arch: {"runs": len(rows), "survive": sum(r["survives"] for r in rows.values()),
                      "cross_game_readers": sum(r["cross_game_tv"] >= 0.1 for r in rows.values())}
               for arch, rows in probes.items()}
    grid, tvs = {}, []
    for game in GAMES:
        rows = _load(f"suite_validity_nenv16_vit_tokens_{game}.json")["arms"]
        cells = [rows[f"vit_tokens@{s}"] for s in SEEDS]
        ok = [r for r in cells if "error" not in r]
        grid[game] = {"returns": [r["return_10eps"] for r in ok], "tvs": [r["max_pairwise_tv"] for r in ok],
                      "error_seeds": [s for s, r in zip(SEEDS, cells) if "error" in r]}
        tvs += grid[game]["tvs"]
    published = _load("suite_validity_nenv16_grid.json")["responsiveness"]["vit"]
    return {
        "_provenance": {
            "produced_by": "summarize_vit_tokens.py",
            "question": "is vit/bossfight's never-ignite a token-pipeline property, and which stage",
            "arms": "vit_tokens = 8x8 patches + pre-LN/dropout 0 + CLS readout with final LayerNorm; "
                    "vit_patch8 / vit_prenorm / vit_cls change one stage each "
                    "(diagnose_policy_saturation.VIT_ABLATIONS)",
            "probe": "--game bossfight --timesteps 100000 --probe-every 500 --n-envs 16 --n-steps 256 --cross-game",
            "alive_threshold": ALIVE,
        },
        "reference_classic_bossfight": probe_row(REFERENCE),
        "probes": probes,
        "probe_summary": summary,
        "grid": grid,
        "grid_responsiveness": {"cells": len(tvs), "cells_tv_over_0p1": sum(1 for t in tvs if t >= 0.1),
                                "tv_mean": round(sum(tvs) / len(tvs), 6) if tvs else None},
        "published_vit_responsiveness": published,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    out = fold()
    if args.check:
        assert _load("vit_token_pipeline.json") == out, "stale: rerun summarize_vit_tokens.py"
        print("OK: results/vit_token_pipeline.json is current")
        return
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    for arch, s in out["probe_summary"].items():
        print(f"{arch:12s} survive {s['survive']}/{s['runs']}  cross-game readers {s['cross_game_readers']}")
    g = out["grid_responsiveness"]
    print(f"grid: {g['cells_tv_over_0p1']}/{g['cells']} responsive, TV {g['tv_mean']}")


if __name__ == "__main__":
    main()
