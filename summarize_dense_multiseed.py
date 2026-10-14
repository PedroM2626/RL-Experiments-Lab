"""Fold the dense-probe multi-seed and mechanism-lever runs into one artifact.

README 3.6.4 published a dose-response measured once per arm (seed 42). 3.6.5 asks the
two questions that single-seed curves cannot answer: does the response replicate across
seeds, and which of the candidate mechanisms survives when each is pulled as a lever.

This script reads every dense-probe artifact, reduces each run to the row the section
publishes (first dead probe, alive count, peak, endpoint, cross-game response), groups
the runs by arm, and writes results/policy_saturation_dense_multiseed.json. The README
gates on that file; nothing here re-runs training.

The (iv) dose (14/10/2026) adds 2x128 and 8x32 -- two more points on the
levels-per-update axis at the fixed published batch of 256 transitions -- so the merged
artifact carries seven five-seed arms plus the six lever runs.

Usage:
    py -3.10 summarize_dense_multiseed.py            # writes the merged artifact
    py -3.10 summarize_dense_multiseed.py --check    # recompute from per-run files, exit 1 on drift
"""
import argparse
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results")

# the seed-42 dose-response arms kept their canonical names when 3.6.4 shipped; every
# other run is the arm's pattern instantiated with its seed
DOSE_RESPONSE = {
    "1x256": ("policy_saturation_dense_nenv1_nsteps256.json", "policy_saturation_dense_nenv1_nsteps256_seed{}.json"),
    # the (iv) dose of 14/10: two more points on the levels-per-update axis at the
    # fixed published batch of 256 transitions, filling the gap between 1 and 4 levels
    "2x128": ("policy_saturation_dense_nenv2_nsteps128_seed42.json", "policy_saturation_dense_nenv2_nsteps128_seed{}.json"),
    "4x64": ("policy_saturation_dense_nenv4_nsteps64.json", "policy_saturation_dense_nenv4_nsteps64_seed{}.json"),
    "8x32": ("policy_saturation_dense_nenv8_nsteps32_seed42.json", "policy_saturation_dense_nenv8_nsteps32_seed{}.json"),
    "16x16": ("policy_saturation_dense_nenv16_nsteps16.json", "policy_saturation_dense_nenv16_nsteps16_seed{}.json"),
    "16x256": ("policy_saturation_dense_nenv16_nsteps256.json", "policy_saturation_dense_nenv16_nsteps256_seed{}.json"),
    "coinrun": ("coinrun_dense_curve.json", "coinrun_dense_seed{}.json"),
}
MECHANISM = {  # one-seed-42-and-43 pairs, all at n_envs=1 -- the levers of 3.6.5
    "repeat4": ("policy_saturation_dense_repeat4_seed{}.json", "action_repeat=4 at the published batch"),
    "nsteps1024": ("policy_saturation_dense_nsteps1024_seed{}.json", "n_steps=1024 (batch 1024, one stream)"),
    "entcoef001": ("policy_saturation_dense_entcoef001_seed{}.json", "ent_coef=0.01 at the published batch"),
}
DEAD, ALIVE = 5e-5, 1e-4


def reduce_run(d):
    c = d["curve"]
    tv = [r["cross_frame_tv_mean"] for r in c]
    p = d["_provenance"]
    return {
        "seed": p["seed"], "file": os.path.basename(p.get("_file", "")),
        "n_envs": p["n_envs"], "n_steps": p["n_steps"],
        "action_repeat": p.get("action_repeat", 1), "ent_coef": p.get("ent_coef", 0.0),
        "seconds": p["training_seconds"],
        "first_dead_at": next((r["steps"] for r in c if r["cross_frame_tv_mean"] < DEAD), None),
        "alive_probes": sum(1 for v in tv if v > ALIVE),
        "peak_tv": max(tv), "peak_at": c[tv.index(max(tv))]["steps"],
        "final_tv": tv[-1], "final_v_std": c[-1]["v_std"],
        "final_entropy": c[-1]["entropy_mean_nats"],
        "l1_max": max(r["tanh_saturated_by_layer"][0] for r in c),
        "cross_game_tv": d["cross_game"]["max_pairwise_tv"],
        "cross_game_argmax": d["cross_game"]["distinct_argmax_across_games"],
    }


def _load(fn):
    with open(os.path.join(RESULTS, fn), encoding="utf-8") as f:
        d = json.load(f)
    d["_provenance"]["_file"] = fn
    return d


def collect():
    arms = {}
    for arm, (canonical, pattern) in DOSE_RESPONSE.items():
        files = [canonical] + [pattern.format(s) for s in (43, 44, 45, 46)]
        rows = [reduce_run(_load(fn)) for fn in files]
        arms[arm] = rows
    levers = {}
    for lever, (pattern, question) in MECHANISM.items():
        rows = [reduce_run(_load(pattern.format(s))) for s in (42, 43)]
        levers[lever] = {"question": question, "rows": rows}
    return arms, levers


def by_arm(rows):
    """The per-arm aggregates the README table prints, so no hand arithmetic on five
    numbers happens in more than one place."""
    return {
        "seeds": len(rows),
        "end_responsive": sum(1 for r in rows if r["final_tv"] > ALIVE),
        "first_dead_at": sorted((r["first_dead_at"] for r in rows), key=lambda x: (x is None, x)),
        "alive_probes": sorted(r["alive_probes"] for r in rows),
        "never_a_dead_probe": sum(1 for r in rows if r["first_dead_at"] is None),
        "cross_game_tv": [round(r["cross_game_tv"], 6) for r in rows],
        "two_argmax_seeds": [r["seed"] for r in rows if r["cross_game_argmax"] > 1],
        "seconds_total": round(sum(r["seconds"] for r in rows), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="recompute from the per-run files and fail if the merged artifact is stale")
    args = ap.parse_args()
    arms, levers = collect()
    payload = {
        "_provenance": {
            "produced_by": "summarize_dense_multiseed.py",
            "question": "does the 3.6.4 dose-response replicate across seeds, and which mechanism lever survives",
            "seeds": [42, 43, 44, 45, 46], "dead_threshold": DEAD, "alive_threshold": ALIVE,
            "end_responsive_def": "final probe cross-frame TV > 1e-4",
        },
        "runs": {arm: rows for arm, rows in arms.items()},
        "by_arm": {arm: by_arm(rows) for arm, rows in arms.items()},
        "levers": levers,
    }
    out = os.path.join(RESULTS, "policy_saturation_dense_multiseed.json")
    if args.check:
        with open(out, encoding="utf-8") as f:
            committed = json.load(f)
        if committed != payload:
            print("the merged artifact is stale: rerun summarize_dense_multiseed.py")
            return 1
        print("the merged artifact matches the per-run files")
        return 0
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    for arm, agg in payload["by_arm"].items():
        print(f"{arm:9s} end_responsive {agg['end_responsive']}/{agg['seeds']}  "
              f"first_dead {agg['first_dead_at']}  alive {agg['alive_probes']}")
    for lever, d in levers.items():
        for r in d["rows"]:
            print(f"{lever:11s} s{r['seed']}: dead={r['first_dead_at']} alive={r['alive_probes']}/101 "
                  f"final={r['final_tv']:.6f} cross={r['cross_game_tv']:.6f} argmax={r['cross_game_argmax']}")
    print(f"saved {os.path.relpath(out, BASE)}")


if __name__ == "__main__":
    raise SystemExit(main())
