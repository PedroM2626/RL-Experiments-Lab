"""Fold the 250k budget sweep against the 100k grid it came from (README 3.6.8).

The 3.6.7 grid measured every arm x game x seed at 100k under the responsive recipe;
this sweep re-measures four arms -- the two that never read (vit, mlp_vector) and two
that read on every cell (classic, cbam) -- on starpilot and dodgeball at 250k, same
recipe, same seeds. Every 250k cell therefore pairs with its own 100k cell, and the
question the fold answers is paired: does the extra 150k steps change return and
response, and does it change them differently for the arms that read and the arms
that do not.

Inputs: results/suite_validity_nenv16_{starpilot,dodgeball}_250k.json (the audit's
per-game fusions of this sweep's shards) and results/suite_validity_nenv16_grid.json
(the 3.6.7 fold). Output: results/suite_validity_nenv16_budget250k.json. Nothing here
trains anything; --check re-folds and compares.

Usage:
    py -3.10 summarize_budget_nenv16.py            # writes the folded artifact
    py -3.10 summarize_budget_nenv16.py --check   # fails if the artifact is stale
"""
import argparse
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results")
GAMES = ("starpilot", "dodgeball")
ARMS = ("classic", "cbam", "vit", "mlp_vector")
SEEDS = (42, 43, 44, 45, 46)


def _load(name):
    with open(os.path.join(RESULTS, name), encoding="utf-8") as f:
        return json.load(f)


def fold():
    grid = _load("suite_validity_nenv16_grid.json")["games"]
    out = {}
    for arm in ARMS:
        out[arm] = {"reads": arm in ("classic", "cbam"), "never_reads": arm in ("vit", "mlp_vector")}
        for game in GAMES:
            fresh = _load(f"suite_validity_nenv16_{game}_250k.json")["arms"]
            rs_new, tvs_new = [], []
            for seed in SEEDS:
                row = fresh[f"{arm}@{seed}"]
                if "error" in row:
                    raise SystemExit(f"{game}/{arm}@{seed} errored -- fold refuses a partial sweep")
                rs_new.append(row["return_10eps"])
                tvs_new.append(row["max_pairwise_tv"])
            old = grid[game][arm]
            rs_old = old["returns"]
            m_old, m_new = sum(rs_old) / len(rs_old), sum(rs_new) / len(rs_new)
            tv_old = old["tvs"]
            out[arm][game] = {
                "returns_100k": rs_old, "returns_250k": rs_new,
                "mean_100k": round(m_old, 3), "mean_250k": round(m_new, 3),
                "delta": round(m_new - m_old, 3),
                # paired per-seed deltas: the honest unit at n=5
                "paired_deltas": [round(n - o, 3) for n, o in zip(rs_new, rs_old)],
                "tv_mean_100k": round(sum(tv_old) / len(tv_old), 6),
                "tv_mean_250k": round(sum(tvs_new) / len(tvs_new), 6),
                "tv_cells_100k_over_0p1": sum(1 for t in tv_old if t >= 0.1),
                "tv_cells_250k_over_0p1": sum(1 for t in tvs_new if t >= 0.1),
            }
    return {
        "_provenance": {
            "produced_by": "summarize_budget_nenv16.py",
            "question": "does a 2.5x budget change return and response differently for "
                        "arms that read the observation and arms that do not",
            "recipe": "n_envs=16 x n_steps=256, seeds 42-46, paired with the 3.6.7 100k cells",
            "arms": list(ARMS), "games": list(GAMES),
        },
        "arms": out,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    payload = fold()
    out = os.path.join(RESULTS, "suite_validity_nenv16_budget250k.json")
    if args.check:
        with open(out, encoding="utf-8") as f:
            committed = json.load(f)
        if committed != payload:
            print("the budget artifact is stale: rerun summarize_budget_nenv16.py")
            return 1
        print("the budget artifact matches the merged files")
        return 0
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    for arm in ARMS:
        for game in GAMES:
            d = payload["arms"][arm][game]
            print(f"{arm:11s} {game:10s} ret {d['mean_100k']:5.2f} -> {d['mean_250k']:5.2f} "
                  f"(delta {d['delta']:+.2f})   TV {d['tv_mean_100k']:.3f} -> {d['tv_mean_250k']:.3f}")
    print(f"saved {os.path.relpath(out, BASE)}")


if __name__ == "__main__":
    raise SystemExit(main())
