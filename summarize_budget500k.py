"""Fold the 500k budget sweep against the 100k grid and the 250k sweep (README 3.6.13).

The last budget question: does five hundred thousand steps change what return and
response do under the responsive recipe -- and does bossfight, flat near zero at 100k
(and untested at 250k), ever leave zero? Sixty cells answer it: the four arms of the
budget line (classic, cbam, vit, mlp_vector) x three games x seeds 42-46 at 500k,
paired with their own 100k rows (the 3.6.7 grid) and, for starpilot/dodgeball, their
own 250k rows (the 3.6.8 sweep).

Inputs: results/suite_validity_nenv16_{starpilot,dodgeball,bossfight}_500k.json (the
audit's per-game fusions of this sweep's shards), results/suite_validity_nenv16_grid.json
and results/suite_validity_nenv16_budget250k.json. Output:
results/suite_validity_nenv16_budget500k.json. Nothing trains; --check re-folds.

Usage:
    py -3.10 summarize_budget500k.py           # writes the folded artifact
    py -3.10 summarize_budget500k.py --check   # fails if the artifact is stale
"""
import argparse
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results")
GAMES = ("starpilot", "dodgeball", "bossfight")
ARMS = ("classic", "cbam", "vit", "mlp_vector")
SEEDS = (42, 43, 44, 45, 46)


def _load(name):
    with open(os.path.join(RESULTS, name), encoding="utf-8") as f:
        return json.load(f)


def fold():
    grid = _load("suite_validity_nenv16_grid.json")["games"]
    b250 = _load("suite_validity_nenv16_budget250k.json")["arms"]
    out = {}
    for arm in ARMS:
        out[arm] = {"reads": arm in ("classic", "cbam")}
        for game in GAMES:
            fresh = _load(f"suite_validity_nenv16_{game}_500k.json")["arms"]
            rs_500, tvs_500 = [], []
            for seed in SEEDS:
                row = fresh[f"{arm}@{seed}"]
                if "error" in row:
                    raise SystemExit(f"{game}/{arm}@{seed} errored -- fold refuses a partial sweep")
                rs_500.append(row["return_10eps"])
                tvs_500.append(row["max_pairwise_tv"])
            old = grid[game][arm]
            m_100 = sum(old["returns"]) / len(old["returns"])
            m_500 = sum(rs_500) / len(rs_500)
            cell = {
                "returns_100k": old["returns"], "returns_500k": rs_500,
                "mean_100k": round(m_100, 3), "mean_500k": round(m_500, 3),
                "delta_vs_100k": round(m_500 - m_100, 3),
                "paired_deltas_vs_100k": [round(n - o, 3) for n, o in zip(rs_500, old["returns"])],
                "tv_mean_100k": round(sum(old["tvs"]) / len(old["tvs"]), 6),
                "tv_mean_500k": round(sum(tvs_500) / len(tvs_500), 6),
                "tv_cells_500k_over_0p1": sum(1 for t in tvs_500 if t >= 0.1),
            }
            if game in ("starpilot", "dodgeball"):
                b = b250[arm][game]
                cell["mean_250k"] = b["mean_250k"]
                cell["returns_250k"] = b["returns_250k"]
                cell["delta_vs_250k"] = round(m_500 - b["mean_250k"], 3)
                cell["tv_mean_250k"] = b["tv_mean_250k"]
            out[arm][game] = cell
    return {
        "_provenance": {
            "produced_by": "summarize_budget500k.py",
            "question": "what do return and response do at 5x the published budget, "
                        "and does bossfight ever leave zero",
            "recipe": "n_envs=16 x n_steps=256, seeds 42-46, paired with the 100k grid "
                      "(all games) and the 250k sweep (starpilot/dodgeball)",
            "arms": list(ARMS), "games": list(GAMES),
        },
        "arms": out,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    payload = fold()
    out = os.path.join(RESULTS, "suite_validity_nenv16_budget500k.json")
    if args.check:
        with open(out, encoding="utf-8") as f:
            committed = json.load(f)
        if committed != payload:
            print("the 500k artifact is stale: rerun summarize_budget500k.py")
            return 1
        print("the 500k artifact matches the merged files")
        return 0
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    for arm in ARMS:
        for game in GAMES:
            d = payload["arms"][arm][game]
            extra = f" (250k {d['mean_250k']})" if "mean_250k" in d else ""
            print(f"{arm:11s} {game:10s} ret {d['mean_100k']:5.2f}{extra} -> {d['mean_500k']:5.2f} "
                  f"(d100 {d['delta_vs_100k']:+.2f})  TV {d['tv_mean_100k']:.3f} -> {d['tv_mean_500k']:.3f} "
                  f"({d['tv_cells_500k_over_0p1']}/5 over 0.1)")
    print(f"saved {os.path.relpath(out, BASE)}")


if __name__ == "__main__":
    raise SystemExit(main())
