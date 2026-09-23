"""Reduce the re-measured SMAX Q-learning cells against the published win-rates.

README sections 15.4 and 15.4.4 report 0.0 win-rate for every Q-learning arm and explain the
recurrent QMIX failure as a property of symmetric combat. That explanation was written while
the mixer target dropped the external reward, the replay buffer interleaved parallel
environments, and the bootstrap ignored terminations — all fixed on 21/09/2026. This script
compares the re-measured cells (produced by jax_port/marl_ql_remeasure.sh: flat 1M VDN/QMIX
and recurrent 10M QMIX) with the frozen published values so the two can be reported side by
side instead of one silently replacing the other.

Usage (from the repository root, inside WSL or not — it only reads JSON):
    python -m jax_port.marl_remeasure_report
"""
import argparse
import glob
import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# published cells are recorded with the nominal budget; a cell that stopped early is not
# comparable, so the ratio below is what makes a re-measurement admissible.
MIN_BUDGET_RATIO = 0.95


def published_marl_cells(path):
    """{(algo, map, seed): winrate} from the committed summary of the original grid."""
    with open(path, encoding="utf-8") as f:
        cells = json.load(f).get("marl", [])
    return {(c["algo"], c["map"], c["seed"]): c.get("winrate") for c in cells}


def load_remeasured(cells_dir):
    out = []
    for path in sorted(glob.glob(os.path.join(cells_dir, "*.json"))):
        with open(path, encoding="utf-8") as f:
            j = json.load(f)
        if "algo" not in j or "map" not in j:
            continue
        # the file name carries the requested budget (<algo>__<map>__seed<n>__<steps>.json),
        # which is what a cell must reach to refute a claim made at that budget.
        target = None
        parts = os.path.splitext(os.path.basename(path))[0].split("__")
        if len(parts) == 4 and parts[3].isdigit():
            target = int(parts[3])
        ev = j.get("eval") or {}
        out.append({
            "algo": j["algo"], "map": j["map"], "seed": j["seed"],
            "timesteps": j.get("timesteps"), "target_timesteps": target,
            "sps": j.get("sps"), "wall_s": j.get("wall_s"),
            "winrate": ev.get("winrate"), "eval_episodes": ev.get("episodes"),
            "mean_return": ev.get("mean_return"), "winrate_train": j.get("winrate_train"),
            "source": os.path.basename(path),
        })
    return out


def report(cells_dir, published_path, budget):
    published = published_marl_cells(published_path)
    rows, skipped, notes = [], [], []
    for c in load_remeasured(cells_dir):
        key = (c["algo"], c["map"], c["seed"])
        old = published.get(key)
        nominal = c["target_timesteps"] or budget
        if nominal is None:
            nominal = c["timesteps"]
            notes.append(f"{c['source']}: no step budget in the file name, completeness "
                         f"could not be judged")
        budget_reached = (c["timesteps"] is not None
                          and c["timesteps"] >= MIN_BUDGET_RATIO * nominal)
        if c["winrate"] is None:
            reason = "no eval block"
        elif not 0.0 <= c["winrate"] <= 1.0:
            # outside [0,1] the record itself is broken, not the arm; averaging it would move
            # the group mean without any observable error
            reason = f"winrate {c['winrate']} is not a probability"
        elif not budget_reached:
            reason = f"stopped at {c['timesteps']}/{nominal} steps"
        else:
            reason = None
        row = dict(c)
        row.update({"published_winrate": old, "target_timesteps": nominal,
                    "budget_reached": bool(budget_reached), "usable": reason is None})
        if reason is not None:
            skipped.append(f"{c['source']}: {reason}")
        elif old is not None:
            row["delta_vs_published"] = round(c["winrate"] - old, 3)
        rows.append(row)
    # group per arm, not per map: flat 1M VDN, flat 1M QMIX and recurrent 10M QMIX are three
    # different claims and averaging them over one map would be meaningless
    by_group = {}
    for row in rows:
        gk = f"{row['algo']}__{row['map']}__{(row['target_timesteps'] or 0) // 1000000}M"
        g = by_group.setdefault(gk, {"n": 0, "winrate_sum": 0.0,
                                     "published_winrate_sum": 0.0, "excluded": 0})
        if not row["usable"]:
            g["excluded"] += 1
            continue
        g["n"] += 1
        g["winrate_sum"] += row["winrate"]
        g["published_winrate_sum"] += row["published_winrate"] or 0.0
    for g in by_group.values():
        g["winrate_mean"] = round(g.pop("winrate_sum") / g["n"], 3) if g["n"] else 0.0
        g["published_winrate_mean"] = (round(g.pop("published_winrate_sum") / g["n"], 3)
                                       if g["n"] else 0.0)
    return {"_provenance": {
                "produced_by": "jax_port/marl_remeasure_report.py",
                "cells_dir": os.path.relpath(cells_dir, BASE).replace(os.sep, "/"),
                "published_reference": os.path.relpath(published_path, BASE).replace(os.sep, "/"),
                "code_state": "after the 21/09/2026 fixes: MARLSequentialBuffer per-environment "
                              "contiguity, terminal mask in the bootstrap, QMIX mixer target "
                              "scaled by r + gamma*(1-d)*Q_tot",
                "hyperparameters": "train_ql defaults (lr 1e-4, 32 envs, tau=0.0 hard target "
                                   "copy every 500 steps) — identical to marl_10m_run.sh",
                "min_budget_ratio": MIN_BUDGET_RATIO,
                "skipped": skipped,
                "budget_notes": notes,
                "n_cells": len(rows),
            },
            "cells": rows, "by_group": by_group}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cells_dir", default=os.path.join(BASE, "jax_port", "results_grade",
                                                            "marl_remeasure"))
    parser.add_argument("--published", default=os.path.join(BASE, "jax_port", "cells_summary.json"))
    parser.add_argument("--budget", type=int, default=None,
                        help="fallback per-cell step budget when the file name does not carry one")
    parser.add_argument("--out", default=os.path.join(BASE, "jax_port", "marl_remeasure_summary.json"))
    args = parser.parse_args()

    rep = report(args.cells_dir, args.published, args.budget)
    if not rep["cells"]:
        print(f"no re-measured cells under {args.cells_dir} — run jax_port/marl_ql_remeasure.sh "
              f"inside WSL first")
        raise SystemExit(1)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    print(f"{'cell':38s} {'steps':>9s} {'winrate':>8s} {'published':>10s} {'delta':>7s}")
    for c in rep["cells"]:
        delta = c.get("delta_vs_published")
        steps = c["timesteps"] if c["timesteps"] is not None else 0
        wr = c["winrate"] if c["winrate"] is not None else float("nan")
        old = c["published_winrate"] if c["published_winrate"] is not None else float("nan")
        print(f"{c['algo']+'__'+c['map']+'__seed'+str(c['seed']):38s} {steps:>9} {wr:>8.3f} "
              f"{old:>10.3f} {(f'{delta:+.3f}' if delta is not None else '   n/a'):>7s}"
              + ("" if c["usable"] else "  [excluded]"))
    for mp, g in sorted(rep["by_group"].items()):
        print(f"{mp}: {g['n']} admissible cells, win-rate {g['winrate_mean']:.3f} "
              f"(published {g['published_winrate_mean']:.3f})")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
