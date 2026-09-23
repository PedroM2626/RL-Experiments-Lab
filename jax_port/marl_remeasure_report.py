"""Reduce the recurrent-QMIX re-measurement cells against the published win-rates.

README section 15.4.4 reports 0.0 win-rate for `qmix-recurrent` on 3m and 2s3z at 10M steps
and explains it as a property of symmetric combat. That explanation was written while the
mixer target dropped the external reward, the replay buffer interleaved parallel
environments, and the bootstrap ignored terminations — all fixed on 21/09/2026. This script
compares the re-measured cells (produced by jax_port/marl_qmix_remeasure.sh) with the frozen
published values so the two can be reported side by side instead of one silently replacing
the other.

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
        ev = j.get("eval") or {}
        out.append({
            "algo": j["algo"], "map": j["map"], "seed": j["seed"],
            "timesteps": j.get("timesteps"), "sps": j.get("sps"), "wall_s": j.get("wall_s"),
            "winrate": ev.get("winrate"), "eval_episodes": ev.get("episodes"),
            "mean_return": ev.get("mean_return"), "winrate_train": j.get("winrate_train"),
            "source": os.path.basename(path),
        })
    return out


def report(cells_dir, published_path, budget):
    published = published_marl_cells(published_path)
    rows, skipped = [], []
    for c in load_remeasured(cells_dir):
        key = (c["algo"], c["map"], c["seed"])
        old = published.get(key)
        nominal = budget if budget else c["timesteps"]
        admissible = c["timesteps"] is not None and c["timesteps"] >= MIN_BUDGET_RATIO * nominal
        row = dict(c)
        row.update({"published_winrate": old, "target_timesteps": nominal,
                    "budget_reached": bool(admissible)})
        if old is not None and c["winrate"] is not None and admissible:
            row["delta_vs_published"] = round(c["winrate"] - old, 3)
        if c["winrate"] is None:
            skipped.append(f"{c['source']}: no eval block")
        elif not admissible:
            skipped.append(f"{c['source']}: stopped at {c['timesteps']}/{nominal} steps")
        rows.append(row)
    by_map = {}
    for row in rows:
        g = by_map.setdefault(row["map"], {"n": 0, "winrate_sum": 0.0,
                                           "published_winrate_sum": 0.0, "incomplete": 0})
        if not row["budget_reached"] or row["winrate"] is None:
            g["incomplete"] += 1
            continue
        g["n"] += 1
        g["winrate_sum"] += row["winrate"]
        g["published_winrate_sum"] += row["published_winrate"] or 0.0
    for g in by_map.values():
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
                "n_cells": len(rows),
            },
            "cells": rows, "by_map": by_map}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cells_dir", default=os.path.join(BASE, "jax_port", "results_grade",
                                                            "marl_remeasure"))
    parser.add_argument("--published", default=os.path.join(BASE, "jax_port", "cells_summary.json"))
    parser.add_argument("--budget", type=int, default=10000000,
                        help="nominal per-cell step budget; cells below it are reported, not averaged")
    parser.add_argument("--out", default=os.path.join(BASE, "jax_port", "marl_remeasure_summary.json"))
    args = parser.parse_args()

    rep = report(args.cells_dir, args.published, args.budget)
    if not rep["cells"]:
        print(f"no re-measured cells under {args.cells_dir} — run jax_port/marl_qmix_remeasure.sh "
              f"inside WSL first")
        raise SystemExit(1)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    print(f"{'cell':38s} {'steps':>9s} {'winrate':>8s} {'published':>10s} {'delta':>7s}")
    for c in rep["cells"]:
        delta = c.get("delta_vs_published")
        print(f"{c['algo']+'__'+c['map']+'__seed'+str(c['seed']):38s} {c['timesteps']:>9} "
              f"{(c['winrate'] if c['winrate'] is not None else float('nan')):>8.3f} "
              f"{(c['published_winrate'] if c['published_winrate'] is not None else float('nan')):>10.3f} "
              f"{(f'{delta:+.3f}' if delta is not None else '   n/a'):>7s}"
              + ("" if c["budget_reached"] else "  [incomplete]"))
    for mp, g in sorted(rep["by_map"].items()):
        print(f"{mp}: {g['n']} admissible cells, win-rate {g['winrate_mean']:.3f} "
              f"(published {g['published_winrate_mean']:.3f})")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
