"""Decide whether the port's exploration arms did anything a control arm would not.

README section 15.4.3 lists 40 `exploration` cells whose ICM/RND/NGU configs score differently
from their PPO control (up to +-0.5 mean reward on maze/heist). Those cells were produced before
any bonus bookkeeping existed, so nothing in them shows the intrinsic reward reached the policy —
and the arms inject ~1e-4 reward per step against a 0/1 extrinsic signal, which is small enough
that chaotic run-to-run drift could produce the same spread on its own.

This script compares three groups per (game, arm), all at identical hyperparameters:

    ppo            no mechanism at all                     (published)
    arm @ beta=0   mechanism trains, nothing injected      (the control, this run)
    arm @ beta=0.01 mechanism trains and injects           (published)

|arm@0 - ppo| is the drift floor: two groups that both inject nothing, differing only by the
presence of the extra network and its RNG stream. |arm@0.01 - ppo| is that same drift plus any
real effect. The arm's published difference is only attributable to curiosity if it clears the
floor, so the report says which side of that line each cell group falls on.

Usage:
    python -m jax_port.exploration_control_report
"""
import argparse
import glob
import json
import math
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = ("icm", "rnd", "ngu")
GAMES = ("maze", "heist")
# a control arm is only a control if it really injected nothing; the tolerance is in reward
# units, and bonus_mean must still be positive to prove the mechanism was computing
EFFECTIVE_TOL = 1e-9


def mean_se(vals):
    n = len(vals)
    m = sum(vals) / n
    if n < 2:
        return m, None, n
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    return m, math.sqrt(var / n), n


def cell_value(j):
    ev = j.get("eval_unseen")
    if isinstance(ev, dict):
        return ev.get("mean")
    return ev


def published_cells(summary_path):
    with open(summary_path, encoding="utf-8") as f:
        cells = json.load(f).get("exploration", [])
    out = {}
    for c in cells:
        out.setdefault((c["game"], c["cfg"]), []).append(c["eval_unseen"])
    return out


def control_cells(cells_dir):
    """{(game, cfg): [eval_unseen]} plus the bonus bookkeeping of each cell."""
    out, stats, bad = {}, [], []
    for path in sorted(glob.glob(os.path.join(cells_dir, "*.json"))):
        with open(path, encoding="utf-8") as f:
            j = json.load(f)
        game, cfg = j.get("game"), j.get("explore")
        val = cell_value(j)
        if game not in GAMES or cfg not in ARMS or val is None:
            continue
        st = j.get("explore_stats")
        if not st:
            bad.append(f"{os.path.basename(path)}: no explore_stats, produced before the "
                       f"bookkeeping existed - not usable as a control")
            continue
        if abs(st["effective_bonus"]) > EFFECTIVE_TOL:
            bad.append(f"{os.path.basename(path)}: beta is {st['beta']} and it injected "
                       f"{st['effective_bonus']:.3e} - this cell is not a control arm")
            continue
        if st["bonus_mean"] <= 0.0:
            bad.append(f"{os.path.basename(path)}: bonus_mean {st['bonus_mean']} - the "
                       f"mechanism computed nothing, so it proves nothing about scale")
            continue
        out.setdefault((game, cfg), []).append(val)
        stats.append(dict(st, cell=os.path.basename(path)))
    return out, stats, bad


def build(control_dir, summary_path):
    pub = published_cells(summary_path)
    ctrl, stats, bad = control_cells(control_dir)
    groups = {}
    for game in GAMES:
        for arm in ARMS:
            ppo = pub.get((game, "ppo"), [])
            arm01 = pub.get((game, arm), [])
            arm0 = ctrl.get((game, arm), [])
            if not (ppo and arm01 and arm0):
                continue
            p_m, p_se, p_n = mean_se(ppo)
            a_m, a_se, a_n = mean_se(arm01)
            c_m, c_se, c_n = mean_se(arm0)
            drift = c_m - p_m
            effect = a_m - p_m
            groups[f"{game}_{arm}"] = {
                "game": game, "arm": arm,
                "ppo": {"mean": round(p_m, 3), "se": None if p_se is None else round(p_se, 3),
                        "n_seeds": p_n, "source": "published cells_summary.json"},
                "arm_beta_0.01": {"mean": round(a_m, 3),
                                  "se": None if a_se is None else round(a_se, 3),
                                  "n_seeds": a_n, "source": "published cells_summary.json"},
                "arm_beta_0": {"mean": round(c_m, 3),
                               "se": None if c_se is None else round(c_se, 3),
                               "n_seeds": c_n, "source": "control run"},
                "drift_floor": round(abs(drift), 3),
                "published_effect": round(effect, 3),
                # the ratio is only meaningful with n=5 per group, where each mean carries an
                # SE of its own; it is reported as an ordering, not as a test statistic
                "effect_over_drift": (round(abs(effect) / abs(drift), 2)
                                      if abs(drift) > 1e-9 else None),
                "attributable_to_mechanism": bool(abs(drift) > 1e-9 and abs(effect) > abs(drift)),
            }
    by_kind = {}
    for s in stats:
        by_kind.setdefault(s["kind"], []).append(s["bonus_mean"])
    bonus_mean = {k: round(sum(v) / len(v), 8) for k, v in sorted(by_kind.items())}
    return {
        "_provenance": {
            "produced_by": "jax_port/exploration_control_report.py",
            "control_cells_dir": os.path.relpath(control_dir, BASE).replace(os.sep, "/"),
            "published_reference": "jax_port/cells_summary.json (suite 'exploration')",
            "control_definition": "same network, same online update, beta=0 so the bonus is "
                                  "computed but never added to the reward",
            "every_control_cell_injected_nothing": bool(stats) and all(
                abs(s["effective_bonus"]) <= EFFECTIVE_TOL for s in stats),
            "bonus_mean_by_kind": bonus_mean,
            "rejected_cells": bad,
            "caveat": "n=5 seeds per group; these are point comparisons, not significance "
                      "tests - see README section 3.8",
        },
        "groups": groups,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--control-dir", default=os.path.join(BASE, "jax_port", "results_grade",
                                                          "exploration_control", "exploration"))
    ap.add_argument("--published", default=os.path.join(BASE, "jax_port", "cells_summary.json"))
    ap.add_argument("--out", default=os.path.join(BASE, "jax_port",
                                                  "exploration_control_summary.json"))
    args = ap.parse_args()

    if not glob.glob(os.path.join(args.control_dir, "*.json")):
        print(f"no control cells under {args.control_dir}. Produce them with:\n"
              f"  python -m jax_port.run_grade --suite exploration --configs icm rnd ngu \\\n"
              f"      --explore-beta 0 --out-dir jax_port/results_grade/exploration_control")
        raise SystemExit(1)

    rep = build(args.control_dir, args.published)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    print(f"{'group':14s} {'ppo':>7s} {'arm@0.01':>9s} {'arm@0':>7s} "
          f"{'drift':>7s} {'effect':>7s}  attributable?")
    for name, g in sorted(rep["groups"].items()):
        print(f"{name:14s} {g['ppo']['mean']:7.2f} {g['arm_beta_0.01']['mean']:9.2f} "
              f"{g['arm_beta_0']['mean']:7.2f} {g['drift_floor']:7.2f} "
              f"{g['published_effect']:+7.2f}  "
              f"{'YES' if g['attributable_to_mechanism'] else 'no (within drift)'}")
    if rep["_provenance"]["rejected_cells"]:
        print("rejected:", *rep["_provenance"]["rejected_cells"], sep="\n  ")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
