"""Section 21 re-asked with the response measure as the target (01/11/2026).

Section 21 identified the benchmark from its *return*: P1 fitted the forward law of return
on the protocol, P2 asked whether the measured return identifies the architecture that
produced it (it barely does), P3 inverted the law for budget. Section 3.6 then showed that
return is not what separates the front ends -- the cross-game response measure is. This
script asks the same three questions with the response as the observable, on the two grids
that carry it:

  sb3   results/suite_validity_nenv16_grid.json  -- 16 arms x 3 games x 5 seeds, 16x256 recipe
  port  jax_port/port_response_grid.json         -- the port's 16 configs, re-trained with the
                                                    response attached (jax_port/audit_port_response.sh)

P2 (identification). Within one (grid, game) stratum the protocol is identical for every
architecture. Each architecture's reference profile averages k seeds; a query averages j
held-out seeds; a nearest-centroid rule (each observable z-scored by the spread of the
reference centroids) names the generating architecture. Observables: return alone, response
(TV) alone, both. Program-identical aliases are one class (sb3: ae=recon,
contrastive=c_aug_noise; port: classic=ae=recon, aug_noise=contrastive), since no observable
can separate two runs of the same program. Scored at the architecture level (chance 1/C) and
at the family level of section 21 (chance 1/F, every member architecture pooled into its
family's centroid).

The family level also corrects a defect of the published section 21.5: its p2_hidden_family
keyed the profile by family and seed, so each configuration of a family overwrote the
previous one and every "family" was in fact the last configuration read (cnn_attention =
spatial, cnn_deep = resnet18, world_model = vae, augmentation = aug_noise). `p2_published_
defect` re-runs both versions on the archived port returns so the size of the error is on
record; symbolic_regression.py itself was fixed the same day (03/11/2026).

P1/P3 (budget law). On the SB3 cells measured at 100k, 250k and 500k (classic, cbam, vit,
mlp_vector; starpilot/dodgeball at all three budgets, bossfight at 100k and 500k) the slope
per decade of budget is fitted for both observables, with a seed-bootstrap CI, and a
symbolic search over (log10 N, game) is cross-validated for each target. P3 inverts each
reader's response law for the budget at which its mean TV would reach 0.5.

    py -3.10 response_identification.py            # writes results/response_identification.json
    py -3.10 response_identification.py --check    # fails if the artifact is stale
"""
import argparse
import json
import math
import os
from itertools import combinations

import numpy as np

from symbolic_regression import FAMILY_OF, wilson_ci

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results")
OUT = os.path.join(RESULTS, "response_identification.json")
SEEDS = (42, 43, 44, 45, 46)
GAMES = ("bossfight", "starpilot", "dodgeball")
SB3_NAME = {"c_aug_color": "aug_color", "c_aug_crop": "aug_crop", "c_aug_noise": "aug_noise",
            "mlp_vector": "mlp"}
SB3_ALIASES = {"recon": "ae", "c_aug_noise": "contrastive"}
PORT_ALIASES = {"ae": "classic", "recon": "classic", "contrastive": "aug_noise"}
OBSERVABLES = {"return": (0,), "response": (1,), "both": (0, 1)}
BUDGET_ARMS = ("classic", "cbam", "vit", "mlp_vector")
# section 21.5's pooled table as published (results/symbolic_regression.json at commit 7c518ef,
# before symbolic_regression.p2_hidden_family was fixed on 03/11/2026)
PUBLISHED_21_5 = {"k_1_j_1": 0.133, "k_1_j_2": 0.1449, "k_1_j_3": 0.1515, "k_2_j_1": 0.1545,
                  "k_2_j_2": 0.1407, "k_2_j_3": 0.1568, "k_3_j_1": 0.1723, "k_3_j_2": 0.1727,
                  "k_4_j_1": 0.203}
READERS = ("classic", "cbam")


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sb3_strata():
    g = _load(os.path.join(RESULTS, "suite_validity_nenv16_grid.json"))["games"]
    out = {}
    for game in GAMES:
        classes = {}
        for arm, d in g[game].items():
            if arm in SB3_ALIASES or len(d["returns"]) != len(SEEDS):
                continue  # an alias, or vae/bossfight whose seed 42 diverged to NaN
            fam = FAMILY_OF[SB3_NAME.get(arm, arm)]
            classes[arm] = {"family": fam, "obs": dict(zip(SEEDS, zip(d["returns"], d["tvs"])))}
        out[game] = classes
    return out


def port_strata(path=os.path.join(BASE, "jax_port", "port_response_grid.json")):
    g = _load(path)["games"]
    out = {}
    for game in GAMES:
        classes = {}
        for cfg, d in g[game].items():
            if cfg in PORT_ALIASES or len(d["returns"]) != len(SEEDS):
                continue
            classes[cfg] = {"family": FAMILY_OF[cfg],
                            "obs": dict(zip(d["seeds"], zip(d["returns"], d["tvs"])))}
        out[game] = classes
    return out


def _nearest(query, centroids, dims):
    keys = sorted(centroids)  # ties go to the first label in sorted order, as sklearn's NearestCentroid
    C = np.array([[centroids[k][d] for d in dims] for k in keys])
    scale = C.std(axis=0)
    scale[scale == 0] = 1.0
    q = np.array([query[d] for d in dims])
    return keys[int(np.argmin((((C - q) / scale) ** 2).sum(axis=1)))]


def identify(classes, level, dims):
    """Exhaustive (k reference seeds, j query seeds) splits; returns {k_j: (hits, n)}."""
    names = sorted(classes)
    labels = sorted({classes[c]["family"] for c in names}) if level == "family" else names
    res = {}
    for j in range(1, 4):
        for k in range(1, len(SEEDS) - j + 1):
            hits = n = 0
            for qs in combinations(SEEDS, j):
                rest = [s for s in SEEDS if s not in qs]
                for ref in combinations(rest, k):
                    prof = {c: np.mean([classes[c]["obs"][s] for s in ref], axis=0) for c in names}
                    if level == "family":
                        cent = {f: np.mean([prof[c] for c in names if classes[c]["family"] == f], axis=0)
                                for f in labels}
                    else:
                        cent = prof
                    for c in names:
                        q = np.mean([classes[c]["obs"][s] for s in qs], axis=0)
                        truth = classes[c]["family"] if level == "family" else c
                        hits += int(_nearest(q, cent, dims) == truth)
                        n += 1
            res[f"k_{k}_j_{j}"] = (hits, n)
    return res, len(labels)


def p2(strata_by_grid):
    out = {}
    for grid, strata in strata_by_grid.items():
        g = {}
        for level in ("architecture", "family"):
            for obs, dims in OBSERVABLES.items():
                pooled, per = {}, {}
                for game, classes in strata.items():
                    res, n_labels = identify(classes, level, dims)
                    per[game] = {"labels": n_labels, "chance": round(1 / n_labels, 4),
                                 "k1_j1": round(res["k_1_j_1"][0] / res["k_1_j_1"][1], 4),
                                 "k4_j1": round(res["k_4_j_1"][0] / res["k_4_j_1"][1], 4),
                                 "k4_j1_ci95_wilson": wilson_ci(*res["k_4_j_1"])}
                    for kk, (h, n) in res.items():
                        pooled.setdefault(kk, [0, 0])
                        pooled[kk][0] += h
                        pooled[kk][1] += n
                g[f"{level}|{obs}"] = {
                    "per_game": per,
                    "pooled": {kk: {"accuracy": round(h / n, 4), "n": n, "ci95_wilson": wilson_ci(h, n)}
                               for kk, (h, n) in sorted(pooled.items())},
                    "mean_chance": round(float(np.mean([p["chance"] for p in per.values()])), 4)}
        out[grid] = g
    return out


def p2_published_defect():
    """The published p2_hidden_family (overwrite by family) against the corrected pooling, on
    the archived port returns of the same four strata section 21.5 reported."""
    allc = _load(os.path.join(BASE, "jax_port", "cells_summary.json"))
    out, pooled = {}, {"published": {}, "corrected": {}}
    for suite in ("hard", "main"):
        cells = allc[suite]
        for game in sorted({c["game"] for c in cells}):
            mine = [c for c in cells if c["game"] == game]
            overwrite, survivors, cfgs = {}, {}, {}
            for c in mine:                                # the published loop, in effect
                overwrite.setdefault(FAMILY_OF[c["cfg"]], {})[c["seed"]] = c["eval_unseen"]
                survivors[FAMILY_OF[c["cfg"]]] = c["cfg"]
                cfgs.setdefault(c["cfg"], {})[c["seed"]] = c["eval_unseen"]
            pub = {f: {"family": f, "obs": {s: (v, 0.0) for s, v in d.items()}} for f, d in overwrite.items()}
            cor = {cfg: {"family": FAMILY_OF[cfg], "obs": {s: (v, 0.0) for s, v in d.items()}}
                   for cfg, d in cfgs.items()}
            r_pub, _ = identify(pub, "architecture", (0,))
            r_cor, f_cor = identify(cor, "family", (0,))
            for name, r in (("published", r_pub), ("corrected", r_cor)):
                for kk, (h, n) in r.items():
                    pooled[name].setdefault(kk, [0, 0])
                    pooled[name][kk][0] += h
                    pooled[name][kk][1] += n
            out[f"{game}|{suite}"] = {
                "families": f_cor, "chance": round(1 / f_cor, 4),
                "published_k1_j1": round(r_pub["k_1_j_1"][0] / r_pub["k_1_j_1"][1], 4),
                "corrected_k1_j1": round(r_cor["k_1_j_1"][0] / r_cor["k_1_j_1"][1], 4),
                "published_k4_j1": round(r_pub["k_4_j_1"][0] / r_pub["k_4_j_1"][1], 4),
                "corrected_k4_j1": round(r_cor["k_4_j_1"][0] / r_cor["k_4_j_1"][1], 4),
                "published_profile_was_config": dict(sorted(survivors.items()))}
    out["_pooled"] = {name: {kk: round(h / n, 4) for kk, (h, n) in sorted(d.items())}
                      for name, d in pooled.items()}
    # both columns above use this module's exact-tie rule; the artifact's own numbers (sklearn's
    # float distances break the frequent bossfight zero-return ties differently) are kept beside
    # them, so the comparison is like-for-like: the published values as section 21.5 printed
    # them, and the current artifact, which symbolic_regression.py regenerates with the fix
    out["_pooled"]["published_artifact"] = PUBLISHED_21_5
    fixed = _load(os.path.join(RESULTS, "symbolic_regression.json"))["P2_hidden_family_identification"]
    out["_pooled"]["fixed_artifact"] = {kk: v["nearest_centroid_accuracy"]
                                        for kk, v in sorted(fixed["pooled_across_strata"].items())}
    return out


def budget_cells():
    """(arm, game, budget, seed, return, tv) for the SB3 budget line."""
    grid = _load(os.path.join(RESULTS, "suite_validity_nenv16_grid.json"))["games"]
    rows = []
    for game in GAMES:
        for arm in BUDGET_ARMS:
            d = grid[game][arm]
            rows += [(arm, game, 100_000, s, r, t) for s, r, t in zip(SEEDS, d["returns"], d["tvs"])]
        for budget in (250_000, 500_000):
            path = os.path.join(RESULTS, f"suite_validity_nenv16_{game}_{budget // 1000}k.json")
            if not os.path.exists(path):
                continue
            arms = _load(path)["arms"]
            for key, r in arms.items():
                arm, seed = key.split("@")
                if arm in BUDGET_ARMS and "error" not in r:
                    rows.append((arm, game, budget, int(seed), r["return_10eps"], r["max_pairwise_tv"]))
    return rows


def _slope(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    return float(np.polyfit(xs, ys, 1)[0]), float(np.polyfit(xs, ys, 1)[1])


def p1_p3(rows, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    laws = {}
    for arm in BUDGET_ARMS:
        laws[arm] = {}
        for target, col in (("return", 4), ("response", 5)):
            per_game = {}
            for game in GAMES:
                cells = [r for r in rows if r[0] == arm and r[1] == game]
                budgets = sorted({r[2] for r in cells})
                if len(budgets) < 2:
                    continue
                by_b = {b: [r[col] for r in cells if r[2] == b] for b in budgets}
                b, a = _slope([math.log10(r[2]) for r in cells], [r[col] for r in cells])
                boots = []
                for _ in range(n_boot):
                    xs, ys = [], []
                    for bb, vals in by_b.items():
                        pick = rng.integers(0, len(vals), len(vals))
                        xs += [math.log10(bb)] * len(vals)
                        ys += [vals[i] for i in pick]
                    boots.append(_slope(xs, ys)[0])
                lo, hi = np.percentile(boots, [2.5, 97.5])
                entry = {"budgets": budgets, "means": {str(bb): round(float(np.mean(v)), 4) for bb, v in by_b.items()},
                         "slope_per_decade": round(b, 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
                         "excludes_zero": bool(lo > 0 or hi < 0)}
                if target == "response" and arm in READERS and b > 0:
                    # P3: invert the fitted law for the budget at which mean TV reaches 0.5
                    n_star = 10 ** ((0.5 - a) / b)
                    entry["budget_for_tv_0p5"] = int(round(n_star, -3))
                    entry["budget_for_tv_0p5_extrapolated"] = bool(n_star > max(budgets) or n_star < min(budgets))
                per_game[game] = entry
            laws[arm][target] = per_game
    return laws


def p1_symbolic(rows, args):
    """Cross-validated symbolic search over (log10 N, game) for each target, readers pooled.

    The search runs on the z-scored target: gplearn's fitness is MAE plus parsimony x program
    length, so with section 21's parsimony (0.005, sized for returns of order 1) a target on
    the TV scale (spread ~0.15) pays more per node than any node can earn, and the search
    returns a constant. Standardizing makes the parsimony price the same for both targets;
    scores are reported back in the target's own units. Two cross-validated linear fits are the
    reference: game intercepts only, and game intercepts plus a per-game slope on log10 N --
    the gap between them is what budget explains.
    """
    from sklearn.linear_model import LinearRegression
    from symbolic_regression import cv_predict, expression, scores, sr_model
    cells = [r for r in rows if r[0] in READERS]
    names = ["log10_N"] + [f"is_{g}" for g in GAMES]
    X = np.array([[math.log10(r[2])] + [1.0 if r[1] == g else 0.0 for g in GAMES] for r in cells])
    X_lin = np.hstack([X[:, 1:], X[:, 1:] * X[:, :1]])
    groups = np.array([r[3] for r in cells])
    out = {"cells": len(cells), "arms": list(READERS), "features": names,
           "target_scaling": "z-scored for the search, scores in original units"}
    for target, col in (("return", 4), ("response", 5)):
        y = np.array([r[col] for r in cells], float)
        mu, sd = float(y.mean()), float(y.std())
        z = (y - mu) / sd
        m = sr_model(args)
        pred = cv_predict(m, X, z, groups) * sd + mu
        m.fit(X, z)
        lin = cv_predict(LinearRegression(fit_intercept=False), X_lin, y, groups)
        game = cv_predict(LinearRegression(fit_intercept=False), X[:, 1:], y, groups)
        out[target] = {**scores(y, pred), "expression_z": expression(m, names),
                       "linear_game_only": scores(y, game), "linear_per_game_slope": scores(y, lin)}
    return out


def build(args):
    strata = {"sb3_16x256": sb3_strata()}
    port_path = os.path.join(BASE, "jax_port", "port_response_grid.json")
    if os.path.exists(port_path):
        strata["port_64x128"] = port_strata(port_path)
    rows = budget_cells()
    return {
        "_provenance": {
            "produced_by": "response_identification.py",
            "question": "section 21's P1/P2/P3 with the cross-game response measure as the observable",
            "grids": {"sb3_16x256": "results/suite_validity_nenv16_grid.json",
                      "port_64x128": "jax_port/port_response_grid.json"},
            "aliases_merged": {"sb3_16x256": SB3_ALIASES, "port_64x128": PORT_ALIASES},
            "search": {"pop": args.pop, "gens": args.gens, "parsimony": args.parsimony, "seed": args.seed},
        },
        "P2_identification": p2(strata),
        "P2_published_defect": p2_published_defect(),
        "P1_P3_budget_law": p1_p3(rows),
        "P1_symbolic": p1_symbolic(rows, args),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pop", type=int, default=1000)
    ap.add_argument("--gens", type=int, default=40)
    ap.add_argument("--parsimony", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    out = build(args)
    if args.check:
        assert _load(OUT) == out, "stale: rerun response_identification.py"
        print("OK: results/response_identification.json is current")
        return
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    for grid, g in out["P2_identification"].items():
        for key, v in g.items():
            print(f"{grid:12s} {key:22s} k1j1={v['pooled']['k_1_j_1']['accuracy']:.3f} "
                  f"k4j1={v['pooled']['k_4_j_1']['accuracy']:.3f} chance~{v['mean_chance']:.3f}")
    for key, d in out["P2_published_defect"].items():
        if key != "_pooled":
            print("defect", key, d["published_k1_j1"], "->", d["corrected_k1_j1"])
    print("defect pooled", out["P2_published_defect"]["_pooled"])


if __name__ == "__main__":
    main()
