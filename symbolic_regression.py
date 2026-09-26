"""Symbolic regression as system identification on the benchmark's own measurements.

Every row analysed here comes from `jax_port/cells_summary.json`, the committed record of the
cells the JAX port grid executed. Nothing in this script is simulated or generated for the
analysis; the inputs are the returns the benchmark already measured.

Three questions are posed, two of them in the inverse direction:

  P1 forward law    recover R ~ f(budget, protocol) from the measurements and compare it with
                    the linear-in-log law R = a + b*log10(N) that scaling analysis assumes by
                    default. The recovered slope per decade is the quantitative version of the
                    README's "curves plateau" statement.
  P2 inverse        treat the architecture family as a hidden parameter of the system and ask
                    how well it can be recovered from the only thing the benchmark reveals -
                    the measured return - against chance and against the number of seeds
                    averaged. This is the identifiability question the 5-seed protocol raises.
  P3 inversion      solve the recovered law for the budget a target return implies and check
                    that inversion against the cells that actually achieved it.

Cross-validation is grouped by seed everywhere, so a seed never appears in both the fit and
the test split of the same configuration. `gen_gap` is never used as a feature: it is
train_minus_unseen and would contain the target.

Usage:
    py -3.10 symbolic_regression.py            # full study
    py -3.10 symbolic_regression.py --quick    # small search, for CI
"""
import argparse
import json
import math
import os
import re
import warnings

import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from gplearn.functions import make_function
from gplearn.genetic import SymbolicRegressor

BASE = os.path.dirname(os.path.abspath(__file__))
CELLS = os.path.join(BASE, "jax_port", "cells_summary.json")
RESULTS = os.path.join(BASE, "results")

MEMORY_CFGS = {"cnn1d", "gru", "lstm", "mamba", "s4", "s5", "tcn", "transformer",
               "transformer_xl"}
# cfg -> the component of the system it identifies. An unmapped cfg raises rather than being
# dropped, so a new suite cannot silently shrink the study.
FAMILY_OF = {
    "classic": "cnn_classic", "cbam": "cnn_attention", "spatial": "cnn_attention",
    "impala": "cnn_deep", "impoola": "cnn_deep", "resnet18": "cnn_deep", "vit": "cnn_vit",
    "mlp": "mlp_vector", "ae": "world_model", "vae": "world_model", "recon": "world_model",
    "contrastive": "contrastive", "aug_color": "augmentation", "aug_crop": "augmentation",
    "aug_noise": "augmentation", "curl": "contrastive", "cpc": "contrastive",
    "acl": "contrastive", "spr": "self_predictive", "spr_aug": "self_predictive",
    "gat": "gnn", "icm": "exploration", "ngu": "exploration", "rnd": "exploration",
    "ppo": "on_policy_learner", "a2c": "on_policy_learner", "dqn": "value_based",
    "qrdqn": "value_based", "dqn_lr3e-4": "value_based", "qrdqn_lr3e-4": "value_based",
    "lstm_attention": "cnn_recurrent", "cnn1d": "memory", "gru": "memory", "lstm": "memory",
    "mamba": "memory", "s4": "memory", "s5": "memory", "tcn": "memory",
    "transformer": "memory", "transformer_xl": "memory",
    "flat": "hrl_flat", "hrl": "hrl_fixed", "hrl_learned": "hrl_learned",
    "skip4": "action_repeat",
}
FORWARD_SUITES = ("algo", "aux", "budget", "exploration", "gnn", "hard", "main", "pilot",
                  "spr", "temporal", "temporal_hard")
N_BOOT = 2000


def _safe_div(x1, x2):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(x2) > 1e-10, np.divide(x1, x2), 1.0)


def _protected_log(x):
    return np.log(np.abs(x) + 1e-12)


def _bounded_pow(x1, x2):
    """Power law with the exponent clipped: the family scaling laws live in."""
    return np.clip(np.abs(x1), 1e-6, None) ** np.clip(x2, -3.0, 3.0)


def _protected_sqrt(x):
    return np.sqrt(np.abs(x))


FUNCTIONS = [
    make_function(function=np.add, name="add", arity=2),
    make_function(function=np.subtract, name="sub", arity=2),
    make_function(function=np.multiply, name="mul", arity=2),
    make_function(function=_safe_div, name="div", arity=2),
    make_function(function=_protected_log, name="log", arity=1),
    make_function(function=_protected_sqrt, name="sqrt", arity=1),
    make_function(function=np.abs, name="abs", arity=1),
    make_function(function=np.negative, name="neg", arity=1),
    make_function(function=_bounded_pow, name="bpow", arity=2),
]


def load_rows(path=CELLS):
    with open(path, encoding="utf-8") as f:
        cells = json.load(f)
    rows = []
    for suite, group in cells.items():
        if suite not in FORWARD_SUITES:
            continue
        for c in group:
            cfg = c["cfg"]
            if cfg not in FAMILY_OF:
                raise KeyError(f"{suite}/{cfg}: not mapped to a family - extend FAMILY_OF "
                               f"instead of dropping the cell")
            if c.get("timesteps") is None:
                continue          # hrl cells were logged without a budget
            rows.append({"suite": suite, "cfg": cfg, "game": c["game"], "seed": c["seed"],
                         "budget": float(c["timesteps"]), "ret": float(c["eval_unseen"]),
                         "family": FAMILY_OF[cfg],
                         "hard": 1.0 if suite in ("hard", "temporal_hard") else 0.0,
                         "memory": 1.0 if cfg in MEMORY_CFGS else 0.0})
    return rows


def design(rows, game_dummies=False):
    """Feature matrices built only from the protocol, never from the measured return."""
    games = sorted({r["game"] for r in rows}) if game_dummies else []
    X = np.array([[math.log10(r["budget"]), r["hard"], r["memory"]]
                  + [1.0 if r["game"] == g else 0.0 for g in games] for r in rows])
    y = np.array([r["ret"] for r in rows], dtype=float)
    groups = np.array([r["seed"] for r in rows])
    return X, y, groups


def cv_predict(model, X, y, groups=None, n_splits=5):
    if groups is not None and len(set(groups)) >= n_splits:
        splitter = GroupKFold(n_splits=min(n_splits, len(set(groups))))
        splits = list(splitter.split(X, y, groups))
    else:
        splits = list(KFold(n_splits=n_splits, shuffle=True, random_state=0).split(X))
    pred = np.zeros_like(y)
    for tr, te in splits:
        m = clone(model)
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def scores(y, pred):
    resid = y - pred
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {"r2": round(1.0 - float(np.sum(resid ** 2)) / ss_tot, 4) if ss_tot > 0 else None,
            "mae": round(float(np.mean(np.abs(resid))), 4),
            "rmse": round(float(np.sqrt(np.mean(resid ** 2))), 4)}


def paired_bootstrap(y, pred_a, pred_b, n_boot=N_BOOT, seed=0):
    """P(model a has the smaller mean squared error), over resampled observations."""
    rng = np.random.default_rng(seed)
    ea, eb = (y - pred_a) ** 2, (y - pred_b) ** 2
    wins = sum(float(ea[i].mean() < eb[i].mean())
               for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot)))
    return round(wins / n_boot, 4)


def sr_model(args, seed=None):
    return SymbolicRegressor(population_size=args.pop, generations=args.gens,
                             tournament_size=20, stopping_criteria=0, p_crossover=0.7,
                             p_subtree_mutation=0.1, p_hoist_mutation=0.01,
                             p_point_mutation=0.1, max_samples=0.9, verbose=0,
                             parsimony_coefficient=args.parsimony,
                             random_state=args.seed if seed is None else seed,
                             function_set=FUNCTIONS, init_depth=(2, 6),
                             init_method="half and half", metric="mean absolute error",
                             const_range=(-1.0, 1.0), n_jobs=1, low_memory=True)


def expression(model, names):
    """gplearn 0.4.2 prints programs as add(X0, mul(0.5, X1)); rename X<i> to labels."""
    if model is None or not hasattr(model, "_program"):
        return None
    text = str(model._program)
    # an index with no label is left visible as X<i>? rather than raising: a feature/name
    # mismatch must show up inside the reported equation, not abort the study
    return re.sub(r"\bX(\d+)\b",
                  lambda m: names[int(m.group(1))] if int(m.group(1)) < len(names)
                  else f"X{m.group(1)}?", text)


def to_latex(expr, names):
    if expr is None:
        return None
    try:
        import sympy
        e = sympy.parse_expr(expr, local_dict={
            "add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b,
            "div": lambda a, b: a / b, "log": sympy.log, "sqrt": sympy.sqrt,
            "abs": sympy.Abs, "neg": lambda a: -a, "bpow": lambda a, b: a ** b,
            **{n: sympy.Symbol(n) for n in names}})
        return sympy.latex(sympy.simplify(e))
    except Exception:
        return None


def fit_summary(X, y, groups, names, args):
    m = sr_model(args)
    pred = cv_predict(m, X, y, groups)
    m.fit(X, y)
    expr = expression(m, names)
    prog = m._program
    return {**scores(y, pred), "expression": expr, "latex": to_latex(expr, names),
            "program_length": int(prog.length_), "program_depth": int(prog.depth_),
            "features": names}, pred


def p1_scaling_laws(rows, args, out):
    """Recover R ~ f(protocol) and compare symbolic search with the assumed linear law.

    Every model is evaluated on both feature sets. Reporting the symbolic fit on a smaller
    feature set than its baselines would make the comparison meaningless, and this is exactly
    the asymmetry that has to be visible in the table.
    """
    proto_names = ["log10N", "hard", "memory"]
    Xp, y, g = design(rows)
    Xg, _, _ = design(rows, game_dummies=True)
    games = sorted({r["game"] for r in rows})
    full_names = proto_names + [f"game_{x}" for x in games]   # sympy cannot parse ":"

    specs = {
        "null_mean": lambda: DummyRegressor(strategy="mean"),
        "linear_in_logN": lambda: LinearRegression(),
        "quadratic_in_logN": lambda: make_pipeline(PolynomialFeatures(2), Ridge(alpha=1.0)),
        "random_forest": lambda: RandomForestRegressor(n_estimators=300, random_state=0,
                                                       n_jobs=1),
    }
    models, preds = {}, {}
    for feat, Xm in (("protocol", Xp), ("protocol+game", Xg)):
        for name, ctor in specs.items():
            key = f"{name}[{feat}]"
            pred = cv_predict(ctor(), Xm, y, g)
            models[key] = dict(scores(y, pred), features=feat, n=len(y))
            preds[key] = pred
        m = sr_model(args)
        pred = cv_predict(m, Xm, y, g)
        m.fit(Xm, y)
        expr = expression(m, full_names if feat == "protocol+game" else proto_names)
        key = f"symbolic[{feat}]"
        models[key] = dict(scores(y, pred), features=feat, n=len(y), expression=expr,
                           latex=to_latex(expr, full_names if feat == "protocol+game"
                                          else proto_names),
                           program_length=int(m._program.length_),
                           program_depth=int(m._program.depth_))
        preds[key] = pred

    out["P1_pooled"] = {
        "rows": len(y), "games": games, "models": models,
        "comparisons": {
            "P_symbolic_beats_linear_same_features": {
                feat: paired_bootstrap(y, preds[f"symbolic[{feat}]"],
                                       preds[f"linear_in_logN[{feat}]"])
                for feat in ("protocol", "protocol+game")},
            "P_symbolic_plus_game_beats_random_forest_plus_game":
                paired_bootstrap(y, preds["symbolic[protocol+game]"],
                                 preds["random_forest[protocol+game]"]),
            "note": "probability from a paired bootstrap of the squared error over the same "
                    "rows; 0.5 means the two laws are indistinguishable on these measurements",
        },
    }

    per_game = {}
    for game in sorted({r["game"] for r in rows}):
        sub = [r for r in rows if r["game"] == game]
        budgets = sorted({round(r["budget"]) for r in sub})
        if len(sub) < 20 or len(budgets) < 2:
            per_game[game] = {"skipped": f"{len(sub)} rows over {len(budgets)} budgets"}
            continue
        Xs, ys, gs = design(sub)
        lin = LinearRegression().fit(Xs[:, :1], ys)   # slope is read off the budget column
        lin_pred = cv_predict(LinearRegression(), Xs, ys, gs)
        sym, sym_pred = fit_summary(Xs, ys, gs, proto_names, args)
        per_game[game] = {
            "rows": len(sub), "budgets": budgets,
            "linear_in_logN": dict(scores(ys, lin_pred),
                                   slope_per_decade=round(float(lin.coef_[0]), 4),
                                   intercept=round(float(lin.intercept_), 4)),
                    "symbolic": sym,
            "P_symbolic_beats_linear": paired_bootstrap(ys, sym_pred, lin_pred),
        }
    out["P1_per_game"] = per_game


def wilson_ci(hits, n, z=1.96):
    """Binomial interval for an accuracy, reported instead of a bootstrap because the
    per-split predictions are not independent resamples of a population."""
    if n == 0:
        return None
    ph = hits / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    hw = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - hw), 4), round(min(1.0, c + hw), 4)]


CLASSIFIERS = {
    "nearest_centroid": NearestCentroid,
    "gaussian_nb": GaussianNB,
    "random_forest": lambda: RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=1),
}


def p2_hidden_family(rows, args, out):
    """Recover the generating architecture family from the measured return alone.

    This is the inverse problem stated over a hidden discrete parameter. Within one
    (game, suite, budget) stratum every configuration shares the protocol, so the only
    observable that distinguishes them is the return itself. The question the benchmark
    implicitly asks - "does this number tell me which architecture produced it?" - is
    answered by accuracy against the 1/F chance level, as a function of how many seeds
    the reference profile and the measurement each average over.
    """
    from itertools import combinations

    strata = {}
    for r in rows:
        strata.setdefault((r["game"], r["suite"], round(r["budget"])), []).append(r)

    study, pooled = {}, {}
    for key in sorted(strata, key=lambda k: (k[0], k[1])):
        game, suite, budget = key
        cells = strata[key]
        seeds = sorted({c["seed"] for c in cells})
        grid = {}
        for c in cells:
            grid.setdefault(c["family"], {})[c["seed"]] = c["ret"]
        fams = sorted(f for f, d in grid.items() if len(d) == len(seeds))
        if len(fams) < 4 or len(seeds) < 5:
            continue
        chance = 1.0 / len(fams)
        study[f"{game}|{suite}|{int(budget)}"] = {
            "families": len(fams), "seeds": len(seeds), "chance_level": round(chance, 4),
            "accuracy": {}}
        for j in range(1, 4):
            for k in range(1, len(seeds) - j + 1):
                hits = {m: 0 for m in list(CLASSIFIERS) + ["lowest_mean_heuristic"]}
                n = 0
                for meas_seeds in combinations(seeds, j):
                    rest = [s for s in seeds if s not in meas_seeds]
                    for ref in combinations(rest, k):
                        means = {f: float(np.mean([grid[f][s] for s in ref])) for f in fams}
                        ordered = sorted(fams, key=lambda f: means[f])
                        Xtr = np.array([[means[f]] for f in fams])
                        ytr = np.array(fams)
                        fitted = {}
                        for name, ctor in CLASSIFIERS.items():
                            try:
                                fitted[name] = clone(ctor()).fit(Xtr, ytr)
                            except Exception:
                                fitted[name] = None
                        for f_true in fams:
                            meas = np.array([[float(np.mean([grid[f_true][s]
                                                             for s in meas_seeds]))]])
                            n += 1
                            hits["lowest_mean_heuristic"] += int(f_true == ordered[0])
                            for name, mdl in fitted.items():
                                if mdl is not None and mdl.predict(meas)[0] == f_true:
                                    hits[name] += 1
                cell = {}
                for name, h in hits.items():
                    acc = h / n if n else 0.0
                    cell[name] = {"accuracy": round(acc, 4), "hits": h, "n": n,
                                  "ci95_wilson": wilson_ci(h, n),
                                  "above_chance": bool(n and acc > chance)}
                    pooled.setdefault(f"k_{k}_j_{j}", {"hit": 0, "n": 0})
                    pooled[f"k_{k}_j_{j}"]["hit"] += h
                    pooled[f"k_{k}_j_{j}"]["n"] += n
                study[f"{game}|{suite}|{int(budget)}"]["accuracy"][f"ref_{k}_measured_{j}"] = cell

    out["P2_hidden_family_identification"] = {
        "design": "within one (game, suite, budget) stratum the protocol is identical for "
                  "every configuration, so the observable left is the measured return. The "
                  "hidden parameter is the architecture family; recovery is scored against the "
                  "1/F chance level. 'ref_k' averages k seeds to build each family's reference "
                  "profile, 'measured_j' averages j held-out seeds into the query.",
        "strata": study,
        "pooled_across_strata": {
            kk: {"nearest_centroid_accuracy": round(vv["hit"] / vv["n"], 4),
                 "n_predictions": vv["n"], "ci95_wilson": wilson_ci(vv["hit"], vv["n"])}
            for kk, vv in sorted(pooled.items())},
        "caveat": "the (reference, measurement) splits of one stratum reuse seeds, so the "
                  "predictions are not independent draws; the Wilson interval is therefore "
                  "optimistic and is reported as a spread indicator, not as a confirmatory test",
    }



def p3_inversion(rows, out):
    """Solve the recovered law for the budget a target return implies, then check it."""
    per_game = {}
    for game in sorted({r["game"] for r in rows}):
        sub = [r for r in rows if r["game"] == game]
        budgets = sorted({round(r["budget"]) for r in sub})
        if len(budgets) < 3:
            per_game[game] = {"skipped": "needs >=3 distinct budgets"}
            continue
        X = np.array([[math.log10(r["budget"])] for r in sub])
        y = np.array([r["ret"] for r in sub])
        lin = LinearRegression().fit(X, y)
        coef, icpt = float(lin.coef_[0]), float(lin.intercept_)
        target = float(y.max())
        best = max(sub, key=lambda r: r["ret"])
        if abs(coef) < 1e-12:
            per_game[game] = {"skipped": "flat law, budget not identifiable"}
            continue
        implied = 10 ** ((target - icpt) / coef)
        lo, hi = budgets[0], budgets[-1]
        per_game[game] = {
            "target_return": round(target, 3),
            "budget_implied_by_the_law": int(round(implied)),
            "budget_of_the_cell_that_achieved_it": int(best["budget"]),
            "ratio_implied_over_actual": round(implied / best["budget"], 3),
            "budgets_available": budgets,
            # an inversion that lands outside the measured span is an extrapolation of the
            # fitted line, not a prediction the data supports
            "implied_within_measured_support": bool(lo <= implied <= hi),
            "measured_budget_span": round(hi / lo, 2),
            "identifiable": bool(coef > 1e-12 and lo <= implied <= hi),
        }
    out["P3_budget_inversion"] = per_game


def pareto_curve(rows, args):
    """Program size vs held-out error: the standard presentation of an SR result."""
    X, y, g = design(rows)
    curve = []
    for par in [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]:
        a = argparse.Namespace(**vars(args))
        a.parsimony = max(par, 1e-6)
        m = sr_model(a, seed=args.seed)
        pred = cv_predict(m, X, y, g)
        m.fit(X, y)
        curve.append({"parsimony": par, "program_length": int(m._program.length_),
                      "program_depth": int(m._program.depth_), **scores(y, pred)})
    return curve


def make_figures(rows, out, args):
    """Three panels, each drawn only from numbers already written into `out`."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []

    # 1. complexity vs held-out accuracy: the Pareto front of the search
    par = out.get("pareto_program_size_vs_error")
    if par:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot([c["program_length"] for c in par], [c["r2"] for c in par], "o-", label="symbolic")
        lin = out["P1_pooled"]["models"]["linear_in_logN[protocol]"]["r2"]
        ax.axhline(lin, ls="--", color="crimson",
                   label=f"linear in log10N (R2={lin})")
        ax.set_xlabel("program length (nodes)")
        ax.set_ylabel("CV $R^2$ (grouped by seed)")
        ax.set_title("Symbolic regression Pareto front - pooled benchmark cells")
        ax.legend()
        fig.tight_layout()
        f = os.path.join(RESULTS, "symbolic_regression_pareto.png")
        fig.savefig(f, dpi=150); plt.close(fig); written.append(f)

    # 2. the plateau, per game, with the fitted slope per decade of budget
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for game, v in sorted(out["P1_per_game"].items()):
        if "skipped" in v:
            continue
        sub = [r for r in rows if r["game"] == game]
        ax.scatter([r["budget"] for r in sub], [r["ret"] for r in sub], s=14, alpha=0.6,
                   label=f"{game} (slope {v['linear_in_logN']['slope_per_decade']:+.2f}/decade)")
    ax.set_xscale("log")
    ax.set_xlabel("environment steps (log)")
    ax.set_ylabel("mean return on unseen levels")
    ax.set_title("Measured scaling across budgets - the slope is the plateau")
    ax.legend(fontsize=7)
    fig.tight_layout()
    f = os.path.join(RESULTS, "symbolic_regression_scaling.png")
    fig.savefig(f, dpi=150); plt.close(fig); written.append(f)

    # 3. identifiability of the hidden family vs the number of seeds observed
    p2 = out["P2_hidden_family_identification"]["pooled_across_strata"]
    by_j = {}
    for key, v in p2.items():
        k, j = key.split("_")[1], key.split("_")[3]
        by_j.setdefault(int(j), []).append((int(k), v["nearest_centroid_accuracy"],
                                           v["n_predictions"]))
    chance = np.mean([s["chance_level"]
                      for s in out["P2_hidden_family_identification"]["strata"].values()])
    fig, ax = plt.subplots(figsize=(8, 5))
    for j, pts in sorted(by_j.items()):
        pts.sort()
        ax.plot([k for k, _a, _n in pts], [a for _k, a, _n in pts], "o-",
                label=f"measurement averaged over {j} seed(s)")
    ax.axhline(chance, ls="--", color="black", label=f"chance 1/F = {chance:.3f}")
    ax.set_xlabel("reference seeds averaged per family (k)")
    ax.set_ylabel("accuracy of recovering the architecture family")
    ax.set_title("Inverse problem: the measured return barely identifies its architecture")
    ax.legend(fontsize=8)
    fig.tight_layout()
    f = os.path.join(RESULTS, "symbolic_regression_identifiability.png")
    fig.savefig(f, dpi=150); plt.close(fig); written.append(f)
    return written


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pop", type=int, default=1000)
    ap.add_argument("--gens", type=int, default=40)
    ap.add_argument("--parsimony", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="small search, for CI")
    ap.add_argument("--no-pareto", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--out", default=os.path.join(RESULTS, "symbolic_regression.json"))
    args = ap.parse_args()
    if args.quick:
        args.pop, args.gens = 200, 8

    rows = load_rows()
    import gplearn
    import sklearn
    budgets = [r["budget"] for r in rows]
    modal = max(set(budgets), key=budgets.count)
    out = {"_provenance": {
        "produced_by": "symbolic_regression.py",
        "data_source": os.path.relpath(CELLS, BASE).replace(os.sep, "/"),
        "rows_used": len(rows),
        "budget_support": {
            "distinct_budgets": sorted({int(b) for b in budgets}),
            "cells_at_the_modal_budget": budgets.count(modal),
            "share_at_modal_budget": round(budgets.count(modal) / len(budgets), 3),
            "note": "the grid was a configuration sweep, not a budget sweep: most cells share "
                    "one budget, so P1 measures how much return varies *across configurations* "
                    "at a near-constant budget, and any P1 slope on log10N is fitted over a "
                    "narrow span rather than tested over decades",
        },
        "rows_excluded": "hrl cells (logged without a budget) and the marl suite (win-rate "
                         "keyed, different measurement)",
        "engine": f"gplearn {gplearn.__version__}, scikit-learn {sklearn.__version__}",
        "cv": "GroupKFold by seed - a seed never appears in both fit and test",
        "leakage_guard": "gen_gap is never a feature; it is train_minus_unseen and would "
                         "contain the target",
        "search": {"population": args.pop, "generations": args.gens,
                   "parsimony": args.parsimony, "random_state": args.seed,
                   "function_set": ["add", "sub", "mul", "div", "log", "sqrt", "abs", "neg",
                                    "bpow"],
                   "guards": "div returns 1 unless |x2|>1e-10; log is log(|x|+1e-12); sqrt is "
                             "sqrt(|x|); bpow clips its exponent to [-3,3]. gplearn rejects "
                             "unary functions that are not closed over negatives, so every "
                             "partial function here is protected by construction."},
        "bootstraps": N_BOOT},
        "rows": len(rows)}

    print(f"rows: {len(rows)} cells from {os.path.basename(CELLS)}")
    p1_scaling_laws(rows, args, out)
    p1 = out["P1_pooled"]
    print("P1 pooled CV R2 (features in brackets):")
    for k, v in p1["models"].items():
        extra = f" len={v['program_length']} depth={v['program_depth']}" if "program_length" in v else ""
        print(f"   {k:42s} R2={v['r2']}  MAE={v['mae']}{extra}")
    print("   P(symbolic > linear, same features):",
          p1["comparisons"]["P_symbolic_beats_linear_same_features"])
    p2_hidden_family(rows, args, out)
    p2 = out["P2_hidden_family_identification"]
    print("P2 strata:", len(p2["strata"]))
    for kk, vv in sorted(p2["pooled_across_strata"].items()):
        print(f"   {kk:12s} nearest_centroid={vv['nearest_centroid_accuracy']:.3f} "
              f"(n={vv['n_predictions']})")
    p3_inversion(rows, out)
    print("P3 identifiable games:",
          sum(1 for v in out["P3_budget_inversion"].values() if v.get("identifiable")))
    if not args.no_pareto:
        out["pareto_program_size_vs_error"] = pareto_curve(rows, args)
        print("Pareto: " + ", ".join(f"{c['program_length']}n/{c['r2']}"
                                     for c in out["pareto_program_size_vs_error"]))

    if not args.no_figures:
        out["figures"] = make_figures(rows, out, args)
        print("figures: " + ", ".join(os.path.basename(f) for f in out["figures"]))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
