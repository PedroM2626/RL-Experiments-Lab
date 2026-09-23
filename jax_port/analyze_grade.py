"""Final grid analysis: rankings + stats vs conclusions of the study (§§3-12).

Reads <jax_port>/results_grade/<suite>/*.json (eval_unseen 100 eps), groups by
(cfg, game) across seeds, applies stats.rank_cells and compares with published
orderings. Output: jax_port/analysis_full.json + tables, plus an optional
cells_summary.json holding the per-cell values that the rankings were built from.

Usage: python jax_port/analyze_grade.py [--results_root DIR] [--out FILE]
                                         [--summary FILE] [--force]

Paths are resolved from this file, not the working directory. Cells that cannot be
read are reported instead of being skipped in silence, and the run refuses to
overwrite a richer analysis_full.json with a thinner one (the raw grid is not in
git, so a fresh clone has nothing to load).
"""

import argparse
import glob
import json
import os

from jax_port.stats import auc_norm, rank_cells

BASE = os.path.dirname(os.path.abspath(__file__))
GRADE_DEFAULT = os.path.join(BASE, "results_grade")
OUT_DEFAULT = os.path.join(BASE, "analysis_full.json")
SUMMARY_DEFAULT = os.path.join(BASE, "cells_summary.json")

RANKING_SUITES = ("main", "exploration", "algo", "hrl", "hard", "pilot", "spr",
                  "gnn", "aux", "temporal", "temporal_hard")
AUC_SUITES = ("main", "exploration", "algo", "hrl", "budget", "hard", "pilot",
              "spr", "gnn", "aux", "temporal", "temporal_hard")


class Loader:
    """Loads grid cells once per suite and keeps a record of what was dropped."""

    def __init__(self, grade_dir):
        self.grade_dir = grade_dir
        self.cache = {}
        self.unreadable = []
        self.no_eval = []
        self.missing_suites = []

    def cells_for(self, suite):
        if suite in self.cache:
            return self.cache[suite]
        suite_dir = os.path.join(self.grade_dir, suite)
        files = sorted(glob.glob(os.path.join(suite_dir, "*.json")))
        if not files:
            self.missing_suites.append(suite)
        out = []
        for f in files:
            try:
                with open(f, encoding="utf-8") as fh:
                    d = json.load(fh)
            except (OSError, json.JSONDecodeError) as e:
                self.unreadable.append((os.path.relpath(f, self.grade_dir), str(e)))
                continue
            ev = (d.get("eval_unseen") or {}).get("mean")
            if ev is None:
                self.no_eval.append(os.path.relpath(f, self.grade_dir))
                continue
            # cfg comes from filename (twin ae/recon/classic share extractor but
            # have independent seeds/draws per cell).
            d["_cfg"] = os.path.basename(f).split("__")[0]
            out.append(d)
        self.cache[suite] = out
        return out

    def marl_cells(self):
        """MARL cells are win-rate keyed (algo, map, budget) rather than eval_unseen."""
        marl_dir = os.path.join(self.grade_dir, "marl")
        files = sorted(glob.glob(os.path.join(marl_dir, "*.json")))
        if not files:
            self.missing_suites.append("marl")
        out = []
        for f in files:
            try:
                with open(f, encoding="utf-8") as fh:
                    d = json.load(fh)
            except (OSError, json.JSONDecodeError) as e:
                self.unreadable.append((os.path.relpath(f, self.grade_dir), str(e)))
                continue
            if "eval" not in d or "map" not in d:
                self.no_eval.append(os.path.relpath(f, self.grade_dir))
                continue
            out.append(d)
        return out


class SummaryLoader:
    """Same read interface, fed by cells_summary.json instead of the raw grid.

    The raw results_grade/ output is git-ignored, so this is what lets a fresh clone
    re-derive analysis_full.json from data that is actually in the repository.
    """

    def __init__(self, path):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        self.grade_dir = os.path.dirname(os.path.abspath(path))
        self.unreadable = []
        self.no_eval = []
        self.missing_suites = []
        self.cache = {}
        self._marl = []
        for suite, rows in data.items():
            if suite == "marl":
                self._marl = [{"algo": r["algo"], "map": r["map"], "seed": r.get("seed"),
                               "timesteps": r.get("timesteps"),
                               "eval": {"winrate": r.get("winrate", 0.0)}} for r in rows]
                continue
            self.cache[suite] = [
                {"_cfg": r["cfg"], "game": r["game"], "seed": r["seed"],
                 "eval_unseen": {"mean": r["eval_unseen"]}, "gen_gap": r.get("gen_gap"),
                 "timesteps": r.get("timesteps"),
                 "curve": [None] * int(r.get("curve_points") or 0)}
                for r in rows]

    def cells_for(self, suite):
        return self.cache.get(suite, [])

    def marl_cells(self):
        return self._marl


def group(cells):
    g = {}
    for d in cells:
        g.setdefault((d["_cfg"], d["game"]), []).append(
            d["eval_unseen"]["mean"])
    return g


def group_seeds(cells):
    g = {}
    for d in cells:
        g.setdefault((d["_cfg"], d["game"]),
                     {})[d["seed"]] = d["eval_unseen"]["mean"]
    return g


def _ranking_rows(rep, suite):
    node = rep.get(suite)
    if not isinstance(node, dict):
        return 0
    return sum(len(r.get("ranking", [])) for r in node.values() if isinstance(r, dict))


def guard_no_data_loss(out_path, rep, force):
    """The raw grid lives outside git; never overwrite a full analysis with a partial one."""
    if force or not os.path.exists(out_path):
        return
    with open(out_path, encoding="utf-8") as fh:
        committed = json.load(fh)
    thinner = {s: (_ranking_rows(committed, s), _ranking_rows(rep, s))
               for s in committed
               if _ranking_rows(committed, s) > _ranking_rows(rep, s)}
    if thinner:
        print("ERROR: refusing to overwrite the committed analysis — it reports more cells "
              "than this run could load:")
        for suite, (old, new) in sorted(thinner.items()):
            print(f"  {suite:16s} committed={old:4d} this run={new:4d}")
        print(f"The raw grid is not part of the repository (results_grade/ is git-ignored). "
              f"Point --results_root at a directory of cell JSONs, or pass --force to accept "
              f"the reduced analysis.")
        raise SystemExit(1)


def build_report(loader):
    rep = {}
    # main: per game + global (mean of means, as in §3.7/3.12)
    g = group(loader.cells_for("main"))
    games = sorted(set(game for _, game in g))
    rep["main"] = {game: rank_cells(
        {cfg: vals for (cfg, gm), vals in g.items() if gm == game})
        for game in games}
    cfgs = sorted(set(cfg for cfg, _ in g))
    gs = group_seeds(loader.cells_for("main"))
    seeds = sorted(set(s for v in gs.values() for s in v))
    rep["main"]["global"] = rank_cells({
        cfg: [sum(gs[(cfg, gm)][s] for gm in games) / len(games) for s in seeds]
        for cfg in cfgs if all(s in gs.get((cfg, gm), {}) for gm in games
                               for s in seeds)})
    for suite in RANKING_SUITES:
        if suite == "main":
            continue
        gg = group(loader.cells_for(suite))
        games = sorted(set(game for _, game in gg))
        rep[suite] = {game: rank_cells(
            {cfg: vals for (cfg, gm), vals in gg.items() if gm == game})
            for game in games}
    # temporal_hard: 100k-easy vs 500k-hard per (cfg, game) — delta to see
    # if transformer recovers with more budget/difficulty.
    t_easy = group(loader.cells_for("temporal"))
    t_hard = group(loader.cells_for("temporal_hard"))
    rep["temporal_delta"] = {}
    for (cfg, game) in sorted(set(t_easy) | set(t_hard)):
        e = t_easy.get((cfg, game))
        h = t_hard.get((cfg, game))
        rep["temporal_delta"][f"{cfg}__{game}"] = {
            "easy_100k": sorted(e) if e else None,
            "hard_500k": sorted(h) if h else None}
    # budget: curves per (cfg, game)
    curves = {}
    for d in loader.cells_for("budget"):
        key = (d["_cfg"], d["game"])
        curves.setdefault(key, []).append((d["timesteps"], d["eval_unseen"]["mean"]))
    rep["budget"] = {f"{c}/{g}": sorted(v) for (c, g), v in curves.items()}
    # marl: win-rate per (algo, map, budget) across seeds
    # (key "map" not "game"; 1M feedforward vs 10M recurrent kept separate)
    mg = {}
    for d in loader.marl_cells():
        budget = 10000000 if d.get("timesteps", 0) >= 10000000 else 1000000
        mg.setdefault((d["algo"], d["map"], budget), []).append(
            (d.get("eval") or {}).get("winrate", 0.0))
    rep["marl"] = {}
    for k3 in sorted(mg):
        algo, mp, budget = k3
        vals = mg[k3]
        key = f"{algo}__{mp}__{budget // 1000000}M"
        rep["marl"][key] = {
            "winrate_mean": round(sum(vals) / len(vals), 3),
            "winrate_by_seed": vals, "n_seeds": len(vals),
            "solved": round(sum(vals) / len(vals), 3) >= 0.5}
    # gen_gap: generalization (lower gap = better). rank_cells sorts
    # descending, so we pass -gap to let the lowest gap lead.
    gg = {}
    for d in loader.cells_for("main"):
        if d.get("gen_gap") is None:
            continue
        gg.setdefault((d["_cfg"], d["game"]), []).append(d["gen_gap"])
    rep["gen_gap"] = {game: rank_cells(
        {cfg: [-v for v in vals] for (cfg, gm), vals in gg.items()
         if gm == game})
        for game in sorted(set(gm for _, gm in gg))}
    # invert sign back for presentation in rankings (mean = real gap)
    for game in rep["gen_gap"]:
        for row in rep["gen_gap"][game]["ranking"]:
            row["mean"] = -row["mean"]
            row["ci95"] = [-row["ci95"][1], -row["ci95"][0]]
    ggs = {}
    for d in loader.cells_for("main"):
        if d.get("gen_gap") is None:
            continue
        ggs.setdefault((d["_cfg"], d["game"]), {})[d["seed"]] = d["gen_gap"]
    games_g = sorted(set(gm for _, gm in ggs))
    seeds_g = sorted(set(v_ for v in ggs.values() for v_ in v))
    rep["gen_gap"]["global"] = rank_cells({
        cfg: [-sum(ggs[(cfg, gm)][s] for gm in games_g) / len(games_g)
              for s in seeds_g]
        for cfg in sorted(set(c for c, _ in ggs))
        if all(s in ggs.get((cfg, gm), {}) for gm in games_g
               for s in seeds_g)})
    for row in rep["gen_gap"]["global"]["ranking"]:
        row["mean"] = -row["mean"]
        row["ci95"] = [-row["ci95"][1], -row["ci95"][0]]
    # how many cells carry a learning curve (AUC is computable for those)
    n_auc = sum(1 for suite in AUC_SUITES for d in loader.cells_for(suite) if d.get("curve"))
    return rep, n_auc


def cell_summary(loader):
    """Per-cell values behind the rankings, so the analysis is checkable without the raw grid."""
    summary = {}
    for suite, cells in loader.cache.items():
        rows = summary.setdefault(suite, [])
        for d in cells:
            rows.append({
                "cfg": d["_cfg"], "game": d.get("game"), "seed": d.get("seed"),
                "eval_unseen": d["eval_unseen"]["mean"],
                "gen_gap": d.get("gen_gap"),
                "timesteps": d.get("timesteps"),
                "curve_points": len(d.get("curve") or []),
            })
    summary["marl"] = [{
        "algo": d["algo"], "map": d["map"], "seed": d.get("seed"),
        "timesteps": d.get("timesteps"),
        "winrate": (d.get("eval") or {}).get("winrate", 0.0),
    } for d in loader.marl_cells()]
    return summary


def print_tables(rep):
    for suite in ("main", "exploration", "algo", "hrl"):
        print(f"== {suite} ==")
        for game, r in rep[suite].items():
            if game == "global":
                continue
            top = [(x["cell"], round(x["mean"], 2)) for x in r["ranking"][:4]]
            print(f"  {game}: {top}")
    g = rep["main"]["global"]["ranking"]
    print("GLOBAL:", [(x["cell"], round(x["mean"], 2)) for x in g[:6]])
    print("== marl ==")
    for k, v in rep["marl"].items():
        print(f"  {k}: winrate={v['winrate_mean']} n={v['n_seeds']} solved={v['solved']}")
    if "gen_gap" in rep:
        print("== gen_gap (lower = generalizes better) ==")
        for game, r in rep["gen_gap"].items():
            if game == "global":
                continue
            top = [(x["cell"], round(x["mean"], 2)) for x in r["ranking"][:4]]
            print(f"  {game}: {top}")
        print("  GLOBAL:", [(x["cell"], round(x["mean"], 2))
                           for x in rep["gen_gap"]["global"]["ranking"][:6]])
    if "temporal" in rep:
        print("== temporal ==")
        for game, r in rep["temporal"].items():
            top = [(x["cell"], round(x["mean"], 2)) for x in r["ranking"][:5]]
            print(f"  {game}: {top}")
    if rep.get("temporal_hard"):
        print("== temporal_hard ==")
        for game, r in rep["temporal_hard"].items():
            top = [(x["cell"], round(x["mean"], 2)) for x in r["ranking"][:5]]
            print(f"  {game}: {top}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results_root", default=GRADE_DEFAULT,
                        help="directory holding <suite>/*.json grid cells")
    parser.add_argument("--out", default=OUT_DEFAULT)
    parser.add_argument("--summary", default=SUMMARY_DEFAULT,
                        help="per-cell summary JSON to write (use '-' to skip)")
    parser.add_argument("--from_summary", default=None,
                        help="rebuild the analysis from a cells_summary.json instead of the raw grid")
    parser.add_argument("--force", action="store_true",
                        help="allow overwriting a richer analysis_full.json")
    args = parser.parse_args()

    if args.from_summary:
        if not os.path.exists(args.from_summary):
            raise SystemExit(f"ERROR: --from_summary not found: {args.from_summary}")
        loader = SummaryLoader(args.from_summary)
        source = args.from_summary
    else:
        if not os.path.isdir(args.results_root):
            raise SystemExit(f"ERROR: --results_root does not exist: {args.results_root}\n"
                             f"The grid outputs are git-ignored; restore them, point this flag "
                             f"at the cell directories, or use --from_summary.")
        loader = Loader(args.results_root)
        source = args.results_root
    rep, n_auc = build_report(loader)

    n_marl = len(loader.marl_cells())
    total = sum(len(v) for v in loader.cache.values()) + n_marl
    if loader.missing_suites:
        print(f"WARNING: no cell files found for: {', '.join(sorted(set(loader.missing_suites)))}")
    if loader.unreadable:
        print(f"WARNING: {len(loader.unreadable)} unreadable cell file(s):")
        for name, err in loader.unreadable[:10]:
            print(f"  {name}: {err}")
    if loader.no_eval:
        print(f"NOTE: {len(loader.no_eval)} cell(s) without eval_unseen.mean were excluded "
              f"(first 10): {loader.no_eval[:10]}")
    if total == 0:
        raise SystemExit(f"ERROR: 0 cells loaded from {source} — nothing to analyse.")

    guard_no_data_loss(args.out, rep, args.force)
    rep["meta"] = {"cells_with_curve": n_auc, "cells_loaded": total, "marl_cells": n_marl,
                   "suites_loaded": {s: len(v) for s, v in sorted(loader.cache.items())}}

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
    if args.summary != "-" and args.summary != args.from_summary:
        with open(args.summary, "w", encoding="utf-8") as fh:
            json.dump(cell_summary(loader), fh, indent=1, sort_keys=True)
        print(f"Saved per-cell summary: {args.summary}")
    print(f"Loaded {total} cells from {source} -> {args.out}")
    print_tables(rep)


if __name__ == "__main__":
    main()
