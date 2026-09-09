"""Analise final da grade: rankings + stats vs conclusoes do estudo (§§3-12).

Le jax_port/results_grade/*/*.json (eval_unseen 100 eps), agrupa por
(cfg, jogo) nas seeds, aplica stats.rank_cells e compara com as
ordenacoes publicadas. Saida: jax_port/analysis_full.json + tabelas.
Uso: python jax_port/analyze_grade.py
"""

import glob
import json
import os

from jax_port.stats import auc_norm, rank_cells

GRADE = "jax_port/results_grade"
OUT = "jax_port/analysis_full.json"


def load_cells(suite):
    cells = []
    # nota: a suite temporal rodou com CWD=jax_port -> sai tambem em
    # jax_port/jax_port/results_grade/temporal/ (movida p/ o lugar certo
    # ao final; o glob duplo blinda a analise em ambos os casos).
    paths = [os.path.join(GRADE, suite, "*.json"),
             os.path.join("jax_port", GRADE, suite, "*.json")]
    files = sorted({f for p in paths for f in glob.glob(p)})
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        ev = (d.get("eval_unseen") or {}).get("mean")
        if ev is None:
            continue
        # cfg vem do nome do arquivo (gêmeos ae/recon/classic partilham
        # extrator mas têm seeds/draws independentes por célula).
        d["_cfg"] = os.path.basename(f).split("__")[0]
        cells.append(d)
    return cells


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


def main():
    rep = {}
    # main: por jogo + global (media das medias, como §3.7/3.12)
    g = group(load_cells("main"))
    games = sorted(set(game for _, game in g))
    rep["main"] = {game: rank_cells(
        {cfg: vals for (cfg, gm), vals in g.items() if gm == game})
        for game in games}
    cfgs = sorted(set(cfg for cfg, _ in g))
    gs = group_seeds(load_cells("main"))
    seeds = sorted(set(s for v in gs.values() for s in v))
    rep["main"]["global"] = rank_cells({
        cfg: [sum(gs[(cfg, gm)][s] for gm in games) / len(games) for s in seeds]
        for cfg in cfgs if all(s in gs.get((cfg, gm), {}) for gm in games
                               for s in seeds)})
    for suite in ("exploration", "algo", "hrl", "hard", "pilot", "spr",
                  "gnn", "aux", "temporal", "temporal_hard"):
        gg = group(load_cells(suite))
        games = sorted(set(game for _, game in gg))
        rep[suite] = {game: rank_cells(
            {cfg: vals for (cfg, gm), vals in gg.items() if gm == game})
            for game in games}
    # temporal_hard: 100k-easy vs 500k-hard por (cfg, jogo) — delta p/ ver
    # se o transformer sai do fundo com mais budget/dificuldade.
    t_easy = group(load_cells("temporal"))
    t_hard = group(load_cells("temporal_hard"))
    rep["temporal_delta"] = {}
    for (cfg, game) in sorted(set(t_easy) | set(t_hard)):
        e = t_easy.get((cfg, game))
        h = t_hard.get((cfg, game))
        rep["temporal_delta"][f"{cfg}__{game}"] = {
            "easy_100k": sorted(e) if e else None,
            "hard_500k": sorted(h) if h else None}
    # budget: curvas por (cfg, jogo)
    curves = {}
    for d in load_cells("budget"):
        key = (d["_cfg"], d["game"])
        curves.setdefault(key, []).append((d["timesteps"], d["eval_unseen"]["mean"]))
    rep["budget"] = {f"{c}/{g}": sorted(v) for (c, g), v in curves.items()}
    # marl: win-rate por (algo, mapa, budget) nas seeds
    # (chave "map" nao "game"; 1M feedforward vs 10M recorrente separados)
    mg = {}
    for f in sorted(glob.glob(os.path.join(GRADE, "marl", "*.json"))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
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
    from jax_port.stats import mean_ci
    # gen_gap: quem generaliza (menor gap = melhor). rank_cells ordena
    # decrescente, entao entramos com -gap para o menor gap liderar.
    gg = {}
    for d in load_cells("main"):
        if d.get("gen_gap") is None:
            continue
        gg.setdefault((d["_cfg"], d["game"]), []).append(d["gen_gap"])
    rep["gen_gap"] = {game: rank_cells(
        {cfg: [-v for v in vals] for (cfg, gm), vals in gg.items()
         if gm == game})
        for game in sorted(set(gm for _, gm in gg))}
    # inverte o sinal de volta p/ presentaao nos rankings (mean = gap real)
    for game in rep["gen_gap"]:
        for row in rep["gen_gap"][game]["ranking"]:
            row["mean"] = -row["mean"]
            row["ci95"] = [-row["ci95"][1], -row["ci95"][0]]
    ggs = {}
    for d in load_cells("main"):
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
    # AUC por celula com curva
    n_auc = 0
    for suite in ("main", "exploration", "algo", "hrl", "budget", "hard",
                  "pilot", "spr", "gnn", "aux", "temporal", "temporal_hard"):
        for pat in (os.path.join(GRADE, suite, "*.json"),
                    os.path.join("jax_port", GRADE, suite, "*.json")):
            for f in glob.glob(pat):
                d = json.load(open(f))
                if d.get("curve"):
                    n_auc += 1
    rep["meta"] = {"cells_with_curve": n_auc}
    with open(OUT, "w") as fh:
        json.dump(rep, fh, indent=1)
    # tabelas resumidas
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
        print("== gen_gap (menor = generaliza melhor) ==")
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
    if "temporal_hard" in rep:
        print("== temporal_hard ==")
        for game, r in rep["temporal_hard"].items():
            top = [(x["cell"], round(x["mean"], 2)) for x in r["ranking"][:5]]
            print(f"  {game}: {top}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
