"""Port grid runner — Study suites, sequential, resume-safe.

Suites (--suite, repeatable and combinable):
  main        : 16 configs x games (bossfight/starpilot/dodgeball)
  exploration : ppo/icm/rnd/ngu x maze/heist
  algo        : ppo/a2c/dqn/qrdqn x starpilot/dodgeball/bossfight (+--lr-sens)
  hrl         : flat/skip4/hrl/hrl_learned x jumper/plunder (frame budget)
  budget      : resnet18+mlp x starpilot/dodgeball (honors given --timesteps)
Usage (see jax_port/README.md for the venv setup):
    python -m jax_port.run_grade --suite main --games bossfight --seeds 42 \
      --timesteps 100000 --eval-full
Cell outputs default to <this package>/results_grade/, independent of the
working directory.
Cell: {cfg}__{game}__seed{s}__{t}k.json; skips completed cells (resume).
--eval-full => 100 stoch + 100 det + 15 train (definitive protocol).
"""

import argparse
import json
import os
import time
import types

BASE = os.path.dirname(os.path.abspath(__file__))

MAIN_CONFIGS = ["classic", "cbam", "spatial", "mlp", "aug_crop", "aug_color",
                "aug_noise", "impala", "impoola", "lstm_attention", "vit",
                "resnet18", "vae", "ae", "recon", "contrastive"]
MAIN_GAMES = ["bossfight", "starpilot", "dodgeball"]
EXPLORE_CONFIGS = ["ppo", "icm", "rnd", "ngu"]
EXPLORE_GAMES = ["maze", "heist"]
ALGO_CONFIGS = ["ppo", "a2c", "dqn", "qrdqn"]
ALGO_GAMES = ["starpilot", "dodgeball", "bossfight"]
HRL_ARMS = ["flat", "skip4", "hrl", "hrl_learned"]
HRL_GAMES = ["jumper", "plunder"]
BUDGET_CONFIGS = ["resnet18", "mlp"]
BUDGET_GAMES = ["starpilot", "dodgeball"]
HARD_CONFIGS = ["classic", "cbam", "spatial", "mlp", "vae", "ae", "recon",
                "contrastive", "aug_crop", "aug_color", "aug_noise"]
PILOT_CONFIGS = ["classic", "cbam", "spatial", "mlp"]
MARL_CONFIGS = ["ippo", "mappo", "vdn", "qmix", "mapoca", "cte", "tarmac"]
MARL_MAPS = ["3m"]
TEMPORAL_CONFIGS = ["mlp", "cnn1d", "tcn", "lstm", "gru", "transformer",
                    "transformer_xl", "mamba", "s4", "s5"]
TEMPORAL_GAMES = ["heist", "maze", "jumper"]

AUG_OF = {"aug_crop": "crop", "aug_color": "color", "aug_noise": "noise"}


def cells(args):
    out = []
    for suite in args.suite:
        if suite == "main":
            games = args.games or MAIN_GAMES
            for cfg in MAIN_CONFIGS:
                ext = {"aug_crop": "classic", "aug_color": "classic",
                       "aug_noise": "classic",
                       "contrastive": "contrastive"}.get(cfg, cfg)
                if cfg == "contrastive":
                    aug, exp = "noise", "none"
                else:
                    aug, exp = AUG_OF.get(cfg, "none"), "none"
                for game in games:
                    for s in args.seeds:
                        for t in args.timesteps:
                            out.append({"suite": suite, "cfg": cfg,
                                        "kind": "ppo", "game": game, "seed": s,
                                        "timesteps": t, "extractor": ext,
                                        "augment": aug, "explore": exp})
        elif suite == "exploration":
            for cfg in EXPLORE_CONFIGS:
                for game in (args.games or EXPLORE_GAMES):
                    for s in args.seeds:
                        for t in args.timesteps:
                            out.append({"suite": suite, "cfg": cfg,
                                        "kind": "ppo", "game": game, "seed": s,
                                        "timesteps": t, "extractor": "classic",
                                        "augment": "none",
                                        "explore": "none" if cfg == "ppo" else cfg})
        elif suite == "algo":
            cfgs = list(ALGO_CONFIGS)
            for cfg in cfgs:
                for game in (args.games or ALGO_GAMES):
                    for s in args.seeds:
                        for t in args.timesteps:
                            out.append({"suite": suite, "cfg": cfg,
                                        "kind": cfg, "game": game, "seed": s,
                                        "timesteps": t, "extractor": "classic",
                                        "augment": "none", "explore": "none"})
            if args.lr_sens:
                for cfg in ("dqn", "qrdqn"):
                    for s in args.seeds:
                        for t in args.timesteps:
                            c = dict(suite=suite, cfg=cfg + "_lr3e-4",
                                     kind=cfg, game="starpilot", seed=s,
                                     timesteps=t, extractor="classic",
                                     augment="none", explore="none", lr=3e-4)
                            out.append(c)
        elif suite == "hrl":
            for arm in HRL_ARMS:
                for game in (args.games or HRL_GAMES):
                    for s in args.seeds:
                        for t in args.timesteps:
                            out.append({"suite": suite, "cfg": arm,
                                        "kind": "hrl", "game": game, "seed": s,
                                        "timesteps": t, "arm": arm})
        elif suite == "budget":
            bsteps = args.budget_steps or args.timesteps
            for cfg in BUDGET_CONFIGS:
                for game in (args.games or BUDGET_GAMES):
                    for s in args.seeds:
                        for t in bsteps:
                            out.append({"suite": suite, "cfg": cfg,
                                        "kind": "ppo", "game": game, "seed": s,
                                        "timesteps": t, "extractor": cfg,
                                        "augment": "none", "explore": "none"})
        elif suite == "hard":
            # Stress test §3.4: same 11 configs from suite, hard mode.
            for cfg in HARD_CONFIGS:
                ext = {"aug_crop": "classic", "aug_color": "classic",
                       "aug_noise": "classic",
                       "contrastive": "contrastive"}.get(cfg, cfg)
                aug = (AUG_OF.get(cfg, "none") if cfg != "contrastive"
                       else "noise")
                for s in args.seeds:
                    for t in args.timesteps:
                        out.append({"suite": suite, "cfg": cfg, "kind": "ppo",
                                    "game": "bossfight", "seed": s,
                                    "timesteps": t, "extractor": ext,
                                    "augment": aug, "explore": "none",
                                    "distribution": "hard"})
        elif suite == "pilot":
            # Pilot coinrun 50k §3.1 (always 50k, as in study).
            for cfg in PILOT_CONFIGS:
                for s in args.seeds:
                    out.append({"suite": suite, "cfg": cfg, "kind": "ppo",
                                "game": "coinrun", "seed": s, "timesteps": 50000,
                                "extractor": cfg, "augment": "none",
                                "explore": "none"})
        elif suite == "spr":
            # EXTENSION beyond study (no parity §1-12): SPR aux.
            for cfg in ("spr", "spr_aug"):
                for game in (args.games or MAIN_GAMES):
                    for s in args.seeds:
                        for t in args.timesteps:
                            out.append({"suite": suite, "cfg": cfg,
                                        "kind": "ppo", "game": game, "seed": s,
                                        "timesteps": t, "extractor": "classic",
                                        "augment": "crop" if cfg == "spr_aug"
                                        else "none",
                                        "explore": "none", "aux": "spr"})
        elif suite == "gnn":
            # EXTENSION beyond study (no parity §1-12): GAT patches.
            for game in (args.games or MAIN_GAMES):
                for s in args.seeds:
                    for t in args.timesteps:
                        out.append({"suite": suite, "cfg": "gat",
                                    "kind": "ppo", "game": game, "seed": s,
                                    "timesteps": t, "extractor": "gat",
                                    "augment": "none", "explore": "none"})
        elif suite == "aux":
            # EXTENSION beyond study: CURL/CPC/ACL (satisfies user request;
            # SPR has its own suite). Classic + aux, no extra aug.
            for cfg in ("curl", "cpc", "acl"):
                for game in (args.games or MAIN_GAMES):
                    for s in args.seeds:
                        for t in args.timesteps:
                            out.append({"suite": suite, "cfg": cfg,
                                        "kind": "ppo", "game": game, "seed": s,
                                        "timesteps": t, "extractor": "classic",
                                        "augment": "none", "explore": "none",
                                        "aux": cfg})
        elif suite == "marl":
            # EXTENSION: MARL on SMAX (outside Procgen study).
            for cfg in MARL_CONFIGS:
                for game in (args.maps or MARL_MAPS):
                    for s in args.seeds:
                        for t in args.timesteps:
                            out.append({"suite": suite, "cfg": cfg,
                                        "kind": cfg, "game": game, "seed": s,
                                        "timesteps": t})
        elif suite == "temporal":
            # EXTENSION: memory bake-off (frame_stack=4) on games that
            # demand temporality (heist/maze/jumper).
            for cfg in TEMPORAL_CONFIGS:
                for game in (args.games or TEMPORAL_GAMES):
                    for s in args.seeds:
                        for t in args.timesteps:
                            out.append({"suite": suite, "cfg": cfg,
                                        "kind": "ppo", "game": game, "seed": s,
                                        "timesteps": t, "extractor": cfg,
                                        "augment": "none", "explore": "none",
                                        "stack": 4})
        elif suite == "temporal_hard":
            # Follows bake-off question: at 500k in games where memory
            # matters crucially (heist-hard, bossfight), does transformer make a difference?
            for cfg in TEMPORAL_CONFIGS:
                for game in (args.games or ["heist", "bossfight"]):
                    for s in args.seeds:
                        for t in args.timesteps:
                            out.append({"suite": suite, "cfg": cfg,
                                        "kind": "ppo", "game": game,
                                        "seed": s, "timesteps": t,
                                        "extractor": cfg,
                                        "augment": "none", "explore": "none",
                                        "stack": 4,
                                        "distribution": "hard"})
    return out


def run_cell(cell, args):
    from jax_port import train as T
    from jax_port import train_dqn as D
    from jax_port import train_hrl as H
    from jax_port.marl import ppo_marl as MP
    from jax_port.marl import train_paradigms as MPA
    from jax_port.marl import train_ql as MQ
    tag = f"{cell['cfg']}__{cell['game']}__seed{cell['seed']}__{cell['timesteps']//1000}k"
    path = os.path.join(args.out_dir, cell["suite"], tag + ".json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and not args.overwrite:
        return {"skipped": path}
    ee = (100, 100, 15) if args.eval_full else (
        args.eval_eps, args.eval_det_eps, args.eval_train_eps)
    if cell["kind"] == "hrl":
        ns = types.SimpleNamespace(
            game=cell["game"], arm=cell["arm"], frames=cell["timesteps"],
            seed=cell["seed"], num_envs=args.num_envs, rollout=128,
            minibatch=1024, eval_eps=ee[0], eval_det_eps=ee[1],
            eval_train_eps=ee[2], eval_envs=8, out=path)
        return H.train(ns)
    if cell["kind"] in ("ippo", "mappo"):
        ns = types.SimpleNamespace(
            algo=cell["kind"], map=cell["game"], timesteps=cell["timesteps"],
            seed=cell["seed"], num_envs=args.num_envs, rollout=args.rollout,
            minibatch=args.minibatch, recurrent=args.recurrent, lr=args.lr,
            ent=args.ent, no_walls=args.no_walls,
            eval_eps=ee[0], eval_envs=8, out=path)
        return MP.train(ns)
    if cell["kind"] in ("vdn", "qmix"):
        ns = types.SimpleNamespace(
            algo=cell["kind"], map=cell["game"], timesteps=cell["timesteps"],
            seed=cell["seed"], num_envs=32, lr=args.ql_lr,
            recurrent=args.recurrent, tau=args.tau,
            target_update_interval=args.target_update_interval,
            eval_eps=ee[0], eval_envs=8, out=path)
        return MQ.train(ns)
    if cell["kind"] in ("mapoca", "cte", "tarmac"):
        ns = types.SimpleNamespace(
            algo=cell["kind"], map=cell["game"], timesteps=cell["timesteps"],
            seed=cell["seed"], num_envs=args.num_envs, rollout=128,
            minibatch=1024, eval_eps=ee[0], eval_envs=8, out=path)
        return MPA.train(ns)
    if cell["kind"] in ("dqn", "qrdqn"):
        ns = types.SimpleNamespace(
            game=cell["game"], algo=cell["kind"], extractor="classic",
            timesteps=cell["timesteps"], seed=cell["seed"],
            lr=cell.get("lr", 1e-4), num_envs=32, eval_eps=ee[0],
            eval_det_eps=ee[1], eval_train_eps=ee[2], eval_envs=8, out=path)
        return D.train(ns)
    algo = cell["kind"]  # ppo | a2c
    ns = types.SimpleNamespace(
        game=cell["game"], algo=algo, extractor=cell["extractor"],
        distribution=cell.get("distribution", "easy"),
        obs=None, augment=cell.get("augment", "none"),
        explore=cell.get("explore", "none"), aux=cell.get("aux", "none"),
        stack=cell.get("stack", 1),
        timesteps=cell["timesteps"],
        seed=cell["seed"], num_envs=args.num_envs, rollout=args.rollout,
        minibatch=args.minibatch, eval_eps=ee[0], eval_det_eps=ee[1],
        eval_train_eps=ee[2], eval_envs=8, out=path)
    return T.train(ns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", nargs="+",
                    default=["main"],
                    choices=["main", "exploration", "algo", "hrl", "budget",
                             "hard", "pilot", "spr", "gnn", "aux", "marl",
                             "temporal", "temporal_hard"])
    ap.add_argument("--games", nargs="*", default=None)
    ap.add_argument("--maps", nargs="*", default=None,
                    help="SMAX maps for marl suite (default: 3m)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--timesteps", type=int, nargs="+", default=[100000])
    ap.add_argument("--budget-steps", type=int, nargs="+", default=None,
                    help="timesteps only for budget suite (default: --timesteps)")
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--rollout", type=int, default=128)
    ap.add_argument("--minibatch", type=int, default=1024)
    ap.add_argument("--recurrent", action="store_true",
                    help="Recurrent IPPO GRU-128 (JaxMARL default for SMAX)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ql-lr", type=float, default=1e-4,                    help="lr for vdn/qmix (study; paper uses 5e-5)")
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--no-walls", action="store_true")
    ap.add_argument("--eval-eps", type=int, default=10)
    ap.add_argument("--eval-det-eps", type=int, default=0)
    ap.add_argument("--eval-train-eps", type=int, default=0)
    ap.add_argument("--eval-full", action="store_true")
    ap.add_argument("--configs", nargs="*", default=None,
                    help="filters cfgs (e.g. classic mlp mlp_vector ppo icm flat)")
    ap.add_argument("--lr-sens", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--out-dir", default=os.path.join(BASE, "results_grade"))
    ap.add_argument("--tau", type=float, default=0.0,
                    help="Polyak coefficient for VDN/QMIX target nets; 0 = hard copy "
                         "(train_ql default, and what every published MARL cell used)")
    ap.add_argument("--target-update-interval", type=int, default=500,
                    help="hard target copy frequency in gradient steps (used when --tau is 0)")
    ap.add_argument("--master", default=os.path.join(BASE, "results_grade", "master.json"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    master = {}
    if os.path.exists(args.master) and not args.overwrite:
        with open(args.master) as fh:
            master = json.load(fh)
    t0 = time.perf_counter()
    all_cells = cells(args)
    if args.configs:
        all_cells = [c for c in all_cells if c["cfg"] in args.configs]
    for i, cell in enumerate(all_cells):
        key = (f"{cell['suite']}/{cell['cfg']}__{cell['game']}__"
               f"seed{cell['seed']}__{cell['timesteps']//1000}k")
        prev = master.get(key)
        if prev and prev.get("ok") and not args.overwrite:
            print(f"[{i}] skip {key}", flush=True)
            continue
        print(f"[{i}] {key}", flush=True)
        try:
            r = run_cell(cell, args)
            master[key] = {"ok": True,
                           "sps": r.get("sps"),
                           "eval_unseen": r.get("eval_unseen"),
                           "out": r.get("skipped", "")}
        except Exception as e:  # noqa: BLE001 (grid cannot crash)
            import traceback
            traceback.print_exc()
            master[key] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        with open(args.master, "w") as fh:
            json.dump(master, fh, indent=2)
    dt = time.perf_counter() - t0
    ok = sum(1 for v in master.values() if v.get("ok"))
    print(f"grade: {ok}/{len(master)} ok in {dt:.0f}s -> {args.master}")


if __name__ == "__main__":
    main()
