"""Record a dashboard-style video of a trained agent, from an existing checkpoint.

STATUS (11/10/2026): built to produce the README-top demo, but it is on hold. Rendering it
now would show the panel "what it thinks" as one unchanging row, because the maze/heist
policies do not condition on the observation (README 3.6.2). It is kept because that panel
is how the defect was found, and because it is the artifact to re-run once the arms are fixed.

No training happens here: this loads a published .zip and runs inference, the same access
path re_eval_100.py uses. Every panel is filled from a value the run actually produced:

  * the game frame and the step/return counters come from the environment;
  * "what the network sees" is the 512-D latent the features extractor hands to the actor;
  * the action table is the policy's own categorical probabilities;
  * the button diagram decodes the chosen action index through Procgen's own key-combo
    table (read from the installed package at runtime, never hardcoded here, so the video
    cannot drift from the library that defines it);
  * the reward-prediction trace is the critic's V(s) per step.

Progress is reported as episode steps against the 1000-step Procgen episode limit, because
that is the only progress quantity the protocol exposes for maze: Procgen hands the agent
pixels and a reward, nothing about goal distance.

Usage:
    py -3.10 record_dashboard_video.py --checkpoint logs_maze_heist_norm/.../maze_rnd_seed42.zip
    py -3.10 record_dashboard_video.py --episodes 4 --pick goal   # prefer an episode that scored
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from procgen_wrapper import make_procgen_env

BASE = os.path.dirname(os.path.abspath(__file__))
EPISODE_LIMIT = 1000          # Procgen's own max episode length
PROVENANCE = os.path.join(BASE, "results", "exploration_normalized.json")


def action_combos(game):
    """The action index -> pressed-keys table, taken from the installed procgen package.

    Parsed with ast rather than hardcoded: the video labels buttons, and a hand-copied table
    could drift from the library that actually defines the action space.
    """
    import ast
    import inspect
    src = inspect.getsource(__import__("procgen.env", fromlist=["env"]))
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "get_combos":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Return) and isinstance(sub.value, ast.List):
                    combos = [ast.literal_eval(ast.unparse(e)) for e in sub.value.elts]
                    if len(combos) != 15:
                        raise RuntimeError(
                            f"expected 15 action combos from procgen.env, read {len(combos)}")
                    return [tuple(c) for c in combos]
    raise RuntimeError("procgen.env has no get_combos() returning a literal list; "
                       "refusing to fall back to a hardcoded table")


def rollout(model, game, level_seed, max_steps):
    """One episode, recording everything the dashboard shows."""
    env = make_procgen_env(game, num_levels=0, distribution_mode="easy",
                           seed=level_seed, vector=False)
    policy = model.policy
    obs = env.reset()
    obs = obs[0] if isinstance(obs, tuple) else obs
    steps = []
    total = 0.0
    for _ in range(max_steps):
        ot, _ = policy.obs_to_tensor(np.expand_dims(obs, 0))
        with torch.no_grad():
            feats = policy.extract_features(ot, policy.pi_features_extractor)
            latent_pi = policy.mlp_extractor.forward_actor(feats)
            dist = policy._get_action_dist_from_latent(latent_pi)
            probs = dist.distribution.probs.numpy()[0].astype(np.float64)
            value = float(policy.predict_values(ot).numpy()[0, 0])
        action = int(dist.distribution.sample().numpy()[0])
        obs, reward, term, trunc, _ = env.step(action)
        obs = obs[0] if isinstance(obs, tuple) else obs
        total += float(reward)
        steps.append({"obs": obs.copy(), "probs": probs, "value": value, "action": action,
                      "reward": float(reward), "return": total,
                      "latent": latent_pi.numpy()[0].copy()})
        if term or trunc:
            break
    env.close()
    return steps


def draw(step_idx, steps, combos, meta, ax_layout):
    """Compose one dashboard frame into the figure's axes grid."""
    s = steps[step_idx]
    (ax_game, ax_lat, ax_act, ax_btn, ax_val, ax_bar) = ax_layout

    ax_game.imshow(np.transpose(s["obs"], (1, 2, 0)), interpolation="nearest")
    ax_game.set_title("O JOGO  ·  %s" % meta["game"], fontsize=9, loc="left")
    ax_game.set_xticks([]); ax_game.set_yticks([])
    goal = "  ·  recompensa +%.0f" % s["reward"] if s["reward"] > 0 else ""
    ax_game.set_xlabel("passo %d/%d   retorno %.2f%s" % (step_idx + 1, len(steps), s["return"], goal),
                       fontsize=8, loc="left")

    ax_lat.clear(); ax_lat.imshow(s["latent"].reshape(4, 128), aspect="auto", cmap="viridis")
    ax_lat.set_title("O QUE A REDE VÊ  ·  latente de 512-D do extrator", fontsize=8)
    ax_lat.set_xticks([]); ax_lat.set_yticks([])

    ax_act.clear()
    order = np.argsort(-s["probs"])[:6]
    labels = ["+".join(combos[i]) if combos[i] else "nada" for i in order]
    ax_act.barh(range(len(order)), [s["probs"][i] * 100 for i in order],
                color=["#f0a500" if i == s["action"] else "#3a4256" for i in order])
    ax_act.set_yticks(range(len(order)))
    ax_act.set_yticklabels(["%s  %.0f%%" % (l, s["probs"][i] * 100) for l, i in zip(labels, order)],
                           fontsize=7)
    ax_act.invert_yaxis()
    ax_act.set_title("O QUE ELA PENSA  ·  6 ações mais prováveis", fontsize=8)
    ax_act.set_xlim(0, 100); ax_act.tick_params(axis="x", labelsize=7)
    for sp in ("top", "right"):
        ax_act.spines[sp].set_visible(False)

    ax_btn.clear()
    ax_btn.set_xlim(0, 4); ax_btn.set_ylim(0, 2); ax_btn.axis("off")
    pressed = set(combos[s["action"]])
    for x, name in [(0, "LEFT"), (1, "DOWN"), (2, "RIGHT"), (3, "UP")]:
        ax_btn.add_patch(plt.Rectangle((x, 0.4 if name in ("LEFT", "RIGHT") else 0.0), 0.8, 0.8,
                                       color="#f0a500" if name in pressed else "#242938"))
    for x, name in [(1.1, "A"), (2.1, "B"), (3.1, "X"), (0.1, "Y")]:
        ax_btn.add_patch(plt.Circle((x, 1.5), 0.28, color="#f0a500" if name in pressed else "#242938"))
    ax_btn.text(0, 1.95, "BOTÕES  ·  ação %d" % s["action"], fontsize=8)

    ax_val.clear()
    vs = [x["value"] for x in steps[:step_idx + 1]]
    ax_val.plot(vs, color="#7fd1ae", lw=1)
    ax_val.set_ylim(min(-0.2, min(vs) - 0.1), max(1.0, max(vs) + 0.1))
    ax_val.set_title("PREVISÃO DE RECOMPENSA  ·  V(s) = %.3f" % s["value"], fontsize=8)
    ax_val.tick_params(labelsize=7)
    for sp in ("top", "right"):
        ax_val.spines[sp].set_visible(False)

    ax_bar.clear(); ax_bar.axis("off")
    frac = (step_idx + 1) / len(steps)
    ax_bar.add_patch(plt.Rectangle((0, 0.3), 1, 0.4, color="#242938"))
    ax_bar.add_patch(plt.Rectangle((0, 0.3), frac, 0.4, color="#f0a500"))
    ax_bar.set_xlim(0, 1); ax_bar.set_ylim(0, 1)
    ax_bar.text(1.0, 0.75, "passo %d de %d (limite de episódio %d)"
                % (step_idx + 1, len(steps), EPISODE_LIMIT), fontsize=7, ha="right")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--game", default="maze")
    ap.add_argument("--level-seed", type=int, default=1042, help="eval seed (levels unseen)")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--pick", choices=("first", "goal"), default="goal",
                    help="goal = prefer the recorded episode with the highest return")
    ap.add_argument("--max-steps", type=int, default=EPISODE_LIMIT)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--stride", type=int, default=2, help="render every Nth step (keeps the gif small)")
    ap.add_argument("--out", default=os.path.join(BASE, "results", "agent_dashboard"))
    args = ap.parse_args()

    from stable_baselines3 import PPO
    model = PPO.load(args.checkpoint, device="cpu")
    combos = action_combos(args.game)

    runs = [rollout(model, args.game, args.level_seed + k, args.max_steps)
            for k in range(args.episodes)]
    chosen = max(runs, key=lambda r: r[-1]["return"]) if args.pick == "goal" else runs[0]
    print(f"episódios gravados: {len(runs)} | escolhido: {len(chosen)} passos, "
          f"retorno {chosen[-1]['return']:.2f}")

    arm = os.path.basename(args.checkpoint).replace(".zip", "")
    stat = ""
    if os.path.exists(PROVENANCE):
        with open(PROVENANCE, encoding="utf-8") as f:
            prov = json.load(f)
        key = f"{args.game}_{arm.split('_')[1]}" if "_" in arm else None
        row = prov.get("statistics_100eps", {}).get(key) if key else None
        if row:
            stat = f" · {row['mean']:.2f} IC95 [{row['ci95'][0]:.2f}, {row['ci95'][1]:.2f}] em 100 eps"
    meta = {"game": f"{args.game} · {arm}{stat}"}

    fig = plt.figure(figsize=(11, 5.2), dpi=100)
    fig.patch.set_facecolor("#10141f")
    layout = [fig.add_axes(l) for l in
              [[0.035, 0.16, 0.30, 0.66], [0.395, 0.60, 0.28, 0.22],
               [0.395, 0.16, 0.28, 0.38], [0.715, 0.42, 0.26, 0.40],
               [0.715, 0.16, 0.26, 0.20], [0.035, 0.05, 0.94, 0.06]]]
    for a in layout:
        a.set_facecolor("#10141f")
    fig.suptitle("Benchmark de RL em Procgen — o que um agente treinado faz, passo a passo",
                 color="w", fontsize=11, x=0.035, ha="left", y=0.965)

    os.makedirs(args.out, exist_ok=True)
    idxs = list(range(0, len(chosen), args.stride))
    if idxs[-1] != len(chosen) - 1:
        idxs.append(len(chosen) - 1)

    import imageio.v3 as iio
    frames = []
    for n, i in enumerate(idxs):
        draw(i, chosen, combos, meta, layout)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        frames.append(buf)
        if n % 25 == 0:
            print(f"  frame {n+1}/{len(idxs)}", flush=True)
    plt.close(fig)

    mp4 = args.out + ".mp4"
    iio.imwrite(mp4, np.stack(frames), fps=args.fps, codec="libx264",
                ffmpeg_params=["-pix_fmt", "yuv420p"])
    gif = args.out + ".gif"
    iio.imwrite(gif, frames[:: max(1, args.stride)], fps=max(4, args.fps // 3), loop=0)
    for p in (mp4, gif):
        print(f"{p}: {os.path.getsize(p)/1e6:.2f} MB")


if __name__ == "__main__":
    main()
