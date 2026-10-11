"""Diagnose when a PPO policy stops conditioning on the observation.

The maze/heist checkpoints of README 3.6/3.6.1 were measured (27/09/2026) to emit the SAME
action distribution for every frame of a real episode: cross-frame KL 0.000000, one argmax
over 20 distinct frames, V(s) std 0.00000, and 64/64 units of the actor's first Tanh at
|tanh| > 0.999. The same extractor at INITIALIZATION is not saturated at all (feature mean
0.291 / max 2.30, 0.0% saturated units), so the blindness is acquired during training.

This script watches the acquisition: it trains PPO on one game and, every --probe-every
steps, measures on a fixed set of held-out frames
  * the extractor feature scale (mean, max),
  * the fraction of the actor's latent units inside the Tanh saturation zone,
  * the maximum cross-frame KL between action distributions,
  * how many distinct argmax actions the policy picks across those frames.

If saturation climbs while cross-frame KL collapses, the mechanism is a runaway feature
scale pushing a Tanh latent out of its responsive range, where the policy gradient through
it vanishes and the policy freezes into a state-independent action prior.

Usage:
    py -3.10 diagnose_policy_saturation.py --timesteps 20000 --out results/policy_saturation_curve.json
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from models.sb3_extractors import ClassicCNNExtractor
from procgen_wrapper import make_procgen_env

BASE = os.path.dirname(os.path.abspath(__file__))


class LayerNormExtractor(BaseFeaturesExtractor):
    """Same CNN as ClassicCNNExtractor, but its 512-D output is LayerNormed.

    The counterfactual arm: if the un-normalized feature scale growing during training is what
    saturates the Tanh latent, clamping the scale at the extractor output should keep the
    policy responsive at the published 100k budget.
    """

    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        inner = ClassicCNNExtractor(observation_space, features_dim)
        self.cnn = inner.cnn
        self.linear = nn.Sequential(inner.linear, nn.LayerNorm(features_dim))

    def forward(self, observations):
        if observations.dtype == torch.uint8:
            observations = observations.float() / 255.0
        return self.linear(self.cnn(observations))
SATURATION_ZONE = 0.999      # |tanh(x)| above this is numerically saturated


def probe_frames(game, seed, n=20):
    """Frames from a policy-independent random walk, so the probe set is not adapted to."""
    rng = np.random.default_rng(seed)
    out = []
    for lvl in range(4):
        env = make_procgen_env(game, num_levels=0, distribution_mode="easy",
                               seed=seed + lvl, vector=False)
        o = env.reset()
        o = o[0] if isinstance(o, tuple) else o
        for _ in range(5):
            out.append(o.copy())
            o, r, te, tr, _ = env.step(int(rng.integers(0, 15)))
            o = o[0] if isinstance(o, tuple) else o
        env.close()
    return out[:n]


def measure(model, frames):
    pol = model.policy
    dev = next(pol.parameters()).device
    tens = [torch.as_tensor(np.expand_dims(f, 0)).to(dev) for f in frames]
    with torch.no_grad():
        feats = torch.stack([pol.features_extractor(t) for t in tens[:8]])
        probs, sat = [], []
        for t in tens:
            f = pol.extract_features(t, pol.pi_features_extractor)
            net = pol.mlp_extractor.policy_net
            post = f
            for layer in net:                       # walk the latent MLP, count after each Tanh
                post = layer(post)
                if isinstance(layer, torch.nn.Tanh):
                    sat.append(float((post.abs() > SATURATION_ZONE).float().mean()))
            p = pol._get_action_dist_from_latent(post).distribution.probs.detach().cpu().numpy()[0]
            probs.append(p / p.sum())
        P = np.stack(probs)
        p0 = P[0]
        kl = max(float(np.sum(p0 * np.log(p0 / (q + 1e-12)))) for q in P)
        # total variation against the first frame: unlike KL it does not vanish when the
        # distributions are merely near-uniform, so it separates "responding weakly" from
        # "responding not at all"
        tv = float(np.mean([0.5 * np.abs(q - p0).sum() for q in P]))
        ent = float(np.mean([-np.sum(q * np.log(q + 1e-12)) for q in P]))
        values = [float(pol.predict_values(t).detach().cpu().numpy().ravel()[0]) for t in tens]
    return {
        "feat_mean": round(float(feats.mean().item()), 4),
        "feat_max": round(float(feats.max().item()), 3),
        "tanh_saturated_fraction": round(float(np.mean(sat)), 4),
        "cross_frame_kl_max": round(kl, 6),
        "cross_frame_tv_mean": round(tv, 6),
        "entropy_mean_nats": round(ent, 4),
        "distinct_argmax_actions": int(len(set(int(x) for x in P.argmax(1)))),
        "v_std": round(float(np.std(values)), 5),
    }


GAMES_FOR_CROSS_CHECK = ("maze", "heist", "bossfight", "starpilot", "dodgeball")


def cross_game_response(model, games=GAMES_FOR_CROSS_CHECK, seed=1042):
    """Total variation between the action distributions of five DIFFERENT games.

    Within one Procgen game consecutive frames barely differ (a 300-step random walk in maze
    yields 19 distinct frames), so a small within-game TV cannot tell "weak signal" from "no
    response at all". Frames from different games differ grossly: if the TV between those is
    still ~0, the policy is not reading the observation.
    """
    import numpy as _np
    pol = model.policy
    probs, sat = [], []
    for g in games:
        env = make_procgen_env(g, num_levels=0, distribution_mode="easy", seed=seed, vector=False)
        o = env.reset()
        o = o[0] if isinstance(o, tuple) else o
        env.close()
        with torch.no_grad():
            t = torch.as_tensor(_np.expand_dims(o, 0)).to(next(pol.parameters()).device)
            feats = pol.extract_features(t, pol.pi_features_extractor)
            net = pol.mlp_extractor.policy_net
            first = net[1](net[0](feats))
            sat.append(float((first.abs() > SATURATION_ZONE).float().mean()))
            p = pol._get_action_dist_from_latent(net[3](net[2](first))).distribution.probs
            p = p.detach().cpu().numpy()[0]
            probs.append(p / p.sum())
    P = _np.stack(probs)
    tv = max(float(0.5 * _np.abs(P[i] - P[j]).sum()) for i in range(len(P)) for j in range(i + 1, len(P)))
    return {"games": list(games), "max_pairwise_tv": round(tv, 6),
            "distinct_argmax_across_games": int(len(set(int(x) for x in P.argmax(1)))),
            "first_tanh_saturated_fraction": round(float(_np.mean(sat)), 4)}


class Watcher(BaseCallback):
    def __init__(self, frames, every):
        super().__init__()
        self.frames, self.every, self.curve = frames, every, []

    def _on_step(self):
        if self.n_calls % self.every == 0:
            row = {"steps": self.n_calls}
            row.update(measure(self.model, self.frames))
            self.curve.append(row)
            print(f"  step {self.n_calls:7d}  feat_mean={row['feat_mean']:7.3f} "
                  f"feat_max={row['feat_max']:7.2f}  saturated={row['tanh_saturated_fraction']*100:5.1f}%  "
                  f"KL={row['cross_frame_kl_max']:.5f} TV={row['cross_frame_tv_mean']:.5f} "
                  f"ent={row['entropy_mean_nats']:.3f} argmax_distintos={row['distinct_argmax_actions']}/20",
                  flush=True)
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="maze")
    ap.add_argument("--arm", default="ppo", choices=("ppo", "rnd"))
    ap.add_argument("--timesteps", type=int, default=20000)
    ap.add_argument("--probe-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--extractor", choices=("repo", "layernorm"), default="repo",
                    help="repo = the published ClassicCNNExtractor; layernorm = the counterfactual")
    ap.add_argument("--out", default=os.path.join(BASE, "results", "policy_saturation_curve.json"))
    ap.add_argument("--save-model", default=None)
    ap.add_argument("--cross-game", action="store_true",
                    help="after training, measure the response to five different Procgen games")
    args = ap.parse_args()

    frames = probe_frames(args.game, args.seed + 1000)
    print(f"probe: {len(frames)} frames de {args.game} (random walk, seed {args.seed + 1000})")

    def make_env():
        env = make_procgen_env(args.game, num_levels=200, distribution_mode="easy",
                               seed=args.seed, vector=False)
        if args.arm == "rnd":
            from models.bonuses import RNDWrapper
            env = RNDWrapper(env)
        return Monitor(env)
    from stable_baselines3.common.monitor import Monitor
    venv = DummyVecEnv([make_env])

    model = PPO("CnnPolicy", venv, verbose=0, learning_rate=3e-4, n_steps=256, batch_size=64,
                n_epochs=3, gamma=0.99, gae_lambda=0.95, clip_range=0.2, seed=args.seed,
                device=args.device,
                policy_kwargs={"features_extractor_class":
                               LayerNormExtractor if args.extractor == "layernorm" else ClassicCNNExtractor,
                               "features_extractor_kwargs": dict(features_dim=512)})
    w = Watcher(frames, args.probe_every)
    print(f"treinando {args.arm}/{args.extractor} em {args.game}, {args.timesteps} passos, device={args.device}")
    row0 = {"steps": 0}
    row0.update(measure(model, frames))
    w.curve.append(row0)
    print(f"  passo      0  feat_mean={row0['feat_mean']:7.3f} feat_max={row0['feat_max']:7.2f} "
          f"saturated={row0['tanh_saturated_fraction']*100:5.1f}%  KL={row0['cross_frame_kl_max']:.5f} "
          f"TV={row0['cross_frame_tv_mean']:.5f} ent={row0['entropy_mean_nats']:.3f} "
          f"argmax_distintos={row0['distinct_argmax_actions']}/20", flush=True)
    model.learn(total_timesteps=args.timesteps, callback=w)
    venv.close()

    cross = cross_game_response(model) if args.cross_game else None
    payload = {
        "_provenance": {
            "produced_by": "diagnose_policy_saturation.py",
            "question": "when does the trained policy stop conditioning on the observation",
            "game": args.game, "arm": args.arm, "timesteps": args.timesteps,
            "extractor": args.extractor,
            "seed": args.seed, "probe_frames": len(frames),
            "saturation_zone": SATURATION_ZONE,
        },
        "curve": w.curve,
        "cross_game": cross,
    }
    if args.save_model:
        model.save(args.save_model)
        print(f"policy salva em {args.save_model}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"curva salva em {os.path.relpath(args.out, BASE).replace(os.sep, '/')}")


if __name__ == "__main__":
    main()
