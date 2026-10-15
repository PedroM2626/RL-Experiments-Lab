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
    py -3.10 diagnose_policy_saturation.py --timesteps 50000 --probe-every 500 --n-envs 16 --n-steps 16
"""
import argparse
import json
import os
import time
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from compare_action_repeat import VariableRepeatEnv
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


def probe_frames(game, seed, n=20, frame_stack=1, vector=False):
    """Frames from a policy-independent random walk, so the probe set is not adapted to."""
    rng = np.random.default_rng(seed)
    out = []
    for lvl in range(4):
        env = make_procgen_env(game, num_levels=0, distribution_mode="easy",
                               seed=seed + lvl, vector=vector, frame_stack=frame_stack)
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
        # float() at the measurement site only: the vector arm's extractor is a flatten,
        # so its uint8 observation comes back out as uint8 and .mean() refuses it. The
        # cnn arms' features are already float, so this is a no-op for every published
        # curve -- their reproduction must stay bitwise (tests gate it).
        feats = torch.stack([pol.features_extractor(t) for t in tens[:8]]).float()
        probs, sat = [], []
        for t in tens:
            f = pol.extract_features(t, pol.pi_features_extractor)
            net = pol.mlp_extractor.policy_net
            post = f
            k = 0                                      # ordinal of the Tanh, not of the layer:
            for layer in net:                          # policy_net is [Linear, Tanh, Linear, Tanh]
                post = layer(post)                     # and counting by index would leave an
                if isinstance(layer, torch.nn.Tanh):   # empty bucket per Linear -> NaN means
                    while len(sat) <= k:
                        sat.append([])
                    sat[k].append(float((post.abs() > SATURATION_ZONE).float().mean()))
                    k += 1
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
    flat = [x for layer in sat for x in layer]
    return {
        "feat_mean": round(float(feats.mean().item()), 4),
        "feat_max": round(float(feats.max().item()), 3),
        "tanh_saturated_fraction": round(float(np.mean(flat)), 4),
        # per layer, so "which came first" is answerable: the first Tanh sees the extractor
        # output, the second one the mixed latent. A single average over both hides a layer
        # that saturates early behind a layer that never does.
        "tanh_saturated_by_layer": [round(float(np.mean(layer)), 4) for layer in sat],
        "cross_frame_kl_max": round(kl, 6),
        "cross_frame_tv_mean": round(tv, 6),
        "entropy_mean_nats": round(ent, 4),
        "distinct_argmax_actions": int(len(set(int(x) for x in P.argmax(1)))),
        "v_std": round(float(np.std(values)), 5),
    }


GAMES_FOR_CROSS_CHECK = ("maze", "heist", "bossfight", "starpilot", "dodgeball")


def cross_game_response(model, games=GAMES_FOR_CROSS_CHECK, seed=1042, vector=False,
                        frame_stack=1):
    """Total variation between the action distributions of five DIFFERENT games.

    Within one Procgen game consecutive frames barely differ (a 300-step random walk in maze
    yields 19 distinct frames), so a small within-game TV cannot tell "weak signal" from "no
    response at all". Frames from different games differ grossly: if the TV between those is
    still ~0, the policy is not reading the observation.

    vector=True feeds the same games through the 256-D downsampled observation, because the
    mlp_vector arm's input space is a vector and would otherwise reject every frame.
    """
    import numpy as _np
    pol = model.policy
    dev = next(pol.parameters()).device
    probs, sat, feats = [], [], []
    for g in games:
        env = make_procgen_env(g, num_levels=0, distribution_mode="easy", seed=seed,
                               vector=vector, frame_stack=frame_stack)
        o = env.reset()
        o = o[0] if isinstance(o, tuple) else o
        env.close()
        with torch.no_grad():
            t = torch.as_tensor(_np.expand_dims(o, 0)).to(dev)
            f = pol.extract_features(t, pol.pi_features_extractor)
            # magnitude measured through the same extractor measure() uses, but via
            # extract_features so SB3's obs preprocessor runs first: the raw module fed a
            # uint8 frame returns uint8 features for the vector arm, whose .mean() then fails.
            feats.append(pol.extract_features(t, pol.features_extractor))
            net = pol.mlp_extractor.policy_net
            first = net[1](net[0](f))
            sat.append(float((first.abs() > SATURATION_ZONE).float().mean()))
            p = pol._get_action_dist_from_latent(net[3](net[2](first))).distribution.probs
            p = p.detach().cpu().numpy()[0]
            probs.append(p / p.sum())
    P = _np.stack(probs)
    tv = max(float(0.5 * _np.abs(P[i] - P[j]).sum()) for i in range(len(P)) for j in range(i + 1, len(P)))
    F = torch.cat(feats)
    return {"games": list(games), "observation": "vector" if vector else "image",
            "max_pairwise_tv": round(tv, 6),
            "feat_mean": round(float(F.mean()), 4), "feat_max": round(float(F.max()), 3),
            "distinct_argmax_across_games": int(len(set(int(x) for x in P.argmax(1)))),
            "first_tanh_saturated_fraction": round(float(_np.mean(sat)), 4)}


class Watcher(BaseCallback):
    """Probes on the ENVIRONMENT-step grid, not on the vec-env step count.

    One _on_step is one vec step, which is n_envs environment steps -- so the old
    n_calls cadence would space a 16-env arm's probes 16x apart and give every arm a
    different grid. model.num_timesteps is the only clock all arms share: the probe
    fires when it crosses each grid point. The row records the nominal point and the
    actual count, which differ by at most n_envs - 1.
    """

    def __init__(self, frames, every, timesteps):
        super().__init__()
        self.frames, self.every, self.curve = frames, every, []
        self._next = every
        self._last = (timesteps // every) * every

    def _probe(self, at):
        row = {"steps": at, "env_steps": int(self.model.num_timesteps)}
        row.update(measure(self.model, self.frames))
        self.curve.append(row)
        print(f"  step {at:7d}  feat_mean={row['feat_mean']:7.3f} "
              f"feat_max={row['feat_max']:7.2f}  saturated={row['tanh_saturated_fraction']*100:5.1f}%  "
              f"KL={row['cross_frame_kl_max']:.5f} TV={row['cross_frame_tv_mean']:.5f} "
              f"ent={row['entropy_mean_nats']:.3f} argmax_distintos={row['distinct_argmax_actions']}/20",
              flush=True)

    def _on_step(self):
        while self._next <= self.model.num_timesteps and self._next <= self._last:
            self._probe(self._next)
            self._next += self.every
        return True

    def flush(self):
        """Training must not leave grid points unprobed: a curve with holes would read
        as "nothing happened there" instead of "the probe never ran". num_timesteps
        cannot undershoot the budget, so an incomplete grid here is a configuration
        error (probe-every vs n_envs) and fails loudly rather than shipping quietly."""
        if self._next <= self._last:
            last = self.curve[-1]["steps"] if self.curve else "none"
            raise RuntimeError(f"probe grid incomplete: last probe at {last}, "
                               f"{self._last} never reached")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="maze")
    ap.add_argument("--arm", default="ppo", choices=("ppo", "rnd"))
    ap.add_argument("--timesteps", type=int, default=20000)
    ap.add_argument("--probe-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-envs", type=int, default=1,
                    help="parallel environment count. The published recipe is 1; raising it "
                         "is the intervention on the rollout itself, because each update "
                         "then sees N decorrelated trajectories instead of one long one. "
                         "Each env gets its own level stream (construction seed + rank) -- "
                         "N envs on one seed would be N copies of the same trajectory")
    ap.add_argument("--n-steps", type=int, default=256,
                    help="rollout length per env per update. n_envs * n_steps is the "
                         "per-update batch, so holding it at the published 256 while "
                         "raising n_envs isolates the rollout's SHAPE from its SIZE")
    ap.add_argument("--action-repeat", type=int, default=1,
                    help="frames repeated per agent step in the TRAINING env only "
                         "(VariableRepeatEnv, the compare_action_repeat.py wrapper). The "
                         "mechanism lever for the input-correlation account: at n_envs=1 "
                         "a repeat of k multiplies how much the screen can change between "
                         "two observations the policy actually sees. Costs k frames per "
                         "agent step, so the budget in agent steps is unchanged but the "
                         "run sees k times the primitive frames")
    ap.add_argument("--ent-coef", type=float, default=0.0,
                    help="PPO entropy coefficient. The published recipe leaves it at 0.0; "
                         "the mechanism lever for the entropy account: if the collapse is "
                         "the marginal being a comfortable optimum of the clipped "
                         "objective, active entropy pressure should change whether the "
                         "policy stays responsive")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--arch", choices=("classic", "vit", "mlp", "impoola"), default="classic",
                    help="which front end to probe. classic = the published extractor "
                         "(--extractor repo|layernorm applies to it); vit/impoola = those "
                         "front ends of the 16-arm comparison, probed because the grid "
                         "measures them among the non-readers; mlp = the MlpPolicy over "
                         "the 256-D downsampled observation (vector obs, frame stacking "
                         "unavailable there)")
    ap.add_argument("--extractor", choices=("repo", "layernorm"), default="repo",
                    help="repo = the published ClassicCNNExtractor; layernorm = the counterfactual")
    ap.add_argument("--out", default=os.path.join(BASE, "results", "policy_saturation_curve.json"))
    ap.add_argument("--save-model", default=None)
    ap.add_argument("--cross-game", action="store_true",
                    help="after training, measure the response to five different Procgen games")
    ap.add_argument("--frame-stack", type=int, default=1,
                    help="tiles the last N frames into the observation. The intervention that "
                         "tests the low-input-variance account directly: if the frame barely "
                         "changes between steps, showing N of them is what makes motion visible "
                         "inside a single observation, and a policy that still cannot condition "
                         "is not being starved of input change")
    args = ap.parse_args()

    vector = args.arch == "mlp"
    if vector:
        assert args.frame_stack == 1 and args.extractor == "repo", \
            "the mlp arm observes a 256-D vector: no frames to stack, no extractor to swap"

    frames = probe_frames(args.game, args.seed + 1000, frame_stack=args.frame_stack,
                          vector=vector)
    print(f"probe: {len(frames)} frames de {args.game} (random walk, seed {args.seed + 1000})")

    def make_env(rank):
        env = make_procgen_env(args.game, num_levels=200, distribution_mode="easy",
                               seed=args.seed + rank, vector=vector, frame_stack=args.frame_stack)
        if args.action_repeat > 1:
            env = VariableRepeatEnv(env, args.action_repeat)
        if args.arm == "rnd":
            from models.bonuses import RNDWrapper
            env = RNDWrapper(env)
        return Monitor(env)
    from stable_baselines3.common.monitor import Monitor
    venv = DummyVecEnv([partial(make_env, rank) for rank in range(args.n_envs)])

    front = None
    if args.arch == "vit":
        from models.combined_extractors import ViTExtractor
        front = ViTExtractor
    elif args.arch == "impoola":
        from models.combined_extractors import ImpoolaCNNExtractor
        front = ImpoolaCNNExtractor
    if vector:
        policy, pk = "MlpPolicy", {}
    else:
        policy = "CnnPolicy"
        if front is None:
            front = LayerNormExtractor if args.extractor == "layernorm" else ClassicCNNExtractor
        pk = {"features_extractor_class": front,
              "features_extractor_kwargs": dict(features_dim=512)}
    model = PPO(policy, venv, verbose=0, learning_rate=3e-4, n_steps=args.n_steps,
                batch_size=64,
                n_epochs=3, gamma=0.99, gae_lambda=0.95, clip_range=0.2, seed=args.seed,
                ent_coef=args.ent_coef, device=args.device, policy_kwargs=pk)
    w = Watcher(frames, args.probe_every, args.timesteps)
    print(f"treinando {args.arm}/{args.arch} em {args.game}, {args.timesteps} passos, "
          f"n_envs={args.n_envs} x n_steps={args.n_steps} (batch por update "
          f"{args.n_envs * args.n_steps}), device={args.device}")
    row0 = {"steps": 0, "env_steps": 0}
    row0.update(measure(model, frames))
    w.curve.append(row0)
    print(f"  passo      0  feat_mean={row0['feat_mean']:7.3f} feat_max={row0['feat_max']:7.2f} "
          f"saturated={row0['tanh_saturated_fraction']*100:5.1f}%  KL={row0['cross_frame_kl_max']:.5f} "
          f"TV={row0['cross_frame_tv_mean']:.5f} ent={row0['entropy_mean_nats']:.3f} "
          f"argmax_distintos={row0['distinct_argmax_actions']}/20", flush=True)
    started = time.time()
    model.learn(total_timesteps=args.timesteps, callback=w)
    seconds = round(time.time() - started, 1)
    w.flush()
    venv.close()

    cross = cross_game_response(model, vector=vector, frame_stack=args.frame_stack) \
        if args.cross_game else None
    payload = {
        "_provenance": {
            "produced_by": "diagnose_policy_saturation.py",
            "question": "when does the trained policy stop conditioning on the observation",
            "game": args.game, "arm": args.arm, "timesteps": args.timesteps,
            "arch": args.arch,
            "extractor": args.extractor, "frame_stack": args.frame_stack,
            "n_envs": args.n_envs, "n_steps": args.n_steps,
            "per_update_batch": args.n_envs * args.n_steps,
            "action_repeat": args.action_repeat, "ent_coef": args.ent_coef,
            "seed": args.seed, "probe_frames": len(frames),
            "probe_every": args.probe_every,
            "training_seconds": seconds,
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
