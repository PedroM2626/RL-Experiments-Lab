"""Where does the blindness live: in the front end, or in the head?

README 3.6.9 found vit/bossfight never ignites -- its trained action distribution is
frame-invariant from the first update on. Two architectures could produce that: the
front end collapses its features to frame-invariance (the head never receives a
signal), or the features vary and the head learned to ignore them. This script trains
one cell (the grid recipe, one arm, one game) and measures BOTH levels on the same
held-out frames the probe uses:

  * feature response -- per-dimension cross-frame range, normalized by the run's own
    mean |activation| scale (one shared normalizer, so dims are comparable and a
    dim oscillating around zero cannot explode the measure); reported as the mean
    normalized range and the fraction of dims whose range exceeds the scale;
  * action response -- the probe's cross-frame TV on the same frames, and the audit's
    cross-game TV on five games' initial frames with the feature spread over those
    same frames.

The comparison that answers the question is between arms on the same game: a reader
(classic) calibrates what live features look like; the non-readers then say whether
their front ends encode state at all.

Usage:
    py -3.10 diagnose_feature_vs_actor.py --game bossfight --arch vit
"""
import argparse
import json
import os
import time
from functools import partial

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from diagnose_policy_saturation import (LayerNormExtractor, ViTLayerNormExtractor,
                                         probe_frames, cross_game_response)
from models.sb3_extractors import ClassicCNNExtractor
from procgen_wrapper import make_procgen_env

BASE = os.path.dirname(os.path.abspath(__file__))


def feature_levels(model, frames):
    """The extractor's features for each frame, preprocessed the way the policy sees them."""
    pol = model.policy
    dev = next(pol.parameters()).device
    with torch.no_grad():
        tens = [torch.as_tensor(np.expand_dims(f, 0)).to(dev) for f in frames]
        feats = torch.stack([pol.extract_features(t, pol.features_extractor).float()
                              for t in tens])
    return feats.squeeze(1)


def feature_response(model, frames):
    """Cross-frame spread of the features, normalized by the run's own activation scale."""
    f = feature_levels(model, frames)
    scale = float(f.abs().mean())
    rng = (f.amax(0) - f.amin(0)).squeeze(0)
    norm = rng / (scale + 1e-9)
    return {"feat_scale": round(scale, 4),
            "feat_range_over_scale_mean": round(float(norm.mean()), 4),
            "feat_dims_range_over_scale": round(float((norm > 1.0).float().mean()), 4),
            "feat_dims": int(f.shape[-1])}


def action_tv(model, frames):
    """The probe's measure: mean total variation of the action distribution against
    the first frame -- the same quantity diagnose_policy_saturation.measure reports.
    The features must go through the head's policy_net before the distribution, the
    same traversal measure() performs."""
    pol = model.policy
    dev = next(pol.parameters()).device
    with torch.no_grad():
        probs = []
        for f in frames:
            t = torch.as_tensor(np.expand_dims(f, 0)).to(dev)
            post = pol.extract_features(t, pol.pi_features_extractor)
            for layer in pol.mlp_extractor.policy_net:
                post = layer(post)
            p = pol._get_action_dist_from_latent(post).distribution.probs
            probs.append(p.detach().cpu().numpy()[0])
    P = np.stack(probs)
    return round(float(np.mean([0.5 * np.abs(P[i] - P[0]).sum() for i in range(len(P))])), 6)


def cross_frames(games, seed=1042, vector=False):
    """The audit's cross-game frames: the initial frame of each of five games."""
    out = []
    for g in games:
        env = make_procgen_env(g, num_levels=0, distribution_mode="easy", seed=seed,
                               vector=vector)
        o = env.reset()
        out.append((o[0] if isinstance(o, tuple) else o).copy())
        env.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="bossfight")
    ap.add_argument("--arch", choices=("classic", "vit", "mlp", "impoola", "vit_ln"),
                    default="vit")
    ap.add_argument("--timesteps", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--n-steps", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    vector = args.arch == "mlp"
    frames = probe_frames(args.game, args.seed + 1000, vector=vector)
    cg_frames = cross_frames(("maze", "heist", "bossfight", "starpilot", "dodgeball"),
                             vector=vector)

    def make_env(rank):
        return Monitor(make_procgen_env(args.game, num_levels=200, distribution_mode="easy",
                                        seed=args.seed + rank, vector=vector))
    venv = DummyVecEnv([partial(make_env, rank) for rank in range(args.n_envs)])

    front = None
    if args.arch == "vit":
        from models.combined_extractors import ViTExtractor
        front = ViTExtractor
    elif args.arch == "impoola":
        from models.combined_extractors import ImpoolaCNNExtractor
        front = ImpoolaCNNExtractor
    elif args.arch == "vit_ln":
        front = ViTLayerNormExtractor
    if vector:
        policy, pk = "MlpPolicy", {}
    else:
        policy = "CnnPolicy"
        if front is None:
            front = ClassicCNNExtractor
        pk = {"features_extractor_class": front, "features_extractor_kwargs": dict(features_dim=512)}

    model = PPO(policy, venv, verbose=0, learning_rate=3e-4, n_steps=args.n_steps,
                batch_size=64, n_epochs=3, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                seed=args.seed, device=args.device, policy_kwargs=pk)
    print(f"treinando {args.arch} em {args.game}, {args.timesteps} passos "
          f"({args.n_envs}x{args.n_steps})", flush=True)
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps)
    venv.close()
    seconds = round(time.time() - t0, 1)

    row = {
        "within_game": {"features": feature_response(model, frames),
                        "action_tv": action_tv(model, frames)},
        "cross_game": {"features": feature_response(model, cg_frames),
                       "action_tv": action_tv(model, cg_frames),
                       **cross_game_response(model, vector=vector)},
    }
    out = args.out or os.path.join(BASE, "results",
                                   f"feature_vs_actor_{args.arch}_{args.game}.json")
    payload = {"_provenance": {
        "produced_by": "diagnose_feature_vs_actor.py",
        "question": "do the blind cells' front ends encode state, and does the head use it",
        "game": args.game, "arch": args.arch, "timesteps": args.timesteps,
        "n_envs": args.n_envs, "n_steps": args.n_steps, "seed": args.seed,
        "seconds": seconds,
        "feature_measure": "per-dim cross-frame range over the run's mean |activation|; "
                           "dims_range_over_scale = fraction of dims whose range "
                           "exceeds that scale",
    }, **row}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    w, c = row["within_game"], row["cross_game"]
    print(f"within-game: action TV={w['action_tv']}  feat range/scale="
          f"{w['features']['feat_range_over_scale_mean']}  "
          f"dims>scale={w['features']['feat_dims_range_over_scale']}")
    print(f"cross-game : action TV={c['action_tv']}  feat range/scale="
          f"{c['features']['feat_range_over_scale_mean']}  "
          f"dims>scale={c['features']['feat_dims_range_over_scale']}")
    print(f"saved {os.path.relpath(out, BASE)}")


if __name__ == "__main__":
    main()
