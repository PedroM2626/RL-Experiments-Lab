"""Why does the `vae` arm diverge to NaN (README 3.6.2 / 3.6.7)?

The published VAEExtractor samples z = mu + eps * exp(0.5 * logvar) with `logvar` an
unbounded linear head and no KL term anywhere in the PPO objective (the module docstring
says so), so nothing holds the posterior variance. The hypothesis this script tests: PPO's
gradient drives `logvar` (or `mu`) upward until the sampled features overflow -- on
dodgeball the extractor output reached 7.2e13 -- and on bossfight/seed 42 the logits go NaN.

It trains the arm exactly as the audit does (compare_suite.train_one's recipe: PPO,
lr 3e-4, n_steps 256 per env, batch 64, 3 epochs, per-rank level streams), with forward
hooks recording, per rollout, the max of logvar, |mu|, the sampled std and the extractor
output. A run ends either at its budget or at the first non-finite value, and the JSON
records the trajectory up to that point plus the error.

--logvar-clamp L runs the fixed variant instead (VAEExtractor(logvar_clamp=L)), so the
same script measures the bug and the fix under one recipe.

    py -3.10 -u diagnose_vae_nan.py --game bossfight --seed 42 --out results/vae_nan_bossfight_s42.json
    py -3.10 -u diagnose_vae_nan.py --game bossfight --seed 42 --logvar-clamp 10 --out results/vae_clamped_bossfight_s42.json
"""
import argparse
import json
import math
import os
import time

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from models.world_model_extractors import VAEExtractor
from procgen_wrapper import make_procgen_env


class _Tracker:
    """Running maxima of the posterior and the features over the current rollout window."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.logvar_max = -math.inf
        self.mu_absmax = 0.0
        self.std_max = 0.0
        self.feat_max = 0.0
        self.nonfinite = False

    def posterior(self, mu, logvar):
        lv, m = logvar.detach(), mu.detach()
        self.logvar_max = max(self.logvar_max, float(lv.max()))
        self.mu_absmax = max(self.mu_absmax, float(m.abs().max()))
        self.std_max = max(self.std_max, float(torch.exp(0.5 * lv).max()))
        self.nonfinite |= not bool(torch.isfinite(lv).all() and torch.isfinite(m).all())

    def features(self, out):
        o = out.detach()
        self.feat_max = max(self.feat_max, float(o.abs().max()))
        self.nonfinite |= not bool(torch.isfinite(o).all())

    def row(self, steps):
        return {"steps": int(steps), "logvar_max": _f(self.logvar_max), "mu_absmax": _f(self.mu_absmax),
                "std_max": _f(self.std_max), "feat_max": _f(self.feat_max), "nonfinite": self.nonfinite}


def _f(x):
    return x if math.isfinite(x) else str(x)


class _Log(BaseCallback):
    def __init__(self, tracker, rows):
        super().__init__()
        self.tracker, self.rows = tracker, rows

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        # the window closing here covers the previous update's minibatches and this rollout
        self.rows.append(self.tracker.row(self.num_timesteps))
        self.tracker.reset()


def instrument(extractor, tracker):
    orig = extractor._posterior

    def posterior(observations):
        mu, logvar = orig(observations)
        tracker.posterior(mu, extractor.clamp_logvar(logvar))
        return mu, logvar
    extractor._posterior = posterior
    extractor.register_forward_hook(lambda _m, _i, out: tracker.features(out))


def run(game, seed, timesteps, n_envs, n_steps, logvar_clamp, device):
    def make_env(rank):
        return Monitor(make_procgen_env(game, num_levels=200, distribution_mode="easy", seed=seed + rank, vector=False))
    vec = DummyVecEnv([(lambda r=rank: make_env(r)) for rank in range(n_envs)])
    kw = dict(features_dim=512, latent_dim=128)
    if logvar_clamp is not None:
        kw["logvar_clamp"] = logvar_clamp
    model = PPO("CnnPolicy", vec, verbose=0, learning_rate=3e-4, n_steps=n_steps, batch_size=64, n_epochs=3,
                gamma=0.99, gae_lambda=0.95, clip_range=0.2, seed=seed, device=device,
                policy_kwargs={"features_extractor_class": VAEExtractor, "features_extractor_kwargs": kw})
    tracker, rows = _Tracker(), []
    for ext in {id(e): e for e in (model.policy.features_extractor, model.policy.pi_features_extractor,
                                    model.policy.vf_features_extractor)}.values():
        instrument(ext, tracker)
    t0, error = time.time(), None
    try:
        model.learn(total_timesteps=timesteps, callback=_Log(tracker, rows))
    except ValueError as e:  # torch.distributions rejects NaN logits
        error = str(e).split("\n")[0][:200]
        rows.append(tracker.row(model.num_timesteps))
    vec.close()
    return {"game": game, "seed": seed, "timesteps": timesteps, "n_envs": n_envs, "n_steps": n_steps,
            "logvar_clamp": logvar_clamp, "steps_reached": int(model.num_timesteps), "error": error,
            "seconds": round(time.time() - t0, 1), "curve": rows}, model


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--game", default="bossfight")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timesteps", type=int, default=100000)
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--n-steps", type=int, default=256)
    ap.add_argument("--logvar-clamp", type=float, default=None)
    ap.add_argument("--response", action="store_true", help="attach the cross-game response measure")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    row, model = run(args.game, args.seed, args.timesteps, args.n_envs, args.n_steps, args.logvar_clamp, args.device)
    if args.response and row["error"] is None:
        from diagnose_policy_saturation import cross_game_response
        row["cross_game_response"] = cross_game_response(model, vector=False)
    last = row["curve"][-1] if row["curve"] else {}
    print(f"{args.game} s{args.seed} clamp={args.logvar_clamp}: reached {row['steps_reached']} "
          f"error={row['error']!r} last={last}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2)


if __name__ == "__main__":
    main()
