"""Action-Repeat Sweep (k in {1, 2, 4, 8}) in PyTorch / SB3.

Isolates temporal granularity (frame skip) from hierarchical abstraction on jumper and plunder.
Budget: 100,000 primitive environment frames (100k // k decisions).
"""

import argparse
import json
import os
import gymnasium as gymn
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from procgen_wrapper import make_procgen_env

FRAMES = 100_000


class VariableRepeatEnv(gymn.ActionWrapper):
    def __init__(self, env, k):
        super().__init__(env)
        self.k = max(1, int(k))

    def action(self, a):
        return a

    def step(self, a):
        tot, term, trunc, obs, info = 0.0, False, False, None, {}
        for _ in range(self.k):
            obs, r, term, trunc, info = self.env.step(a)
            tot += r
            if term or trunc:
                break
        return obs, tot, term, trunc, info


def make_repeat_env(game, num_levels, seed, k):
    env = make_procgen_env(game, num_levels=num_levels, distribution_mode="easy", seed=seed, vector=False)
    if k > 1:
        return VariableRepeatEnv(env, k)
    return env


def train_repeat(game, k, seed, device="auto"):
    vec = DummyVecEnv([lambda: Monitor(make_repeat_env(game, 200, seed, k))])
    model = PPO(
        "CnnPolicy",
        vec,
        verbose=0,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        seed=seed,
        device=device,
    )
    ts = FRAMES // k
    model.learn(total_timesteps=ts)
    vec.close()
    return model


def eval_repeat(model, game, k, seed, n_unseen=30):
    unseen = DummyVecEnv([lambda: Monitor(make_repeat_env(game, 0, seed + 1000, k))])
    m_st, _ = evaluate_policy(model, unseen, n_eval_episodes=n_unseen, deterministic=False)
    m_dt, _ = evaluate_policy(model, unseen, n_eval_episodes=n_unseen, deterministic=True)
    unseen.close()
    return float(m_st), float(m_dt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", nargs="+", default=["jumper", "plunder"])
    parser.add_argument("--skips", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="results/action_repeat_sb3_results.json")
    args = parser.parse_args()

    results = {}
    for game in args.games:
        for k in args.skips:
            res_list = []
            for s in args.seeds:
                model = train_repeat(game, k, s, device=args.device)
                stoch, det = eval_repeat(model, game, k, s)
                res_list.append({"seed": s, "stoch": stoch, "det": det})
                print(f"[{game}] k={k} seed={s} -> stoch={stoch:.2f}, det={det:.2f}", flush=True)

            stochs = [r["stoch"] for r in res_list]
            dets = [r["det"] for r in res_list]
            results[f"{game}_k{k}"] = {
                "game": game,
                "k": k,
                "stoch_mean": float(np.mean(stochs)),
                "stoch_std": float(np.std(stochs)),
                "det_mean": float(np.mean(dets)),
                "runs": res_list,
            }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
