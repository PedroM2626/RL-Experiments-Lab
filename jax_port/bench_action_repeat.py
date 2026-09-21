"""Benchmark of Action-Repeat Granularity (k in {1, 2, 4, 8}) in JAX.

Isolates temporal decision frequency / inertia from hierarchical abstraction.
- Games: jumper and plunder
- Frame budget: strictly 100,000 primitive environment frames
- Arms: repeat_k1 (flat), repeat_k2, repeat_k4, repeat_k8
- All 15 primitive actions available; no fixed/handcrafted skill library.
- Evaluation: unseen levels (seed+1000) stochastic and deterministic.
"""

import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from procgen import ProcgenGym3Env

from jax_port.backbones import BACKBONES
from jax_port.networks import ActorCritic, preprocess
from jax_port.ppo import compute_gae, make_optimizer, make_update_fn


class VariableRepeatGym3:
    """N gym3 envs with action-repeat of exactly `dur` frames over all 15 actions."""

    def __init__(self, game, num_envs, num_levels, seed, dur=4):
        self.env = ProcgenGym3Env(
            num=num_envs,
            env_name=game,
            num_levels=num_levels,
            distribution_mode="easy",
            rand_seed=seed,
        )
        self.num_envs = num_envs
        self.dur = max(1, int(dur))
        _, d, _ = self.env.observe()
        self.obs = d["rgb"] if isinstance(d, dict) else d

    def step_repeat(self, actions):
        """actions: (N,) primitive actions in [0..14].
        Repeats the action up to `dur` frames. Macro terminates early on done.
        Returns: (obs, rew_sum, any_done, frames_consumed)
        """
        N = self.num_envs
        tot = np.zeros(N, np.float32)
        any_done = np.zeros(N, bool)
        active = np.ones(N, bool)
        frames_consumed = 0

        for _ in range(self.dur):
            if not active.any():
                break
            self.env.act(np.where(active, actions, 0))
            rew_d, d, first_d = self.env.observe()
            self.obs = d["rgb"] if isinstance(d, dict) else d
            rew = np.asarray(rew_d, np.float32)
            first = np.asarray(first_d, bool)
            tot += np.where(active, rew, 0.0)
            any_done |= (active & first)
            active &= ~first
            frames_consumed += int(active.sum()) + int(first.sum())

        return self.obs, tot, any_done, frames_consumed


def train_action_repeat(game, k, seed, target_frames=100_000, num_envs=64, rollout=128, minibatch=1024):
    device = jax.devices()[0]
    key = jax.random.PRNGKey(seed)
    rng = np.random.default_rng(seed)

    env = VariableRepeatGym3(game, num_envs, num_levels=200, seed=seed, dur=k)
    model = ActorCritic(backbone=BACKBONES["classic"](), n_actions=15)

    key, k0 = jax.random.split(key)
    dummy_obs = jnp.zeros((1, 64, 64, 3), jnp.float32)
    params = model.init(k0, dummy_obs, None)
    opt = make_optimizer(lr=3e-4)
    state = (params, opt.init(params))
    update_fn, rollout_fn, forward_fn, _ = make_update_fn(model, opt)

    N, T = num_envs, rollout
    frames = 0
    t0 = time.perf_counter()
    ep_rets = []
    cur_ret = np.zeros(N, np.float32)

    while frames < target_frames:
        b_obs = np.empty((T, N, 64, 64, 3), np.uint8)
        b_act = np.empty((T, N), np.int32)
        b_rew = np.empty((T, N), np.float32)
        b_done = np.empty((T, N), bool)
        b_val = np.empty((T, N), np.float32)
        b_logp = np.empty((T, N), np.float32)

        for t in range(T):
            if frames >= target_frames:
                break
            pin = jnp.asarray(env.obs, device=device)
            act_d, logp_d, val_d, key = rollout_fn(state[0], pin, key)
            jax.block_until_ready((act_d, logp_d, val_d))

            act = np.asarray(act_d)
            b_obs[t], b_act[t] = env.obs, act
            b_val[t], b_logp[t] = np.asarray(val_d), np.asarray(logp_d)

            obs, rew, done, fr = env.step_repeat(act)
            b_rew[t], b_done[t] = rew, done
            frames += fr

            cur_ret += rew
            for i in np.where(done)[0]:
                ep_rets.append(float(cur_ret[i]))
                cur_ret[i] = 0.0

        cut = t + 1 if frames >= target_frames and t < T - 1 else T
        ob = preprocess(jnp.asarray(env.obs, device=device))
        key, kf = jax.random.split(key)
        _, last_v = forward_fn(state[0], ob, kf)

        adv, ret = compute_gae(b_rew[:cut], b_val[:cut], b_done[:cut], np.asarray(last_v), gamma=0.99, lam=0.95)

        flat = lambda a: a.reshape(-1, *a.shape[2:])
        F_obs = flat(b_obs[:cut])
        F_act = flat(b_act[:cut])
        F_logp = flat(b_logp[:cut])
        F_adv = flat(adv)
        F_ret = flat(ret)

        adv_n = (F_adv - F_adv.mean()) / (F_adv.std() + 1e-8)
        M = len(F_obs)

        for _ in range(3):
            perm = rng.permutation(M)
            for s in range(0, M, minibatch):
                mb = perm[s : s + minibatch]
                if len(mb) < 32:
                    continue
                key, ku = jax.random.split(key)
                state, _ = update_fn(
                    state,
                    jnp.asarray(F_obs[mb], device=device),
                    jnp.asarray(F_act[mb], device=device),
                    jnp.asarray(F_logp[mb], device=device),
                    jnp.asarray(adv_n[mb], device=device),
                    jnp.asarray(F_ret[mb], device=device),
                    ku,
                )

    wall = time.perf_counter() - t0

    # Evaluation on unseen levels
    eval_env = VariableRepeatGym3(game, num_envs=16, num_levels=0, seed=seed + 1000, dur=k)
    stoch_rets = evaluate(eval_env, state[0], rollout_fn, n_episodes=30, deterministic=False)
    eval_env_det = VariableRepeatGym3(game, num_envs=16, num_levels=0, seed=seed + 1000, dur=k)
    det_rets = evaluate(eval_env_det, state[0], forward_fn, n_episodes=30, deterministic=True)

    return {
        "game": game,
        "k": k,
        "seed": seed,
        "frames": frames,
        "wall_sec": round(wall, 2),
        "train_mean": float(np.mean(ep_rets[-20:])) if ep_rets else 0.0,
        "stoch_unseen": float(np.mean(stoch_rets)),
        "det_unseen": float(np.mean(det_rets)),
    }


def evaluate(env, params, fwd_fn, n_episodes=30, deterministic=False):
    device = jax.devices()[0]
    key = jax.random.PRNGKey(1234)
    N = env.num_envs
    rets = []
    cur = np.zeros(N, np.float32)

    while len(rets) < n_episodes:
        pin = preprocess(jnp.asarray(env.obs, device=device)) if deterministic else jnp.asarray(env.obs, device=device)
        key, k = jax.random.split(key)
        if deterministic:
            logits, _ = fwd_fn(params, pin, k)
            act = np.asarray(logits.argmax(-1))
        else:
            act_d, _, _, key = fwd_fn(params, pin, key)
            act = np.asarray(act_d)

        _, rew, done, _ = env.step_repeat(act)
        cur += rew
        for i in np.where(done)[0]:
            if len(rets) < n_episodes:
                rets.append(float(cur[i]))
            cur[i] = 0.0

    return rets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", nargs="+", default=["jumper", "plunder"])
    parser.add_argument("--skips", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--frames", type=int, default=100_000)
    parser.add_argument("--out", default="results/action_repeat_results.json")
    args = parser.parse_args()

    results = {}
    print(f"Executing Action-Repeat Sweep: games={args.games}, skips={args.skips}, seeds={args.seeds}", flush=True)

    for game in args.games:
        for k in args.skips:
            res_list = []
            for s in args.seeds:
                res = train_action_repeat(game, k, s, target_frames=args.frames)
                res_list.append(res)
                print(f"  [{game}] k={k} seed={s} -> stoch_unseen={res['stoch_unseen']:.2f}, det={res['det_unseen']:.2f} ({res['wall_sec']}s)", flush=True)
            
            stochs = [r["stoch_unseen"] for r in res_list]
            dets = [r["det_unseen"] for r in res_list]
            key_name = f"{game}_k{k}"
            results[key_name] = {
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

    print(f"\nAction-Repeat sweep completed and saved to {args.out}")


if __name__ == "__main__":
    main()
