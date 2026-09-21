"""Benchmark Pareado Temporal em Stack=1 (Starpilot, 100k steps, seed 42).

Compara:
1. ClassicCNN (NatureCNN feedforward, sem memoria, stack=1)
2. RecurrentLSTM (ClassicCNN + LSTMCell com carry inter-passos no rollout e reset no done)
3. RecurrentS5 (ClassicCNN + 2x S5 step layers com carry inter-passos no rollout e reset no done)

Todos pareados sob os mesmos hiperparametros PPO do estudo:
lr=3e-4, gamma=0.99, lambda=0.95, clip=0.2, epochs=3, num_envs=64, rollout=128, minibatch=1024.
"""

import argparse
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from procgen import ProcgenGym3Env

from jax_port.backbones import ClassicCNN
from jax_port.networks import ActorCritic, ActorCriticXL, preprocess
from jax_port.ppo import compute_gae, make_optimizer, make_update_fn
from jax_port.recurrent_step import RECURRENT_STEP
from jax_port.temporal import make_mem_fns


def get_obs(obs_d):
    return np.asarray(obs_d["rgb"])  # (N, 64, 64, 3) uint8


def run_arm(model_name, args, seed=42):
    print(f"\n{'='*70}\nStarting branch: {model_name} (stack=1, game={args.game})\n{'='*70}", flush=True)
    device = jax.devices()[0]
    key = jax.random.PRNGKey(seed)
    rng = np.random.RandomState(seed)

    env = ProcgenGym3Env(
        num=args.num_envs,
        env_name=args.game,
        num_levels=200,
        distribution_mode=args.distribution,
        rand_seed=seed,
    )

    n_actions = 15
    mem_mode = model_name in RECURRENT_STEP
    mem_shape = RECURRENT_STEP[model_name][1] if mem_mode else (0, 0)
    L, D = mem_shape

    if mem_mode:
        backbone_cls, _ = RECURRENT_STEP[model_name]
        model = ActorCriticXL(backbone=backbone_cls(), n_actions=n_actions)
    else:
        model = ActorCritic(backbone=ClassicCNN(), n_actions=n_actions)

    key, k0 = jax.random.split(key)
    dummy_obs = jnp.zeros((1, 64, 64, 3), jnp.float32)
    if mem_mode:
        dummy_mem = jnp.zeros((1, L, D), jnp.float32)
        params = model.init(k0, dummy_obs, dummy_mem)
    else:
        params = model.init(k0, dummy_obs, None)

    opt = make_optimizer(lr=3e-4, kind="adam")
    opt_state = opt.init(params)
    state = (params, opt_state)

    if mem_mode:
        update_fn, rollout_fn, forward_fn = make_mem_fns(model, opt)
    else:
        update_fn, rollout_fn, forward_fn, _ = make_update_fn(model, opt)

    N, T = args.num_envs, args.rollout
    mem0 = jnp.zeros((N, L, D), jnp.float32) if mem_mode else None

    # Warmup JIT
    key, kw, ku = jax.random.split(key, 3)
    if mem_mode:
        (state, _) = update_fn(
            state,
            jnp.zeros((args.minibatch, 64, 64, 3), jnp.uint8),
            jnp.zeros((args.minibatch, L, D), jnp.float32),
            jnp.zeros((args.minibatch,), jnp.int32),
            jnp.zeros((args.minibatch,), jnp.float32),
            jnp.zeros((args.minibatch,), jnp.float32),
            jnp.zeros((args.minibatch,), jnp.float32),
        )
        _, _, _, _, key = rollout_fn(state[0], jnp.zeros((N, 64, 64, 3), jnp.uint8), mem0, kw)
    else:
        (state, _) = update_fn(
            state,
            jnp.zeros((args.minibatch, 64, 64, 3), jnp.uint8),
            jnp.zeros((args.minibatch,), jnp.int32),
            jnp.zeros((args.minibatch,), jnp.float32),
            jnp.zeros((args.minibatch,), jnp.float32),
            jnp.zeros((args.minibatch,), jnp.float32),
            ku,
        )
        _, _, _, key = rollout_fn(state[0], jnp.zeros((N, 64, 64, 3), jnp.uint8), kw)
    jax.block_until_ready(jax.tree_util.tree_leaves(state[0])[0])

    _, obs_d, _ = env.observe()
    obs = get_obs(obs_d)

    n_iters = max(1, (args.timesteps + N * T - 1) // (N * T))
    done_steps = 0
    ep_rets, ep_lens = [], []
    cur_ret, cur_len = np.zeros(N), np.zeros(N)
    curve = []
    t0 = time.perf_counter()

    for it in range(n_iters):
        b_obs = np.empty((T, N, 64, 64, 3), np.uint8)
        b_mem = np.zeros((T, N, L, D), np.float32) if mem_mode else None
        mems = np.zeros((N, L, D), np.float32) if mem_mode else None
        b_act = np.empty((T, N), np.int32)
        b_rew = np.empty((T, N), np.float32)
        b_done = np.empty((T, N), bool)
        b_val = np.empty((T, N), np.float32)
        b_logp = np.empty((T, N), np.float32)

        for t in range(T):
            pin = jnp.asarray(obs, device=device)
            if mem_mode:
                act_d, logp_d, val_d, mem_d, key = rollout_fn(
                    state[0], pin, jnp.asarray(mems, device=device), key
                )
                jax.block_until_ready((act_d, logp_d, val_d, mem_d))
                mems = np.array(mem_d)
                b_mem[t] = mems
            else:
                act_d, logp_d, val_d, key = rollout_fn(state[0], pin, key)
                jax.block_until_ready((act_d, logp_d, val_d))

            act = np.asarray(act_d)
            logp = np.asarray(logp_d)
            value = np.asarray(val_d)

            b_obs[t], b_act[t] = obs, act
            b_val[t], b_logp[t] = value, logp

            env.act(act)
            rew, obs_d, first = env.observe()
            obs = get_obs(obs_d)
            b_rew[t] = np.asarray(rew, np.float32)
            b_done[t] = np.asarray(first)

            cur_ret += b_rew[t]
            cur_len += 1

            for i in np.where(b_done[t])[0]:
                ep_rets.append(float(cur_ret[i]))
                ep_lens.append(int(cur_len[i]))
                cur_ret[i], cur_len[i] = 0.0, 0

            if mem_mode:
                mems[b_done[t]] = 0.0

        ob_last = preprocess(jnp.asarray(obs, device=device))
        key, kf = jax.random.split(key)
        if mem_mode:
            _, last_v, _ = forward_fn(state[0], ob_last, jnp.asarray(mems, device=device))
        else:
            _, last_v = forward_fn(state[0], ob_last, kf)

        adv, ret = compute_gae(b_rew, b_val, b_done, np.asarray(last_v), gamma=0.99, lam=0.95)

        flat = lambda a: a.reshape(T * N, *a.shape[2:])
        F = (
            flat(b_obs),
            flat(b_act).astype(np.int32),
            flat(b_logp),
            flat(adv),
            flat(ret).astype(np.float32),
        )
        idx = rng.permutation(T * N)
        adv_n = F[3]
        adv_n = (adv_n - adv_n.mean()) / (adv_n.std() + 1e-8)

        for _ep in range(3):  # 3 epochs
            for s in range(0, T * N, args.minibatch):
                mb = idx[s : s + args.minibatch]
                mo = jnp.asarray(F[0][mb], device=device)
                if mem_mode:
                    state, _ = update_fn(
                        state,
                        mo,
                        jnp.asarray(b_mem.reshape(T * N, L, D)[mb], device=device),
                        jnp.asarray(F[1][mb], device=device),
                        jnp.asarray(F[2][mb], device=device),
                        jnp.asarray(adv_n[mb], device=device),
                        jnp.asarray(F[4][mb], device=device),
                    )
                else:
                    key, kz = jax.random.split(key)
                    state, _ = update_fn(
                        state,
                        mo,
                        jnp.asarray(F[1][mb], device=device),
                        jnp.asarray(F[2][mb], device=device),
                        jnp.asarray(adv_n[mb], device=device),
                        jnp.asarray(F[4][mb], device=device),
                        kz,
                    )

        done_steps += T * N
        if (it + 1) % 5 == 0 or done_steps >= args.timesteps:
            el = time.perf_counter() - t0
            mr = float(np.mean(ep_rets[-20:])) if ep_rets else 0.0
            curve.append({"steps": done_steps, "ret20": mr})
            print(
                f"iter={it+1} steps={done_steps} sps={done_steps/el:.0f} train_ret20={mr:.2f} eps={len(ep_rets)}",
                flush=True,
            )

    wall_s = time.perf_counter() - t0
    final_sps = done_steps / wall_s

    # Avaliacao em niveis unseen
    print(f"Avaliando {model_name} em niveis unseen ({args.eval_eps} eps)...", flush=True)
    ev_env = ProcgenGym3Env(
        num=args.eval_envs,
        env_name=args.game,
        num_levels=0,
        distribution_mode=args.distribution,
        rand_seed=seed + 1000,
    )
    _, ev_obs_d, _ = ev_env.observe()
    ev_obs = get_obs(ev_obs_d)
    ev_mems = np.zeros((args.eval_envs, L, D), np.float32) if mem_mode else None
    ev_rets = np.zeros(args.eval_envs)
    all_eval_rets = []

    while len(all_eval_rets) < args.eval_eps:
        ob = preprocess(jnp.asarray(ev_obs, device=device))
        key, kf = jax.random.split(key)
        if mem_mode:
            logits, _, new_m = forward_fn(state[0], ob, jnp.asarray(ev_mems, device=device))
            jax.block_until_ready(logits)
            ev_mems = np.array(new_m)
        else:
            logits, _ = forward_fn(state[0], ob, kf)
            jax.block_until_ready(logits)

        key, ks = jax.random.split(key)
        act = np.asarray(jax.random.categorical(ks, logits))
        ev_env.act(act)
        rew, ev_obs_d, first = ev_env.observe()
        ev_obs = get_obs(ev_obs_d)
        fin = np.asarray(first)
        ev_rets += np.asarray(rew)

        for i in np.where(fin)[0]:
            all_eval_rets.append(float(ev_rets[i]))
            ev_rets[i] = 0.0

        if mem_mode:
            ev_mems[fin] = 0.0

    eval_scores = all_eval_rets[: args.eval_eps]
    eval_mean = float(np.mean(eval_scores))
    eval_std = float(np.std(eval_scores, ddof=1)) if len(eval_scores) > 1 else 0.0
    se = eval_std / np.sqrt(len(eval_scores))
    ci95 = (float(eval_mean - 1.96 * se), float(eval_mean + 1.96 * se))

    result = {
        "model": model_name,
        "timesteps": done_steps,
        "wall_s": round(wall_s, 2),
        "sps": round(final_sps, 1),
        "train_ret_mean20": round(float(np.mean(ep_rets[-20:])), 2) if ep_rets else 0.0,
        "eval_unseen_mean": round(eval_mean, 2),
        "eval_unseen_std": round(eval_std, 2),
        "eval_unseen_ci95": [round(ci95[0], 2), round(ci95[1], 2)],
        "eval_eps": len(eval_scores),
    }
    print(f"Resultado {model_name}: eval_unseen={result['eval_unseen_mean']} CI95={result['eval_unseen_ci95']} SPS={result['sps']}\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, default="starpilot")
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--rollout", type=int, default=128)
    parser.add_argument("--minibatch", type=int, default=1024)
    parser.add_argument("--eval-envs", type=int, default=16)
    parser.add_argument("--eval-eps", type=int, default=20)
    parser.add_argument("--distribution", type=str, default="easy")
    parser.add_argument("--output", type=str, default="/mnt/c/Users/Acer/Downloads/MLE/jax_port/results_stack1_bench.json")
    args = parser.parse_args()

    models = ["classic", "recurrent_lstm", "recurrent_s5"]
    results = []

    for m in models:
        res = run_arm(m, args, seed=args.seed)
        results.append(res)

    print("\n" + "=" * 80)
    print(f"COMPARATIVE SUMMARY — STACK=1 IN {args.game.upper()} (100k steps, seed {args.seed})")
    print("=" * 80)
    print(f"{'Model':<16} {'SPS':>8} {'Wall(s)':>8} {'Train(last20)':>14} {'Eval Unseen':>12} {'95% CI':>16}")
    print("-" * 80)
    for r in results:
        ci_str = f"[{r['eval_unseen_ci95'][0]:.2f}, {r['eval_unseen_ci95'][1]:.2f}]"
        print(f"{r['model']:<16} {r['sps']:>8.0f} {r['wall_s']:>8.1f} {r['train_ret_mean20']:>14.2f} {r['eval_unseen_mean']:>12.2f} {ci_str:>16}")
    print("=" * 80)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved in: {args.output}")


if __name__ == "__main__":
    main()
