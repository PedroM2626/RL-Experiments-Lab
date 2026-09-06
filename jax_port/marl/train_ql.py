"""Loop treino VDN/QR... (VDN/QMIX) sobre SMAX.

Buffer em anel (obs, acts, rew-time, obs2, states, states2, done);
eps-greedy por agente (1.0->0.05 nos primeiros 25%); target hard-copy
a cada 500 grad-steps; lr 1e-4; batch 64; gamma .99.
Eval: greedy win-rate + retorno. Uso:
  train_ql.py --algo qmix --map 3m --timesteps 1000000 --seed 42
"""

import argparse
import copy
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_port.marl.ppo_marl import ActorOnly
from jax_port.marl.qmix import QMixer, make_ql_update
from jax_port.marl.smax_vec import SmaxVec


class MARLBuffer:
    def __init__(self, capacity, n_agents, obs_dim, state_dim):
        self.cap = capacity
        self.obs = np.empty((capacity, n_agents, obs_dim), np.float32)
        self.obs2 = np.empty((capacity, n_agents, obs_dim), np.float32)
        self.act = np.empty((capacity, n_agents), np.int32)
        self.rew = np.empty(capacity, np.float32)
        self.st = np.empty((capacity, state_dim), np.float32)
        self.st2 = np.empty((capacity, state_dim), np.float32)
        self.done = np.empty(capacity, bool)
        self.i, self.full = 0, False

    def add(self, o, a, r, o2, s, s2, d):
        n = o.shape[0]
        for k in range(n):
            j = (self.i + k) % self.cap
            self.obs[j], self.act[j], self.rew[j] = o[k], a[k], r[k]
            self.obs2[j], self.st[j] = o2[k], s[k]
            self.st2[j], self.done[j] = s2[k], d[k]
        self.i = (self.i + n) % self.cap
        self.full = self.full or self.i == 0

    def __len__(self):
        return self.cap if self.full else self.i

    def sample(self, rng, batch):
        idx = rng.integers(0, len(self), size=batch)
        return (self.obs[idx], self.act[idx], self.rew[idx],
                self.obs2[idx], self.st[idx], self.st2[idx], self.done[idx])

    def sample_seq(self, rng, batch, L):
        # blocos consecutivos (wrap modulo = aproximacao documentada).
        # obs/st com L+1 (o_t..o_{t+L}); act/rew/done com L.
        lim = max(L + 2, len(self))
        s0 = rng.integers(0, lim - L - 1, size=batch)
        ii = (s0[:, None] + np.arange(L + 1)[None, :]) % self.cap
        return (self.obs[ii], self.act[ii[:, :L]], self.rew[ii[:, :L]],
                self.st[ii], self.done[ii[:, :L]])


def train_recurrent_ql(args):
    """VDN/QMIX recorrente (GRU-128): rollout com carry, updates em
    sequencias L=32 (h0=zero, aproximacao documentada), resto identico."""
    from jax_port.marl.recurrent import REC_H, RecurrentQ, make_ql_seq_update
    jax.config.update("jax_compilation_cache_dir",
                      os.environ.get("JAX_PORT_CACHE", "/tmp/jax_port_cache"))
    rng = np.random.default_rng(args.seed)
    key = jax.random.PRNGKey(args.seed)
    device = jax.devices()[0]
    N, L = args.num_envs, 32
    print(f"marl {args.algo}-recurrent {args.map} device={device}", flush=True)
    venv = SmaxVec(args.map, num_envs=N, seed=args.seed)
    A, NA = venv.n_agents, venv.n_actions
    qnet = RecurrentQ(n_actions=NA)
    mixer = QMixer(n_agents=A)
    key, k0, k1 = jax.random.split(key, 3)
    params = {"q": qnet.init(k0, jnp.zeros((1, L, venv.obs_dim)),
                             jnp.zeros((1, REC_H)), jnp.zeros((1, L), bool)),
              "mix": mixer.init(k1, jnp.zeros((1, A)),
                                jnp.zeros((1, venv.state_dim)))}
    tgt = copy.deepcopy(params)
    opt = optax.chain(optax.clip_by_global_norm(10.0),
                      optax.adam(getattr(args, "lr", 5e-5)))
    opt_state = opt.init(params)
    update = make_ql_seq_update(qnet, mixer, opt, kind=args.algo)

    @jax.jit
    def greedy_seq(p, obs_seq, h0, dones):
        q, h1 = qnet.apply(p["q"], obs_seq, h0, dones)
        return q.argmax(-1), h1

    key, kw = jax.random.split(key)
    params, opt_state, _ = update(
        params, opt_state, tgt, jnp.zeros((2, L, A, venv.obs_dim)),
        jnp.zeros((2, L, A), jnp.int32), jnp.zeros((2, L)),
        jnp.zeros((2, L, A, venv.obs_dim)),
        jnp.zeros((2, L, venv.state_dim)),
        jnp.zeros((2, L, venv.state_dim)), jnp.zeros((2, L), bool))
    jax.block_until_ready(jax.tree_util.tree_leaves(params)[0])

    buf = MARLBuffer(100000, A, venv.obs_dim, venv.state_dim)
    carry = np.zeros((N * A, REC_H), np.float32)
    steps, grads = 0, 0
    ep_wins, ep_rets, cur, curve = [], [], np.zeros(N), []
    eps_end = max(1, args.timesteps // 4)
    t0 = time.perf_counter()
    while steps < args.timesteps:
        eps = max(0.05, 1.0 - 0.95 * steps / eps_end)
        ob = np.asarray(venv.obs_batch())
        if steps < 5000 or rng.random() < eps:
            act = rng.integers(0, NA, size=(N, A)).astype(np.int32)
        else:
            q1, carry = greedy_seq(
                params, jnp.asarray(ob.reshape(1, N * A, -1)),
                jnp.asarray(carry), jnp.zeros((1, N * A), bool))
            jax.block_until_ready(q1)
            carry = np.array(carry)
            act = np.asarray(q1[0]).reshape(N, A)
        _, rew, done, win = venv.step(act)
        ob2 = np.asarray(venv.obs_batch())
        buf.add(ob, act, rew, ob2, np.asarray(venv.state_batch()),
                np.asarray(venv.state_batch()), done)
        carry[np.repeat(done, A)] = 0.0
        cur += rew
        for i in np.where(done)[0]:
            ep_wins.append(bool(win[i]))
            ep_rets.append(float(cur[i]))
            cur[i] = 0.0
        steps += N
        if len(buf) >= 5000:
            for _ in range(max(1, N // 4)):
                bo, ba, br, bst, bd = buf.sample_seq(rng, 2, L)
                params, opt_state, _ = update(
                    params, opt_state, tgt,
                    jnp.asarray(bo[:, :-1]), jnp.asarray(ba), jnp.asarray(br),
                    jnp.asarray(bo[:, 1:]),
                    jnp.asarray(bst[:, :-1]), jnp.asarray(bst[:, 1:]),
                    jnp.asarray(bd))
                grads += 1
                if grads % 500 == 0:
                    tgt = copy.deepcopy(params)
        if steps % (N * 20) == 0:
            el = time.perf_counter() - t0
            wr = float(np.mean(ep_wins[-20:])) if ep_wins else 0.0
            curve.append({"steps": steps, "winrate": wr})
            print(f"steps={steps} sps={steps/el:.0f} eps={eps:.2f} "
                  f"win20={wr:.2f} buf={len(buf)}", flush=True)
    dt = time.perf_counter() - t0
    out = {"algo": args.algo + "-recurrent", "map": args.map, "seed": args.seed,
           "timesteps": steps, "wall_s": round(dt, 1),
           "sps": round(steps / dt, 1),
           "train_episodes": len(ep_wins),
           "winrate_train": round(float(np.mean(ep_wins)), 3) if ep_wins else 0.0,
           "curve": curve}
    if args.eval_eps > 0:
        out["eval"] = evaluate_recurrent_ql(params, qnet, args)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "curve"}, indent=2))
    return out


def evaluate_recurrent_ql(params, qnet, args):
    """Eval greedy com carry (win-rate padrao SMAC)."""
    from jax_port.marl.recurrent import REC_H
    from jax_port.marl.smax_vec import SmaxVec
    ev = SmaxVec(args.map, num_envs=args.eval_envs, seed=args.seed + 1000)
    E, A = args.eval_envs, ev.n_agents
    carry = np.zeros((E * A, REC_H), np.float32)
    wins, rets = [], []
    cur = np.zeros(E)

    @jax.jit
    def esteps(p, ob, carry_):
        q, new_carry = qnet.apply(p["q"], ob[None], carry_,
                                  jnp.zeros((1, E * A), bool))
        # carry sem dim de tempo: retorna (E*A,H) inteiro.
        return q[0].argmax(-1), new_carry

    while len(wins) < args.eval_eps:
        ob = np.asarray(ev.obs_batch()).reshape(E * A, -1)
        act, carry = esteps(params, jnp.asarray(ob), jnp.asarray(carry))
        jax.block_until_ready((act, carry))
        carry = np.asarray(carry)
        act = np.asarray(act).reshape(E, A)
        _, rew, done, win = ev.step(act)
        cur += rew
        for i in np.where(done)[0]:
            wins.append(bool(win[i]))
            rets.append(float(cur[i]))
            cur[i] = 0.0
        carry = np.array(carry)
        carry[np.repeat(done, A)] = 0.0
    return {"winrate": round(float(np.mean(wins)), 3), "episodes": len(wins),
            "mean_return": round(float(np.mean(rets)), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="vdn", choices=["vdn", "qmix"])
    ap.add_argument("--map", default="3m")
    ap.add_argument("--timesteps", type=int, default=1000000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--recurrent", action="store_true",
                    help="GRU-128 (padrao JaxMARL p/ SMAX)")
    ap.add_argument("--num-envs", type=int, default=32)
    ap.add_argument("--eval-eps", type=int, default=32)
    ap.add_argument("--eval-envs", type=int, default=8)
    ap.add_argument("--out", default="jax_port/marl_ql.json")
    train(ap.parse_args())


def train(args):
    jax.config.update("jax_compilation_cache_dir",
                      os.environ.get("JAX_PORT_CACHE", "/tmp/jax_port_cache"))
    if getattr(args, "recurrent", False):
        return train_recurrent_ql(args)
    rng = np.random.default_rng(args.seed)
    key = jax.random.PRNGKey(args.seed)
    device = jax.devices()[0]
    N = args.num_envs
    print(f"marl {args.algo} {args.map} device={device}", flush=True)
    venv = SmaxVec(args.map, num_envs=N, seed=args.seed)
    A, NA = venv.n_agents, venv.n_actions
    qnet = ActorOnly(n_actions=NA)
    mixer = QMixer(n_agents=A)
    key, k0, k1 = jax.random.split(key, 3)
    params = {"q": qnet.init(k0, jnp.zeros((1, venv.obs_dim))),
              "mix": mixer.init(k1, jnp.zeros((1, A)),
                                jnp.zeros((1, venv.state_dim)))}
    tgt = copy.deepcopy(params)
    opt = optax.chain(optax.clip_by_global_norm(10.0),
                      optax.adam(getattr(args, "lr", 1e-4)))
    opt_state = opt.init(params)
    update = make_ql_update(qnet, mixer, opt, kind=args.algo)

    @jax.jit
    def greedy(p, obs):
        return qnet.apply(p["q"], obs).argmax(-1)

    update(params, opt_state, tgt, jnp.zeros((64, A, venv.obs_dim)),
           jnp.zeros((64, A), jnp.int32), jnp.zeros((64,)),
           jnp.zeros((64, A, venv.obs_dim)), jnp.zeros((64, 72)),
           jnp.zeros((64, 72)), jnp.zeros((64,)))
    greedy(params, jnp.zeros((N * A, venv.obs_dim)))
    jax.block_until_ready(jax.tree_util.tree_leaves(params)[0])

    buf = MARLBuffer(100000, A, venv.obs_dim, venv.state_dim)
    steps, grads = 0, 0
    ep_wins, ep_rets, cur, curve = [], [], np.zeros(N), []
    eps_end = max(1, args.timesteps // 4)
    t0 = time.perf_counter()
    while steps < args.timesteps:
        eps = max(0.05, 1.0 - 0.95 * steps / eps_end)
        ob = np.asarray(venv.obs_batch())
        sb = np.asarray(venv.state_batch())
        if steps < 5000 or rng.random() < eps:
            act = rng.integers(0, NA, size=(N, A)).astype(np.int32)
        else:
            act = np.asarray(greedy(
                params, jnp.asarray(ob.reshape(N * A, -1)))).reshape(N, A)
        _, rew, done, win = venv.step(act)
        ob2 = np.asarray(venv.obs_batch())
        sb2 = np.asarray(venv.state_batch())
        buf.add(ob, act, rew, ob2, sb, sb2, done)
        cur += rew
        for i in np.where(done)[0]:
            ep_wins.append(bool(win[i]))
            ep_rets.append(float(cur[i]))
            cur[i] = 0.0
        steps += N
        if len(buf) >= 5000:
            for _ in range(max(1, N // 4)):
                bo, ba, br, bo2, bs, bs2, bd = buf.sample(rng, 64)
                params, opt_state, _ = update(
                    params, opt_state, tgt,
                    jnp.asarray(bo), jnp.asarray(ba), jnp.asarray(br),
                    jnp.asarray(bo2), jnp.asarray(bs), jnp.asarray(bs2),
                    jnp.asarray(bd))
                grads += 1
                if grads % 500 == 0:
                    tgt = copy.deepcopy(params)
        if steps % (N * 20) == 0:
            el = time.perf_counter() - t0
            wr = float(np.mean(ep_wins[-20:])) if ep_wins else 0.0
            curve.append({"steps": steps, "winrate": wr})
            print(f"steps={steps} sps={steps/el:.0f} eps={eps:.2f} "
                  f"win20={wr:.2f} buf={len(buf)}", flush=True)
    dt = time.perf_counter() - t0
    out = {"algo": args.algo, "map": args.map, "seed": args.seed,
           "timesteps": steps, "wall_s": round(dt, 1),
           "sps": round(steps / dt, 1),
           "train_episodes": len(ep_wins),
           "winrate_train": round(float(np.mean(ep_wins)), 3) if ep_wins else 0.0,
           "curve": curve}
    if args.eval_eps > 0:
        out["eval"] = evaluate(params, greedy, args, venv)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "curve"}, indent=2))
    return out


def evaluate(params, greedy, args, venv):
    from jax_port.marl.smax_vec import SmaxVec
    ev = SmaxVec(args.map, num_envs=args.eval_envs, seed=args.seed + 1000)
    wins, rets = [], []
    cur = np.zeros(args.eval_envs)
    while len(wins) < args.eval_eps:
        ob = np.asarray(ev.obs_batch())
        act = np.asarray(greedy(params, jnp.asarray(
            ob.reshape(-1, ob.shape[-1])))).reshape(args.eval_envs, -1)
        _, rew, done, win = ev.step(act)
        cur += rew
        for i in np.where(done)[0]:
            wins.append(bool(win[i]))
            rets.append(float(cur[i]))
            cur[i] = 0.0
    return {"winrate": round(float(np.mean(wins)), 3), "episodes": len(wins),
            "mean_return": round(float(np.mean(rets)), 2)}


if __name__ == "__main__":
    main()
