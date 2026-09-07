"""IPPO + MAPPO sobre SMAX.

IPPO: ActorCritic compartilhado (MLP sobre obs), PPO sobre o pool
  (T,N,A) com team-reward + done broadcast; GAE por (env,agente).
  Reusa update/rollout/forward de ppo.make_update_fn em batches flat.
MAPPO: actor (obs) + critico centralizado (world_state); vantagens do
  critico central, loss PPO no actor (custom, ~60 linhas).
Hparams do projeto: lr 3e-4, gamma .99, lambda .95, clip .2, 3 epochs,
  vf .5, ent .01, adv-norm por epoca.
Metrica: win-rate greedy + retorno (padrao SMAC), multi-seed, JSON.
Uso: train_marl.py --algo ippo --map 3m --timesteps 1000000
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import json as _json
import numpy as np
import optax
import time as _time

from flax.serialization import from_bytes, to_bytes

from jax_port.backbones import MlpBackbone
from jax_port.marl.recurrent import RecurrentAC, make_ppo_seq_update
from jax_port.marl.smax_vec import SmaxVec
from jax_port.networks import ActorCritic
from jax_port.ppo import compute_gae, make_optimizer


def make_update_float(model, optimizer, clip_range=0.2, vf_coef=0.5,
                      ent_coef=0.01):
    """Irmao float de ppo.make_update_fn (sem /255): obs vetoriais SMAX."""

    def loss_fn(params, obs, act, old_logp, adv, ret):
        logits, value = model.apply(params, obs, None)
        logp_all = jax.nn.log_softmax(logits)
        logp = jnp.take_along_axis(logp_all, act[:, None], 1).squeeze(1)
        ratio = jnp.exp(logp - old_logp)
        pg = -jnp.mean(jnp.minimum(
            ratio * adv, jnp.clip(ratio, 1 - clip_range, 1 + clip_range) * adv))
        v = jnp.mean(jnp.maximum((value - ret) ** 2, (ret + jnp.clip(
            value - ret, -clip_range, clip_range) - ret) ** 2)) / 2.0
        ent = -jnp.mean(jnp.sum(jax.nn.softmax(logits) * logp_all, 1))
        return pg + vf_coef * v - ent_coef * ent

    @jax.jit
    def update(state, obs, act, old_logp, adv, ret):
        params, opt_state = state
        loss, grads = jax.value_and_grad(loss_fn)(params, obs, act, old_logp,
                                                  adv, ret)
        upd, opt_state = optimizer.update(grads, opt_state, params)
        return (optax.apply_updates(params, upd), opt_state), loss

    @jax.jit
    def rollout(params, obs, key):
        logits, value = model.apply(params, obs, None)
        key, ks = jax.random.split(key)
        act = jax.random.categorical(ks, logits)
        logp = jax.nn.log_softmax(logits)[jnp.arange(logits.shape[0]), act]
        return act, logp, value, key

    @jax.jit
    def forward(params, obs, key=None):
        del key
        return model.apply(params, obs, None)

    return update, rollout, forward


class MlpBackboneMA(nn.Module):
    """MLP agnostico a dimensao (obs 75D ou state 72D): 128/128 ReLU."""

    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(128)(x))
        x = nn.relu(nn.Dense(128)(x))
        return x


class ActorOnly(nn.Module):
    n_actions: int

    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(128)(x))
        x = nn.relu(nn.Dense(128)(x))
        return nn.Dense(self.n_actions)(x)


class CentralCritic(nn.Module):
    @nn.compact
    def __call__(self, s):
        x = nn.relu(nn.Dense(128)(s))
        x = nn.relu(nn.Dense(128)(x))
        return nn.Dense(1)(x).squeeze(-1)


def make_mappo_update(actor, critic, optimizer, clip_range=0.2, vf_coef=0.5,
                      ent_coef=0.01):
    def loss_fn(pa, pc, obs, act, old_logp, adv, ret, state):
        logits = actor.apply(pa, obs)
        value = critic.apply(pc, state)
        logp_all = jax.nn.log_softmax(logits)
        logp = jnp.take_along_axis(logp_all, act[:, None], 1).squeeze(1)
        ratio = jnp.exp(logp - old_logp)
        pg = -jnp.mean(jnp.minimum(
            ratio * adv, jnp.clip(ratio, 1 - clip_range, 1 + clip_range) * adv))
        v = jnp.mean(jnp.maximum((value - ret) ** 2, (ret + jnp.clip(
            value - ret, -clip_range, clip_range) - ret) ** 2)) / 2.0
        ent = -jnp.mean(jnp.sum(jax.nn.softmax(logits) * logp_all, 1))
        return pg + vf_coef * v - ent_coef * ent

    @jax.jit
    def update(a_s, c_s, obs, act, old_logp, adv, ret, state):
        (loss, _), (ga, gc) = jax.value_and_grad(
            lambda a, c: (loss_fn(a, c, obs, act, old_logp, adv, ret, state),
                          None), argnums=(0, 1), has_aux=True)(a_s[0], c_s[0])
        ua, naos = optimizer.update(ga, a_s[1], a_s[0])
        uc, ncos = optimizer.update(gc, c_s[1], c_s[0])
        a_s = (optax.apply_updates(a_s[0], ua), naos)
        c_s = (optax.apply_updates(c_s[0], uc), ncos)
        return a_s, c_s, loss

    return update


def train_recurrent(args):
    """IPPO recorrente (GRU-128, padrao JaxMARL p/ SMAX).

    Rollout com carry (N*A,H) e reset em done; minibatches sobre
    SEQUENCIAS (S=N*A, mb=S//2, 2 epochs — paper); BPTT full em T=128;
    lr/ent via flags (paper: 4e-3/0.0; sem annealing, documentado).
    """
    import os
    from flax.serialization import msgpack_restore
    jax.config.update("jax_compilation_cache_dir",
                      os.environ.get("JAX_PORT_CACHE", "/tmp/jax_port_cache"))
    from jax_port.marl.recurrent import REC_H
    rng = np.random.default_rng(args.seed)
    key = jax.random.PRNGKey(args.seed)
    device = jax.devices()[0]
    N, T = args.num_envs, args.rollout
    print(f"marl ippo-recurrent {args.map} device={device}", flush=True)
    venv = SmaxVec(args.map, num_envs=N, seed=args.seed,
                   walls_cause_death=not args.no_walls)
    A, NA = venv.n_agents, venv.n_actions
    S = N * A
    opt = make_optimizer(lr=args.lr)
    model = RecurrentAC(n_actions=NA)
    key, k0 = jax.random.split(key)
    params = model.init(k0, jnp.zeros((1, 1, venv.obs_dim)),
                        jnp.zeros((1, REC_H)), jnp.zeros((1, 1), bool))
    state = (params, opt.init(params))
    update = make_ppo_seq_update(model, opt, ent_coef=args.ent)
    # ckpt/resume: salva (params, opt_state, done_steps) por iteracao;
    # estado RNG nao e restaurado (aproximacao documentada: mesmo lr,
    # nova amostragem; o treino e estocastico por design).
    it0 = 0
    if getattr(args, "resume", None):
        with open(args.resume, "rb") as fh:
            blob = msgpack_restore(fh.read())
        state = (from_bytes(params, blob["params"]),
                 from_bytes(opt.init(params), blob["opt"]))
        it0 = int(blob["it"])
        print(f"resumed {args.resume} it={it0}", flush=True)

    @jax.jit
    def rstep(params_, ob_, carry_, prev_done_, key_):
        logits, value, new_carry = model.apply(
            params_, ob_[None], carry_, prev_done_[None])
        # carry nao tem dim de tempo: retorna (M,H) inteiro ([0] aqui
        # colapsaria para (H,) e quebraria o h0 do update no iter seguinte).
        return logits[0], value[0], new_carry
    key, kw = jax.random.split(key)
    if False:  # warmup desativado p/ diagnostico (compila no 1o update)
        state, _ = update(
            state, jnp.zeros((T, 4, venv.obs_dim)), jnp.zeros((T, 4), jnp.int32),
            jnp.zeros((T, 4)), jnp.zeros((T, 4)), jnp.zeros((T, 4)),
            jnp.zeros((4, REC_H)), jnp.zeros((T, 4), bool))
    jax.block_until_ready(jax.tree_util.tree_leaves(state[0])[0])

    M = N * A
    carry = np.zeros((M, REC_H), np.float32)
    done_steps = it0 * T * N  # alinhado com a retomada (cada it = T*N steps)
    ep_wins, ep_rets, cur, curve = [], [], np.zeros(N), []
    t0 = _time.perf_counter()
    n_iters = max(1, (args.timesteps + N * T - 1) // (N * T))
    for it in range(it0, n_iters):
        b_obs = np.empty((T, M, venv.obs_dim), np.float32)
        b_act = np.empty((T, M), np.int32)
        b_rew = np.empty((T, M), np.float32)
        b_done = np.empty((T, M), bool)
        b_val = np.empty((T, M), np.float32)
        b_logp = np.empty((T, M), np.float32)
        h0 = carry.copy()
        prev_done = np.zeros(M, bool)
        for t in range(T):
            ob = np.asarray(venv.obs_batch()).reshape(M, -1)
            key, kf = jax.random.split(key)
            logits, value, carry = rstep(
                state[0], jnp.asarray(ob), jnp.asarray(carry),
                jnp.asarray(prev_done), kf)
            jax.block_until_ready((logits, carry))
            carry = np.asarray(carry)
            key, ks = jax.random.split(key)
            act = np.asarray(jax.random.categorical(ks, logits)).reshape(N, A)
            lp = jax.nn.log_softmax(logits)
            flat_idx = (np.arange(M), act.reshape(-1))
            logp = np.asarray(lp[flat_idx]).reshape(N, A)
            b_obs[t], b_act[t] = ob.reshape(M, -1), act.reshape(M)
            b_val[t], b_logp[t] = value.reshape(M), logp.reshape(M)
            _, rew, done, win = venv.step(act)
            b_rew[t] = np.broadcast_to(rew[:, None], (N, A)).reshape(-1).copy()
            b_done[t] = np.broadcast_to(done[:, None], (N, A)).reshape(-1).copy()
            prev_done = b_done[t].reshape(-1).copy()
            cur += rew
            for i in np.where(done)[0]:
                ep_wins.append(bool(win[i]))
                ep_rets.append(float(cur[i]))
                cur[i] = 0.0
        done_steps += T * N
        adv = np.empty((T, M), np.float32)
        last_v = b_val.reshape(T, M)[-1] * (
            1 - b_done.reshape(T, M)[-1].astype(np.float32))
        gae = np.zeros(M, np.float32)
        Bd = b_done.reshape(T, M)
        for t in reversed(range(T)):
            nt = 1 - Bd[t].astype(np.float32)
            nv = b_val.reshape(T, M)[t + 1] if t < T - 1 else last_v
            delta = b_rew.reshape(T, M)[t] + 0.99 * nv * nt - b_val.reshape(T, M)[t]
            gae = delta + 0.99 * 0.95 * nt * gae
            adv[t] = gae
        adv = adv.reshape(T, M)
        ret = (adv + b_val).astype(np.float32)
        F = (b_obs.reshape(T, M, -1), b_act.reshape(T, M).astype(np.int32),
             b_logp.reshape(T, M), adv.reshape(T, M), ret.reshape(T, M),
             Bd, h0)
        adv_n = F[3]
        adv_n = (adv_n - adv_n.mean()) / (adv_n.std() + 1e-8)
        sidx = rng.permutation(M)
        mb_s = max(1, M // 2)
        for _ep in range(2):
            for s in range(0, M, mb_s):
                mb = sidx[s:s + mb_s]
                d = lambda a: jnp.asarray(a[:, mb])
                state, _ = update(
                    state, d(F[0]), d(F[1]), d(F[2]), d(adv_n), d(F[4]),
                    jnp.asarray(F[6])[mb], d(Bd))
        el = _time.perf_counter() - t0
        wr = float(np.mean(ep_wins[-20:])) if ep_wins else 0.0
        curve.append({"steps": done_steps, "winrate": wr})
        print(f"iter={it+1} steps={done_steps} sps={done_steps/el:.0f} "
              f"win20={wr:.2f} eps={len(ep_wins)}", flush=True)
        if getattr(args, "ckpt", None) and (it + 1) % 10 == 0:
            _save_ppo_ckpt(args.ckpt, state, it + 1)
        if done_steps >= args.timesteps:
            break
    dt = _time.perf_counter() - t0
    out = {"algo": "ippo-recurrent", "map": args.map, "seed": args.seed,
           "timesteps": done_steps, "wall_s": round(dt, 1),
           "sps": round(done_steps / dt, 1),
           "train_episodes": len(ep_wins),
           "winrate_train": round(float(np.mean(ep_wins)), 3) if ep_wins else 0.0,
           "train_ret_mean": round(float(np.mean(ep_rets)), 2) if ep_rets else 0.0,
           "curve": curve}
    if args.eval_eps > 0:
        out["eval"] = evaluate_recurrent(
            venv, args, state, model, NA, device, key)
    with open(args.out, "w") as fh:
        _json.dump(out, fh, indent=2)
    print(_json.dumps({k: v for k, v in out.items() if k != "curve"}, indent=2))
    return out


def _save_ppo_ckpt(path, state, it):
    """Checkpoint atomico: (params, opt_state, it) em msgpack."""
    from flax.serialization import msgpack_serialize
    import os
    blob = msgpack_serialize(
        {"params": to_bytes(state[0]), "opt": to_bytes(state[1]), "it": it})
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(blob)
    os.replace(tmp, path)


def evaluate_recurrent(venv, args, state, model, NA, device, key):
    """Eval greedy com carry (win-rate padrao SMAC)."""
    from jax_port.marl.recurrent import REC_H
    from jax_port.marl.smax_vec import SmaxVec
    ev = SmaxVec(args.map, num_envs=args.eval_envs, seed=args.seed + 1000)
    E = args.eval_envs
    M = E * venv.n_agents
    carry = np.zeros((M, REC_H), np.float32)
    wins, rets = [], []
    cur = np.zeros(E)

    @jax.jit
    def esteps(params_, ob_, carry_):
        logits, _, new_carry = model.apply(
            params_, ob_[None], carry_,
            jnp.zeros((1, M), bool))
        return logits[0], new_carry

    while len(wins) < args.eval_eps:
        ob = np.asarray(ev.obs_batch()).reshape(M, -1)
        logits, carry = esteps(state[0], jnp.asarray(ob),
                              jnp.asarray(carry))
        jax.block_until_ready((logits, carry))
        carry = np.asarray(carry)
        act = np.asarray(jnp.argmax(logits, -1)).reshape(E, -1)
        _, rew, done, win = ev.step(act)
        cur += rew
        for i in np.where(done)[0]:
            wins.append(bool(win[i]))
            rets.append(float(cur[i]))
            cur[i] = 0.0
        carry = np.array(carry)
        fin = np.repeat(done, venv.n_agents)
        carry[fin] = 0.0
    return {"winrate": round(float(np.mean(wins)), 3), "episodes": len(wins),
            "mean_return": round(float(np.mean(rets)), 2)}


def train(args):
    import os
    jax.config.update("jax_compilation_cache_dir",
                      os.environ.get("JAX_PORT_CACHE", "/tmp/jax_port_cache"))
    assert args.algo in ("ippo", "mappo")
    if getattr(args, "recurrent", False):
        assert args.algo == "ippo", "recorrente: IPPO (MAPPO recorrente e follow-up)"
        return train_recurrent(args)
    rng = np.random.default_rng(args.seed)
    key = jax.random.PRNGKey(args.seed)
    device = jax.devices()[0]
    N, T, MB, A = args.num_envs, args.rollout, args.minibatch, None
    print(f"marl {args.algo} {args.map} device={device}", flush=True)
    venv = SmaxVec(args.map, num_envs=N, seed=args.seed,
                   walls_cause_death=not args.no_walls)
    A = venv.n_agents
    NA = venv.n_actions
    opt = make_optimizer(lr=getattr(args, "lr", 3e-4))
    if args.algo == "ippo":
        # MlpBackbone = mesma convencao MLP do estudo ([64,64] tanh).
        model = ActorCritic(backbone=MlpBackbone(), n_actions=NA)
        key, k0 = jax.random.split(key)
        params = model.init(k0, jnp.zeros((1, venv.obs_dim)), None)
        state = (params, opt.init(params))
        update_fn, rollout_fn, forward_fn = make_update_float(model, opt)
        key, kw = jax.random.split(key)
        state, _ = update_fn(
            state, jnp.zeros((MB, venv.obs_dim), jnp.float32),
            jnp.zeros((MB,), jnp.int32), jnp.zeros((MB,)),
            jnp.zeros((MB,)), jnp.zeros((MB,)))
        del kw
    else:
        actor, critic = ActorOnly(n_actions=NA), CentralCritic()
        key, k0, k1 = jax.random.split(key, 3)
        pa = actor.init(k0, jnp.zeros((1, venv.obs_dim)))
        pc = critic.init(k1, jnp.zeros((1, venv.state_dim)))
        astate, cstate = (pa, opt.init(pa)), (pc, opt.init(pc))
        update_fn = make_mappo_update(actor, critic, opt)

        @jax.jit
        def rollout_fn(pa_, ob, key):
            logits = actor.apply(pa_, ob)
            key, ks = jax.random.split(key)
            act = jax.random.categorical(ks, logits)
            logp = jax.nn.log_softmax(logits)[jnp.arange(logits.shape[0]), act]
            return act, logp, key

        @jax.jit
        def forward_fn(pa_, ob, key=None):
            del key
            return actor.apply(pa_, ob), jnp.zeros((ob.shape[0],))
        key, kw = jax.random.split(key)
        astate, cstate, _ = update_fn(
            astate, cstate, jnp.zeros((MB, venv.obs_dim)),
            jnp.zeros((MB,), jnp.int32), jnp.zeros((MB,)),
            jnp.zeros((MB,)), jnp.zeros((MB,)),
            jnp.zeros((MB, venv.state_dim)))
    jax.block_until_ready(jax.tree_util.tree_leaves(
        state[0] if args.algo == "ippo" else astate[0])[0])

    n_iters = max(1, (args.timesteps + N * T - 1) // (N * T))
    done_steps = 0
    ep_rets, ep_wins, curve = [], [], []
    cur_ret = np.zeros(N)
    t0 = _time.perf_counter()
    for it in range(n_iters):
        b_obs = np.empty((T, N, A, venv.obs_dim), np.float32)
        b_st = np.empty((T, N, venv.state_dim), np.float32)
        b_act = np.empty((T, N, A), np.int32)
        b_rew = np.empty((T, N, A), np.float32)
        b_done = np.empty((T, N, A), bool)
        b_val = np.empty((T, N, A), np.float32)
        b_logp = np.empty((T, N, A), np.float32)
        for t in range(T):
            ob = np.asarray(venv.obs_batch())
            sb = np.asarray(venv.state_batch())
            if args.algo == "ippo":
                key, kf = jax.random.split(key)
                logits, value = forward_fn(
                    state[0], jnp.asarray(ob.reshape(N * A, -1)), kf)
                jax.block_until_ready((logits, value))
                logits = np.asarray(logits).reshape(N, A, NA)
                value = np.asarray(value).reshape(N, A)
                key, ks = jax.random.split(key)
                act = np.asarray(jax.random.categorical(
                    ks, jnp.asarray(logits)))
                lp = jax.nn.log_softmax(jnp.asarray(logits))
                logp = np.asarray(lp[np.arange(N)[:, None],
                                     np.arange(A), act])
            else:
                key, kf = jax.random.split(key)
                logits = np.asarray(actor.apply(
                    astate[0], jnp.asarray(ob.reshape(N * A, -1))).reshape(
                        N, A, NA))
                value_c = np.asarray(critic.apply(
                    cstate[0], jnp.asarray(sb))).reshape(N)
                value = np.broadcast_to(value_c[:, None], (N, A)).copy()
                key, ks = jax.random.split(key)
                act = np.asarray(jax.random.categorical(
                    ks, jnp.asarray(logits)))
                lp = jax.nn.log_softmax(jnp.asarray(logits))
                logp = np.asarray(lp[np.arange(N)[:, None],
                                     np.arange(A), act])
            b_obs[t], b_st[t] = ob, sb
            b_act[t], b_val[t], b_logp[t] = act, value, logp
            _, rew, done, win = venv.step(act)
            b_rew[t] = np.broadcast_to(rew[:, None], (N, A)).copy()
            b_done[t] = np.broadcast_to(done[:, None], (N, A)).copy()
            cur_ret += rew
            if done.any():
                for i in np.where(done)[0]:
                    ep_wins.append(bool(win[i]))
                    ep_rets.append(float(cur_ret[i]))
                    cur_ret[i] = 0.0
        # GAE por (env, agente)
        Tn = T * N * A
        adv = np.empty((T, N, A), np.float32)
        last_v = b_val[-1] * (1 - b_done[-1].astype(np.float32))
        gae = np.zeros((N, A), np.float32)
        for t in reversed(range(T)):
            nt = 1 - b_done[t].astype(np.float32)
            nv = b_val[t + 1] if t < T - 1 else last_v
            delta = b_rew[t] + 0.99 * nv * nt - b_val[t]
            gae = delta + 0.99 * 0.95 * nt * gae
            adv[t] = gae
        ret = (adv + b_val).astype(np.float32)
        F = (b_obs.reshape(Tn, -1), b_act.reshape(Tn).astype(np.int32),
             b_logp.reshape(Tn), adv.reshape(Tn), ret.reshape(Tn),
             b_st.reshape(T * N, -1))
        adv_n = F[3]
        adv_n = (adv_n - adv_n.mean()) / (adv_n.std() + 1e-8)
        idx = rng.permutation(Tn)
        for _ep in range(3):
            for s in range(0, Tn, MB):
                mb = idx[s:s + MB]
                d = lambda a: jnp.asarray(a[mb])
                if args.algo == "ippo":
                    state, _ = update_fn(state, d(F[0]), d(F[1]), d(F[2]),
                                         d(adv_n), d(F[4]))
                else:
                    full = np.repeat(np.arange(T * N), A)
                    sb_mb = F[5][full[mb]]
                    astate, cstate, _ = update_fn(
                        astate, cstate, d(F[0]), d(F[1]), d(F[2]),
                        d(adv_n), d(F[4]), jnp.asarray(sb_mb))
        done_steps += T * N
        el = _time.perf_counter() - t0
        wr = float(np.mean(ep_wins[-20:])) if ep_wins else 0.0
        curve.append({"steps": done_steps, "winrate": wr})
        print(f"iter={it+1} steps={done_steps} sps={done_steps/el:.0f} "
              f"win20={wr:.2f} eps={len(ep_wins)}", flush=True)
        if done_steps >= args.timesteps:
            break
    dt = _time.perf_counter() - t0
    out = {"algo": args.algo, "map": args.map, "seed": args.seed,
           "timesteps": done_steps, "wall_s": round(dt, 1),
           "sps": round(done_steps / dt, 1),
           "train_episodes": len(ep_wins),
           "winrate_train": round(float(np.mean(ep_wins)), 3) if ep_wins else 0.0,
           "train_ret_mean": round(float(np.mean(ep_rets)), 2) if ep_rets else 0.0,
           "curve": curve}
    if args.eval_eps > 0:
        out["eval"] = evaluate(venv, args, forward_fn if args.algo == "ippo"
                               else None, astate if args.algo == "mappo" else
                               state, NA, device, key)
    import json as _j
    with open(args.out, "w") as fh:
        _j.dump(out, fh, indent=2)
    print(_j.dumps({k: v for k, v in out.items() if k != "curve"}, indent=2))
    return out


def evaluate(venv, args, forward_fn, state, n_actions, device, key):
    """Win-rate greedy + retorno (padrao SMAC)."""
    from jax_port.marl.smax_vec import SmaxVec
    ev = SmaxVec(args.map, num_envs=args.eval_envs, seed=args.seed + 1000,
                 walls_cause_death=not args.no_walls)
    wins, rets, n = [], [], 0
    cur = np.zeros(args.eval_envs)
    while len(wins) < args.eval_eps:
        ob = np.asarray(ev.obs_batch())
        flat = jnp.asarray(ob.reshape(-1, ob.shape[-1]))
        if args.algo == "ippo":
            logits, _ = forward_fn(state[0], flat, None)
        else:
            logits = ActorOnly(n_actions=n_actions).apply(state[0], flat)
        act = np.asarray(jnp.argmax(logits, -1)).reshape(args.eval_envs, -1)
        _, rew, done, win = ev.step(act)
        cur += rew
        for i in np.where(done)[0]:
            wins.append(bool(win[i]))
            rets.append(float(cur[i]))
            cur[i] = 0.0
            n += 1
        if n >= args.eval_eps:
            break
    return {"winrate": round(float(np.mean(wins)), 3), "episodes": len(wins),
            "mean_return": round(float(np.mean(rets)), 2)}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="ippo", choices=["ippo", "mappo"])
    ap.add_argument("--map", default="3m")
    ap.add_argument("--no-walls", action="store_true",
                    help="walls_cause_death=False (default do env e True)")
    ap.add_argument("--recurrent", action="store_true",
                    help="GRU-128 (padrao JaxMARL p/ SMAX; so IPPO)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--timesteps", type=int, default=1000000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-envs", type=int, default=32)
    ap.add_argument("--rollout", type=int, default=128)
    ap.add_argument("--minibatch", type=int, default=1024)
    ap.add_argument("--eval-eps", type=int, default=32)
    ap.add_argument("--eval-envs", type=int, default=8)
    ap.add_argument("--out", default="jax_port/marl_train.json")
    ap.add_argument("--ckpt", default=None,
                    help="path p/ checkpoint (params+opt a cada it)")
    ap.add_argument("--resume", default=None,
                    help="retoma de checkpoint salvo (params+opt)")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
