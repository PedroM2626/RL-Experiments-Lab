"""Treino MA-POCA / CTE / TarMAC sobre SMAX.

mapoca : rollout com o actor + alvos TD p/ o critico + update
         contrafactual (paradigms.make_mapoca_update).
cte    : wrapper espaco-produto (8^A acoes; assert <= 4096, i.e. 3m)
         + PPO joint via make_update_float (reuso total).
tarmac : actor com comunicacao (TarMACActor) + critico centralizado,
         loop estilo MAPPO com batches (E,A) preservados p/ a atencao.
Eval: win-rate greedy + retorno (padrao SMAC). Uso:
  train_paradigms.py --algo mapoca --map 3m --timesteps 1000000
"""

import argparse
import json
import os
import time

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_port.backbones import MlpBackbone
from jax_port.marl.paradigms import (AgentEncoder, AttnCritic, JointBackbone,
                                     PolicyHead, TarMACActor,
                                     make_mapoca_update)
from jax_port.marl.ppo_marl import CentralCritic, make_update_float
from jax_port.marl.smax_vec import SmaxVec
from jax_port.networks import ActorCritic
from jax_port.ppo import compute_gae, make_optimizer



def train(args):
    jax.config.update("jax_compilation_cache_dir",
                      os.environ.get("JAX_PORT_CACHE", "/tmp/jax_port_cache"))
    assert args.algo in ("mapoca", "cte", "tarmac")
    key = jax.random.PRNGKey(args.seed)
    device = jax.devices()[0]
    print(f"marl {args.algo} {args.map} device={device}", flush=True)
    venv = SmaxVec(args.map, num_envs=args.num_envs, seed=args.seed)
    A, NA, N = venv.n_agents, venv.n_actions, args.num_envs
    opt = make_optimizer(lr=3e-4)
    out_extra = {}

    if args.algo == "cte":
        nj = NA ** A
        assert nj <= 4096, f"espaco-produto {nj} grande demais (use 3m)"
        journal = {"n_joint": nj}

        model = ActorCritic(backbone=JointBackbone(), n_actions=nj)
        key, k0 = jax.random.split(key)
        in_dim = A * venv.obs_dim + venv.state_dim
        params = model.init(k0, jnp.zeros((1, in_dim)), None)
        state = (params, opt.init(params))
        update_fn, _, forward_fn = make_update_float(model, opt)

        def digits(a):
            return np.stack(
                [(a // (NA ** k)) % NA for k in range(A)], -1).astype(np.int32)

        # rollout joint inline (obs concatenada + decode p/ o env)
        rng = np.random.default_rng(args.seed)
        key2 = jax.random.PRNGKey(args.seed + 999)
        T, MB = args.rollout, args.minibatch
        n_iters = max(1, (args.timesteps + N * T - 1) // (N * T))
        done_steps = 0
        ep_wins, ep_rets, cur, curve = [], [], np.zeros(N), []
        t0 = time.perf_counter()
        pw = NA ** np.arange(A)
        for it in range(n_iters):
            Bo = np.empty((T, N, in_dim), np.float32)
            Ba = np.empty((T, N), np.int32)
            Br = np.empty((T, N), np.float32)
            Bd = np.empty((T, N), bool)
            Bv = np.empty((T, N), np.float32)
            Bl = np.empty((T, N), np.float32)
            for t in range(T):
                ob = np.asarray(venv.obs_batch())
                sb = np.asarray(venv.state_batch())
                job = np.concatenate([ob.reshape(N, -1), sb], -1)
                key2, kf = jax.random.split(key2)
                logits, value = forward_fn(state[0], jnp.asarray(job), kf)
                jax.block_until_ready((logits, value))
                logits, value = np.asarray(logits), np.asarray(value)
                key2, ks = jax.random.split(key2)
                ja = np.asarray(jax.random.categorical(ks, jnp.asarray(logits)))
                lp = jax.nn.log_softmax(jnp.asarray(logits))
                Bo[t], Ba[t] = job, ja
                Bv[t], Bl[t] = value, np.asarray(lp[np.arange(N), ja])
                act = ((ja[:, None] // pw[None, :]) % NA).astype(np.int32)
                _, rew, done, win = venv.step(act)
                Br[t], Bd[t] = rew.astype(np.float32), done
                cur += rew
                for i in np.where(done)[0]:
                    ep_wins.append(bool(win[i]))
                    ep_rets.append(float(cur[i]))
                    cur[i] = 0.0
            done_steps += T * N
            adv = np.empty((T, N), np.float32)
            last_v = Bv[-1] * (1 - Bd[-1].astype(np.float32))
            gae = np.zeros(N, np.float32)
            for t in reversed(range(T)):
                nt = 1 - Bd[t].astype(np.float32)
                nv = Bv[t + 1] if t < T - 1 else last_v
                delta = Br[t] + 0.99 * nv * nt - Bv[t]
                gae = delta + 0.99 * 0.95 * nt * gae
                adv[t] = gae
            ret = (adv + Bv).astype(np.float32)
            Tn = T * N
            F = (Bo.reshape(Tn, -1), Ba.reshape(Tn).astype(np.int32),
                 Bl.reshape(Tn), adv.reshape(Tn), ret.reshape(Tn))
            adv_n = F[3]
            adv_n = (adv_n - adv_n.mean()) / (adv_n.std() + 1e-8)
            idx = rng.permutation(Tn)
            for _ep in range(3):
                for s in range(0, Tn, MB):
                    mb = idx[s:s + MB]
                    d = lambda a: jnp.asarray(a[mb])
                    state, _ = update_fn(state, d(F[0]), d(F[1]), d(F[2]),
                                         d(adv_n), d(F[4]))
            el = time.perf_counter() - t0
            wr = float(np.mean(ep_wins[-20:])) if ep_wins else 0.0
            curve.append({"steps": done_steps, "winrate": wr})
            print(f"iter={it+1} steps={done_steps} sps={done_steps/el:.0f} "
                  f"win20={wr:.2f} eps={len(ep_wins)}", flush=True)
            if done_steps >= args.timesteps:
                break
        dt = time.perf_counter() - t0
        out = {"algo": "cte", "map": args.map, "seed": args.seed,
               "n_joint": journal["n_joint"], "timesteps": done_steps,
               "wall_s": round(dt, 1), "sps": round(done_steps / dt, 1),
               "train_episodes": len(ep_wins),
               "winrate_train": round(float(np.mean(ep_wins)), 3) if ep_wins else 0.0,
               "train_ret_mean": round(float(np.mean(ep_rets)), 2) if ep_rets else 0.0,
               "curve": curve}
        out_extra = {"state": state, "kind": "cte", "NA": NA, "A": A,
                     "n_joint": journal["n_joint"]}
    elif args.algo == "tarmac":
        model = ActorCritic(backbone=MlpBackbone(), n_actions=NA)
        # TarMAC: backbone substituido pelo actor comunicante no rollout;
        # treino segue MAPPO (critico central) com minibatches (E,A).
        comm = TarMACActor(n_actions=NA)
        critic = CentralCritic()
        key, k0, k1, k2 = jax.random.split(key, 4)
        p_comm = comm.init(k0, jnp.zeros((1, A, venv.obs_dim)))
        p_cri = critic.init(k1, jnp.zeros((1, venv.state_dim)))
        astate = (p_comm, opt.init(p_comm))
        cstate = (p_cri, opt.init(p_cri))

        @jax.jit
        def comm_logits(pp, ob):
            return comm.apply(pp, ob)

        @jax.jit
        def comm_update(a_s, c_s, obs, act, old_logp, adv, ret, state_b):
            def loss_fn(pa, pc):
                logits = comm.apply(pa, obs)
                value = critic.apply(pc, state_b)
                logp_all = jax.nn.log_softmax(logits)
                logp = jnp.take_along_axis(logp_all, act[..., None], -1).squeeze(-1)
                ratio = jnp.exp(logp - old_logp)
                pg = -jnp.mean(jnp.minimum(
                    ratio * adv,
                    jnp.clip(ratio, 0.8, 1.2) * adv))
                v = jnp.mean(jnp.maximum(
                    (value - ret) ** 2,
                    (ret + jnp.clip(value - ret, -0.2, 0.2) - ret) ** 2)) / 2.0
                ent = -jnp.mean(jnp.sum(jax.nn.softmax(logits) * logp_all, (1, 2)))
                return pg + 0.5 * v - 0.01 * ent
            (tot, _), (ga, gc) = jax.value_and_grad(
                lambda a, c: (loss_fn(a, c), None), argnums=(0, 1),
                has_aux=True)(a_s[0], c_s[0])
            ua, naos = opt.update(ga, a_s[1], a_s[0])
            uc, ncos = opt.update(gc, c_s[1], c_s[0])
            return (optax.apply_updates(a_s[0], ua), naos), \
                (optax.apply_updates(c_s[0], uc), ncos), tot

        import optax as _ox  # noqa (optax ja importado? nao: importar)
        rng = np.random.default_rng(args.seed)
        key2 = jax.random.PRNGKey(args.seed + 999)
        T, MB, N = args.rollout, args.minibatch, args.num_envs
        n_iters = max(1, (args.timesteps + N * T - 1) // (N * T))
        done_steps = 0
        ep_wins, ep_rets, cur, curve = [], [], np.zeros(N), []
        t0 = time.perf_counter()
        for it in range(n_iters):
            Bo = np.empty((T, N, A, venv.obs_dim), np.float32)
            Bs = np.empty((T, N, venv.state_dim), np.float32)
            Ba = np.empty((T, N, A), np.int32)
            Br = np.empty((T, N, A), np.float32)
            Bd = np.empty((T, N, A), bool)
            Bv = np.empty((T, N, A), np.float32)
            Bl = np.empty((T, N, A), np.float32)
            Bc = np.empty((T, N), np.float32)
            for t in range(T):
                ob = np.asarray(venv.obs_batch())
                sb = np.asarray(venv.state_batch())
                key2, kf = jax.random.split(key2)
                logits = np.asarray(comm_logits(astate[0], jnp.asarray(ob)))
                vc = np.asarray(critic.apply(cstate[0], jnp.asarray(sb))).reshape(N)
                value = np.broadcast_to(vc[:, None], (N, A)).copy()
                key2, ks = jax.random.split(key2)
                act = np.asarray(jax.random.categorical(ks, jnp.asarray(logits)))
                lp = jax.nn.log_softmax(jnp.asarray(logits))
                logp = np.asarray(lp[np.arange(N)[:, None], np.arange(A), act])
                Bo[t], Bs[t], Ba[t] = ob, sb, act
                Bv[t], Bl[t], Bc[t] = value, logp, vc
                _, rew, done, win = venv.step(act)
                Br[t] = np.broadcast_to(rew[:, None], (N, A)).copy()
                Bd[t] = np.broadcast_to(done[:, None], (N, A)).copy()
                cur += rew
                for i in np.where(done)[0]:
                    ep_wins.append(bool(win[i]))
                    ep_rets.append(float(cur[i]))
                    cur[i] = 0.0
            done_steps += T * N
            Tn = T * N
            adv = np.empty((T, N, A), np.float32)
            last_v = Bv[-1] * (1 - Bd[-1].astype(np.float32))
            gae = np.zeros((N, A), np.float32)
            for t in reversed(range(T)):
                nt = 1 - Bd[t].astype(np.float32)
                nv = Bv[t + 1] if t < T - 1 else last_v
                delta = Br[t] + 0.99 * nv * nt - Bv[t]
                gae = delta + 0.99 * 0.95 * nt * gae
                adv[t] = gae
            ret = (adv + Bv).astype(np.float32)
            # retornos do critico central (por env), p/ a loss de valor
            adv_c = np.empty((T, N), np.float32)
            last_vc = Bc[-1] * (1 - Bd[-1, :, 0].astype(np.float32))
            gae_c = np.zeros(N, np.float32)
            Brt = Br[:, :, 0]
            for t in reversed(range(T)):
                nt = 1 - Bd[t, :, 0].astype(np.float32)
                nv = Bc[t + 1] if t < T - 1 else last_vc
                delta = Brt[t] + 0.99 * nv * nt - Bc[t]
                gae_c = delta + 0.99 * 0.95 * nt * gae_c
                adv_c[t] = gae_c
            ret_c = (adv_c + Bc).astype(np.float32)
            adv_n = adv.reshape(Tn * A)
            adv_n = (adv_n - adv_n.mean()) / (adv_n.std() + 1e-8)
            adv_n = adv_n.reshape(T, N, A)
            eidx = rng.permutation(Tn)
            for _ep in range(3):
                for s in range(0, Tn, MB):
                    mb = eidx[s:s + MB]
                    # fatia por ENV (preserva dim agente p/ atencao)
                    astate, cstate, _ = comm_update(
                        astate, cstate, jnp.asarray(Bo.reshape(Tn, A, -1)[mb]),
                        jnp.asarray(Ba.reshape(Tn, A)[mb]),
                        jnp.asarray(Bl.reshape(Tn, A)[mb]),
                        jnp.asarray(adv_n.reshape(Tn, A)[mb]),
                        jnp.asarray(ret_c.reshape(Tn)[mb]),
                        jnp.asarray(Bs.reshape(Tn, -1)[mb]))
            el = time.perf_counter() - t0
            wr = float(np.mean(ep_wins[-20:])) if ep_wins else 0.0
            curve.append({"steps": done_steps, "winrate": wr})
            print(f"iter={it+1} steps={done_steps} sps={done_steps/el:.0f} "
                  f"win20={wr:.2f} eps={len(ep_wins)}", flush=True)
            if done_steps >= args.timesteps:
                break
        dt = time.perf_counter() - t0
        out = {"algo": "tarmac", "map": args.map, "seed": args.seed,
               "timesteps": done_steps, "wall_s": round(dt, 1),
               "sps": round(done_steps / dt, 1),
               "train_episodes": len(ep_wins),
               "winrate_train": round(float(np.mean(ep_wins)), 3) if ep_wins else 0.0,
               "train_ret_mean": round(float(np.mean(ep_rets)), 2) if ep_rets else 0.0,
               "curve": curve}
        out_extra = {"astate": astate, "kind": "tarmac", "NA": NA, "A": A}
    else:  # mapoca
        encoder, critic = AgentEncoder(), AttnCritic(n_actions=NA)
        actor = PolicyHead(n_actions=NA)
        # NOTA: rollout inline abaixo (versao unificada).
        key, k0, k1, k2 = jax.random.split(key, 4)
        p_enc = AgentEncoder().init(k0, jnp.zeros((1, venv.obs_dim)))
        # policy consome embeddings 64D (nao obs!): init com a forma certa.
        p_pol = PolicyHead(n_actions=NA).init(
            k1, jnp.zeros((1, 64)))
        p_cri = critic.init(k2, jnp.zeros((1, A, 64)),
                            jnp.zeros((1, A, NA)), jnp.zeros((1, 72)))
        astate = ({"enc": p_enc, "pol": p_pol}, opt.init(
            {"enc": p_enc, "pol": p_pol}))
        cstate = (p_cri, opt.init(p_cri))
        update_fn = make_mapoca_update(AgentEncoder(), critic, PolicyHead(n_actions=NA), opt, NA)
        rng = np.random.default_rng(args.seed)
        key2 = jax.random.PRNGKey(args.seed + 999)
        T, MB, N = args.rollout, args.minibatch, args.num_envs
        n_iters = max(1, (args.timesteps + N * T - 1) // (N * T))
        done_steps = 0
        ep_wins, ep_rets, cur, curve = [], [], np.zeros(N), []
        t0 = time.perf_counter()
        for it in range(n_iters):
            Bo = np.empty((T, N, A, venv.obs_dim), np.float32)
            Bs = np.empty((T, N, venv.state_dim), np.float32)
            Ba = np.empty((T, N, A), np.int32)
            Br = np.empty((T, N, A), np.float32)
            Bd = np.empty((T, N, A), bool)
            Bl = np.empty((T, N, A), np.float32)
            Brt = np.empty((T, N), np.float32)
            for t in range(T):
                ob = np.asarray(venv.obs_batch())
                sb = np.asarray(venv.state_batch())
                key2, kf = jax.random.split(key2)
                embs = np.asarray(AgentEncoder().apply(
                    astate[0]["enc"], jnp.asarray(ob.reshape(N * A, -1))).reshape(N, A, -1))
                logits = np.asarray(PolicyHead(n_actions=NA).apply(
                        astate[0]["pol"], jnp.asarray(embs.reshape(N * A, -1))).reshape(N, A, NA))
                key2, ks = jax.random.split(key2)
                act = np.asarray(jax.random.categorical(ks, jnp.asarray(logits)))
                lp = jax.nn.log_softmax(jnp.asarray(logits))
                logp = np.asarray(lp[np.arange(N)[:, None], np.arange(A), act])
                Bo[t], Bs[t], Ba[t], Bl[t] = ob, sb, act, logp
                _, rew, done, win = venv.step(act)
                Br[t] = np.broadcast_to(rew[:, None], (N, A)).copy()
                Bd[t] = np.broadcast_to(done[:, None], (N, A)).copy()
                Brt[t] = rew.astype(np.float32)
                cur += rew
                for i in np.where(done)[0]:
                    ep_wins.append(bool(win[i]))
                    ep_rets.append(float(cur[i]))
                    cur[i] = 0.0
            done_steps += T * N
            Tn = T * N * A
            # Retornos Monte-Carlo por env (critico Q sem bootstrap:
            # sem target net por construcao).
            G = np.empty((T, N), np.float32)
            g = np.zeros(N, np.float32)
            dn = Bd[:, :, 0]
            for t in reversed(range(T)):
                g = Brt[t] + 0.99 * (1 - dn[t].astype(np.float32)) * g
                G[t] = g
            Gret = G  # (T,N) retornos MC por env (critico Q global)
            Srep = Bs.reshape(T * N, -1)
            Gret_f = Gret.reshape(T * N)
            idx = rng.permutation(T * N)
            for _ep in range(3):
                for s in range(0, T * N, MB):
                    mb = idx[s:s + MB]
                    d = lambda a: jnp.asarray(a[mb])
                    astate, cstate, _ = update_fn(
                        astate, cstate, d(Bo.reshape(T * N, A, -1)),
                        d(Ba.reshape(T * N, A)), d(Bl.reshape(T * N, A)),
                        d(Gret_f), d(Srep))
            el = time.perf_counter() - t0
            wr = float(np.mean(ep_wins[-20:])) if ep_wins else 0.0
            curve.append({"steps": done_steps, "winrate": wr})
            print(f"iter={it+1} steps={done_steps} sps={done_steps/el:.0f} "
                  f"win20={wr:.2f} eps={len(ep_wins)}", flush=True)
            if done_steps >= args.timesteps:
                break
        dt = time.perf_counter() - t0
        out = {"algo": "mapoca", "map": args.map, "seed": args.seed,
               "timesteps": done_steps, "wall_s": round(dt, 1),
               "sps": round(done_steps / dt, 1),
               "train_episodes": len(ep_wins),
               "winrate_train": round(float(np.mean(ep_wins)), 3) if ep_wins else 0.0,
               "train_ret_mean": round(float(np.mean(ep_rets)), 2) if ep_rets else 0.0,
               "curve": curve}
        out_extra = {"astate": astate, "kind": "mapoca", "NA": NA, "A": A}
    if args.eval_eps > 0:
        out["eval"] = evaluate_paradigm(venv, args, out_extra, device, key)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "curve"}, indent=2))
    return out


def evaluate_paradigm(venv, args, extra, device, key):
    from jax_port.marl.smax_vec import SmaxVec
    ev = SmaxVec(args.map, num_envs=args.eval_envs, seed=args.seed + 1000)
    wins, rets = [], []
    cur = np.zeros(args.eval_envs)
    kind = extra["kind"]
    while len(wins) < args.eval_eps:
        ob = np.asarray(ev.obs_batch())
        E = args.eval_envs
        if kind == "cte":
            sb = np.asarray(ev.state_batch())
            job = jnp.asarray(np.concatenate([ob.reshape(E, -1), sb], -1))
            logits, _ = ActorCritic(backbone=JointBackbone(),
                                    n_actions=extra["n_joint"]).apply(
                                        extra["state"][0], job)
            ja = np.asarray(jnp.argmax(logits, -1))
            pw = extra["NA"] ** np.arange(extra["A"])
            act = ((ja[:, None] // pw[None, :]) % extra["NA"]).astype(np.int32)
        elif kind == "tarmac":
            logits = np.asarray(TarMACActor(
                n_actions=extra["NA"]).apply(extra["astate"][0],
                                            jnp.asarray(ob)))
            act = np.asarray(jnp.argmax(logits, -1))
        else:
            embs = np.asarray(AgentEncoder().apply(
                extra["astate"][0]["enc"],
                jnp.asarray(ob.reshape(E * extra["A"], -1))).reshape(
                    E, extra["A"], -1))
            logits = np.asarray(PolicyHead(n_actions=extra["NA"]).apply(
                    extra["astate"][0]["pol"],
                    jnp.asarray(embs.reshape(E * extra["A"], -1))).reshape(
                        E, extra["A"], extra["NA"]))
            act = np.asarray(jnp.argmax(logits, -1))
        _, rew, done, win = ev.step(act)
        cur += rew
        for i in np.where(done)[0]:
            wins.append(bool(win[i]))
            rets.append(float(cur[i]))
            cur[i] = 0.0
    return {"winrate": round(float(np.mean(wins)), 3), "episodes": len(wins),
            "mean_return": round(float(np.mean(rets)), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="mapoca", choices=["mapoca", "cte", "tarmac"])
    ap.add_argument("--map", default="3m")
    ap.add_argument("--timesteps", type=int, default=1000000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-envs", type=int, default=32)
    ap.add_argument("--rollout", type=int, default=128)
    ap.add_argument("--minibatch", type=int, default=1024)
    ap.add_argument("--eval-eps", type=int, default=32)
    ap.add_argument("--eval-envs", type=int, default=8)
    ap.add_argument("--out", default="jax_port/marl_paradigm.json")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
