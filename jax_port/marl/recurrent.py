"""Recurrent MARL policies (JaxMARL default: GRU-128).

RecurrentAC: obs -> Dense128 -> GRU128 -> policy/value heads.
Sequence forward via unrolled Python loop (static L) — same
pattern as Dreamer (no lax.scan over submodule methods).
h0 per (env, agent); carry reset where done (passed mask).
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

REC_H = 128


class RecurrentAC(nn.Module):
    n_actions: int

    def setup(self):
        self.feat = nn.Dense(REC_H)
        self.w_ir = nn.Dense(REC_H)
        self.w_hr = nn.Dense(REC_H, use_bias=False)
        self.w_iz = nn.Dense(REC_H)
        self.w_hz = nn.Dense(REC_H, use_bias=False)
        self.w_in = nn.Dense(REC_H)
        self.w_hn = nn.Dense(REC_H, use_bias=False)
        self.b_hn = self.param("b_hn", nn.initializers.zeros, (REC_H,))
        self.pi = nn.Dense(self.n_actions)
        self.vf = nn.Dense(1)

    def gru(self, h, x):
        r = jax.nn.sigmoid(self.w_ir(x) + self.w_hr(h))
        z = jax.nn.sigmoid(self.w_iz(x) + self.w_hz(h))
        n = jnp.tanh(self.w_in(x) + r * (self.w_hn(h) + self.b_hn))
        return (1 - z) * n + z * h

    def step(self, carry, obs):
        h = nn.relu(self.feat(obs))
        h2 = self.gru(carry, h)
        return h2, self.pi(h2), self.vf(h2).squeeze(-1)

    def __call__(self, obs_seq, h0, dones=None):
        # obs_seq: (T,B,O) time-major; h0: (B,H); dones: (T,B)|None.
        h = h0
        ls, vs = [], []
        for t in range(obs_seq.shape[0]):
            if dones is not None:
                m = (1 - dones[t].astype(jnp.float32))[:, None]
                h = h * m
            h, l, v = self.step(h, obs_seq[t])
            ls.append(l)
            vs.append(v)
        return jnp.stack(ls), jnp.stack(vs), h


class RecurrentQ(nn.Module):
    n_actions: int

    def setup(self):
        self.feat = nn.Dense(REC_H)
        self.w_ir = nn.Dense(REC_H)
        self.w_hr = nn.Dense(REC_H, use_bias=False)
        self.w_iz = nn.Dense(REC_H)
        self.w_hz = nn.Dense(REC_H, use_bias=False)
        self.w_in = nn.Dense(REC_H)
        self.w_hn = nn.Dense(REC_H, use_bias=False)
        self.b_hn = self.param("b_hn", nn.initializers.zeros, (REC_H,))
        self.q = nn.Dense(self.n_actions)

    def gru(self, h, x):
        r = jax.nn.sigmoid(self.w_ir(x) + self.w_hr(h))
        z = jax.nn.sigmoid(self.w_iz(x) + self.w_hz(h))
        n = jnp.tanh(self.w_in(x) + r * (self.w_hn(h) + self.b_hn))
        return (1 - z) * n + z * h

    def __call__(self, obs_seq, h0, dones=None):
        h = h0
        outs = []
        for t in range(obs_seq.shape[0]):
            if dones is not None:
                m = (1 - dones[t].astype(jnp.float32))[:, None]
                h = h * m
            f = nn.relu(self.feat(obs_seq[t]))
            h = self.gru(h, f)
            outs.append(self.q(h))
        return jnp.stack(outs), h


def make_ppo_seq_update(model, optimizer, clip_range=0.2, vf_coef=0.5,
                        ent_coef=0.0):
    """PPO over minibatches of SEQUENCES. Time-major input (T,S,...);
    full BPTT in window T."""

    @jax.jit
    def update(state, obs, act, old_logp, adv, ret, h0, dones):
        # Time-major input (T,S,...) — matches loop buffers.
        params, opt_state = state
        ot, at, lt, adt, rt, dt = obs, act, old_logp, adv, ret, dones

        def loss_fn(p):
            logits, value, _ = model.apply(p, ot, h0, dt)
            logp_all = jax.nn.log_softmax(logits)
            logp = jnp.take_along_axis(logp_all, at[..., None], -1).squeeze(-1)
            ratio = jnp.exp(logp - lt)
            pg = -jnp.mean(jnp.minimum(
                ratio * adt,
                jnp.clip(ratio, 1 - clip_range, 1 + clip_range) * adt))
            v = jnp.mean(jnp.maximum(
                (value - rt) ** 2,
                (rt + jnp.clip(value - rt, -clip_range, clip_range)
                 - rt) ** 2)) / 2.0
            ent = -jnp.mean(jnp.sum(jax.nn.softmax(logits) * logp_all, -1))
            return pg + vf_coef * v - ent_coef * ent

        loss, grads = jax.value_and_grad(loss_fn)(params)
        upd, opt_state = optimizer.update(grads, opt_state, params)
        return (optax.apply_updates(params, upd), opt_state), loss

    return update


def make_ql_seq_update(qnet, mixer, optimizer, gamma=0.99, kind="vdn"):
    """Recurrent VDN/QMIX: Q per sequence (h0=zero, documented
    approximation), feedforward mixer per step, TD with target net."""

    @jax.jit
    def update(params, opt_state, tgt_params, obs, act, rew, obs2, state,
               state2, done):
        # obs: (S,L,A,O); act: (S,L,A); rew/done: (S,L); states: (S,L,St)
        S, L, A = act.shape
        h0 = jnp.zeros((S * A, REC_H))
        dn = jnp.broadcast_to(done[:, :, None], (S, L, A))

        def agent_q(qp, o):
            # (S,L,A,O) -> time-major (L,S*A,O) for the module.
            ot = o.transpose(1, 0, 2, 3).reshape(L, S * A, -1)
            dnt = dn.transpose(1, 0, 2).reshape(L, S * A)
            q, _ = qnet.apply(qp, ot, h0, dnt)
            return q.reshape(L, S, A, -1).transpose(1, 0, 2, 3)

        def loss_fn(p):
            q = agent_q(p["q"], obs)
            qa = q[jnp.arange(S)[:, None, None], jnp.arange(L)[None, :, None],
                   jnp.arange(A)[None, None, :], act]
            if kind == "vdn":
                qtot = qa.sum(-1)
            else:
                qtot = mixer.apply(
                    p["mix"], qa.reshape(S * L, A),
                    state.reshape(S * L, -1)).reshape(S, L)
            tq = agent_q(tgt_params["q"], obs2)
            ta_max = tq.max(-1)  # (S,L,A) max per agent
            if kind == "vdn":
                next_qtot = ta_max.sum(-1)
            else:
                next_qtot = mixer.apply(
                    tgt_params["mix"], ta_max.reshape(S * L, A),
                    state2.reshape(S * L, -1)).reshape(S, L)
            ttot = rew + gamma * (1.0 - done) * jax.lax.stop_gradient(next_qtot)
            return ((qtot - jax.lax.stop_gradient(ttot)) ** 2).mean()

        loss, grads = jax.value_and_grad(loss_fn)(params)
        upd, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, upd), opt_state, loss

    return update
