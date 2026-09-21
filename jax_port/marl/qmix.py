"""VDN + QMIX in Flax (MARL extension; parity with earlier phase §3.3).

VDN: Q_tot = sum_a Q_a (additive factorization).
QMIX: Q_tot = monotonic hypernetwork mixer (weights >= 0 via abs) over
  Q_a, conditioned on global state (72D). Embed 32.
Shared Q-networks: ActorOnly (MLP 128/128, n_actions output).
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from jax_port.marl.ppo_marl import ActorOnly


class QMixer(nn.Module):
    n_agents: int
    embed: int = 32

    @nn.compact
    def __call__(self, q_a, state):
        # q_a: (B,A); state: (B,S)
        w1 = jnp.abs(nn.Dense(self.n_agents * self.embed)(state))
        w1 = w1.reshape((-1, self.n_agents, self.embed))
        b1 = nn.Dense(self.embed)(state)
        h = nn.elu(jnp.einsum("ba,bae->be", q_a, w1) + b1)
        w2 = jnp.abs(nn.Dense(self.embed)(state)).reshape((-1, self.embed, 1))
        b2 = nn.Dense(1)(state).squeeze(-1)
        return (jnp.einsum("be,beo->bo", h, w2).squeeze(-1) + b2)


def make_ql_update(qnet, mixer, optimizer, gamma=0.99, kind="vdn"):
    @jax.jit
    def update(params, opt_state, tgt_params, obs, act, rew, obs2, state,
               state2, done):
        B, A = act.shape

        def joint_q(p, o, s):
            q = qnet.apply(p["q"], o.reshape(B * A, -1)).reshape(B, A, -1)
            qa = q[jnp.arange(B)[:, None], jnp.arange(A), act]
            if kind == "vdn":
                return qa.sum(-1), q
            return mixer.apply(p["mix"], qa, s), q

        def loss_fn(p):
            qtot, _ = joint_q(p, obs, state)
            tq_tot, _ = joint_q(tgt_params, obs2, state2)
            target = rew + gamma * (1.0 - done) * tq_tot
            return ((qtot - jax.lax.stop_gradient(target)) ** 2).mean()

        loss, grads = jax.value_and_grad(loss_fn)(params)
        upd, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, upd), opt_state, loss

    return update
