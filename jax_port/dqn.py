"""DQN / QR-DQN in JAX — parity with ``compare_algo_families.py:32`` (§12).

Study hyperparameters (adaptations for small budget): 100k buffer (uint8 ring
buffer for images), learning_starts 5000, exploration_fraction 0.25
(eps 1.0->0.05 linear), lr 1e-4, train_freq 4, gradient_steps 1,
target_update 500, batch 64, gamma 0.99; QR: 200 quantiles, Huber kappa 1.
Standard DQN (no double; SB3 does not use double by default).

Batching adjustment (documented): study uses n_envs=1; here parallel N envs
and updates per transition maintained at ~1/4 (N transitions per env
iteration -> N/4 grad-steps of batch 64 per iteration).
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_port.backbones import BACKBONES


class QNet(nn.Module):
    backbone: nn.Module
    n_actions: int = 15
    quantiles: int = 0  # 0 = scalar DQN; >0 = QR-DQN with N quantiles

    @nn.compact
    def __call__(self, x):
        feat = self.backbone(x)
        if self.quantiles:
            q = nn.Dense(self.n_actions * self.quantiles)(feat)
            return q.reshape((-1, self.n_actions, self.quantiles))
        return nn.Dense(self.n_actions)(feat)


def make_dqn_update(net, optimizer, gamma=0.99, quantiles=0, kappa=1.0):
    @jax.jit
    def update(params, opt_state, target_params, obs, act, rew, obs2, done):
        o = obs.astype(jnp.float32) / 255.0
        o2 = obs2.astype(jnp.float32) / 255.0

        def loss_fn(p):
            if quantiles:
                quant = net.apply(p, o)  # (B,A,N)
                qa = quant[jnp.arange(obs.shape[0]), act]  # (B,N)
                tq = net.apply(target_params, o2)  # (B,A,N)
                ta = tq.max(axis=1)  # SB3/sb3-contrib: max over actions
                t = rew[:, None] + gamma * (1.0 - done[:, None]) * ta
                t = jax.lax.stop_gradient(t)
                diff = t[:, None, :] - qa[:, :, None]
                hub = jnp.where(jnp.abs(diff) <= kappa,
                                0.5 * diff ** 2, kappa * (jnp.abs(diff) - 0.5 * kappa))
                taus = (jnp.arange(quantiles, dtype=jnp.float32) + 0.5) / quantiles
                w = jnp.abs(taus[None, :, None] - (diff < 0).astype(jnp.float32))
                return (w * hub).mean()
            else:
                q = net.apply(p, o)
                qa = q[jnp.arange(obs.shape[0]), act]
                tq = net.apply(target_params, o2)
                t = rew + gamma * (1.0 - done) * tq.max(axis=1)
                t = jax.lax.stop_gradient(t)
                return ((qa - t) ** 2).mean()

        loss, grads = jax.value_and_grad(loss_fn)(params)
        upd, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, upd), opt_state, loss

    @jax.jit
    def greedy(params, obs):
        q = net.apply(params, obs.astype(jnp.float32) / 255.0)
        if quantiles:
            q = q.mean(axis=-1)
        return q.argmax(axis=1)

    return update, greedy


class ReplayBuffer:
    """Anel numpy uint8 (obs, act, rew, obs2, done)."""

    def __init__(self, capacity, obs_shape):
        self.cap, self.obs_shape = capacity, obs_shape
        self.obs = np.empty((capacity,) + obs_shape, np.uint8)
        self.obs2 = np.empty((capacity,) + obs_shape, np.uint8)
        self.act = np.empty(capacity, np.int32)
        self.rew = np.empty(capacity, np.float32)
        self.done = np.empty(capacity, bool)
        self.i, self.full = 0, False

    def add(self, o, a, r, o2, d):
        n = o.shape[0]
        for k in range(n):
            j = (self.i + k) % self.cap
            self.obs[j], self.act[j] = o[k], a[k]
            self.rew[j], self.obs2[j], self.done[j] = r[k], o2[k], d[k]
        self.i = (self.i + n) % self.cap
        self.full = self.full or self.i + n >= self.cap or self.i == 0

    def __len__(self):
        return self.cap if self.full else self.i

    def sample(self, rng, batch):
        idx = rng.integers(0, len(self), size=batch)
        return (self.obs[idx], self.act[idx], self.rew[idx],
                self.obs2[idx], self.done[idx])
