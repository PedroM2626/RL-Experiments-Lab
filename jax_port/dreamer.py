"""Full Dreamer in Flax (extension; closed loop, not only the RSSM).

DreamerV3-lite components, all Flax, discrete ProcGen:
  RSSM: ClassicCNN encoder (1024) + GRU deterministic 256 + 64D gaussian
    latent (prior p(z|h) / posterior q(z|h,feat)) + decoders (obs/reward/continue)
  WM losses: recon BCE + reward MSE + continue BCE + KL dyn 0.5 / rep 0.1
  behaviour: MLP(512) actor+critic trained in IMAGINATION
    (H=15 latent rollouts from the prior), lambda-returns, EMA critic,
    entropy 3e-4. The policy acts in the real env only to fill the buffer.
Documented simplifications vs V3: gaussian latent (not 32x32
categorical), GRU-256 (not 512+), raw MSE reward loss (no symlog/twohot),
no free-nats / KL clipping.
Loop: real collection (actor) -> WM updates on BxL buffer sequences
  -> imagination + actor/critic updates -> repeat.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp

DET = 256
STOCH = 64
HIDDEN = 512


class Encoder(nn.Module):
    @nn.compact
    def __call__(self, x):
        from jax_port.backbones import ClassicCNN
        return ClassicCNN()(x)  # 1024D (VALID, ~600k)


class RSSM(nn.Module):
    n_actions: int = 15

    def setup(self):
        self.cell = nn.GRUCell(DET)
        self.prior = nn.Dense(STOCH * 2)
        self.post = nn.Dense(STOCH * 2)

    def initial(self, b):
        return {"h": jnp.zeros((b, DET)),
                "z": jnp.zeros((b, STOCH))}

    def observe(self, prev, act_prev, feat, key):
        # prev: {h, z}; act_prev: acao que levou a feat
        x = jnp.concatenate([prev["z"], jax.nn.one_hot(act_prev, self.n_actions)], -1)
        h, _ = self.cell(prev["h"], x)  # GRUCell devolve (carry, y)
        mu, logvar = jnp.split(self.post(jnp.concatenate([h, feat], -1)), 2, -1)
        key, ks = jax.random.split(key)
        z = mu + jnp.exp(0.5 * logvar) * jax.random.normal(ks, mu.shape)
        return {"h": h, "z": z, "mu": mu, "logvar": logvar}

    def imagine(self, prev, act, key):
        x = jnp.concatenate([prev["z"], jax.nn.one_hot(act, self.n_actions)], -1)
        h, _ = self.cell(prev["h"], x)
        mu, logvar = jnp.split(self.prior(h), 2, -1)
        key, ks = jax.random.split(key)
        z = mu + jnp.exp(0.5 * logvar) * jax.random.normal(ks, mu.shape)
        return {"h": h, "z": z, "mu": mu, "logvar": logvar}

    def feat(self, s):
        return jnp.concatenate([s["h"], s["z"]], -1)


class Decoders(nn.Module):
    n_channels: int = 3

    def setup(self):
        from jax_port.dream import Decoder
        self.dec = Decoder(n_channels=self.n_channels)
        self.rew = nn.Dense(1)
        self.cont = nn.Dense(1)

    def __call__(self, f):
        obs = self.dec(f)  # estado completo (h,z)
        return obs, self.rew(f).squeeze(-1), self.cont(f).squeeze(-1)


class ActorCriticLatent(nn.Module):
    n_actions: int = 15

    @nn.compact
    def __call__(self, f):
        h = nn.relu(nn.Dense(HIDDEN)(f))
        h = nn.relu(nn.Dense(HIDDEN)(h))
        return nn.Dense(self.n_actions)(h), nn.Dense(1)(h).squeeze(-1)
