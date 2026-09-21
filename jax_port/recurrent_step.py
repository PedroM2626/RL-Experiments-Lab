"""Modelos recorrentes passo-a-passo com carry entre transições do ambiente (stack=1).

Diferente do temporal.py (que fazia recorrencia interna apenas dentro dos 4 frames da pilha),
estes modelos recebem um unico frame (B, 64, 64, 3) e um estado de memoria (B, L, D),
atualizando o estado a cada passo do ambiente (t -> t+1) com carry persistente no rollout
e reset no done.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax_port.backbones import ClassicCNN


class S5StepLayer(nn.Module):
    dim: int = 128
    state: int = 64

    @nn.compact
    def __call__(self, x, h_prev):
        # x: (B, dim), h_prev: (B, state)
        A = -jnp.exp(self.param("A_log", nn.initializers.uniform(), (self.state,)))
        Bm = self.param("B", nn.initializers.normal(), (self.dim, self.state))
        Cm = self.param("C", nn.initializers.normal(), (self.state, self.dim))
        Dm = self.param("D_skip", nn.initializers.zeros, (self.dim,))

        dA = jnp.exp(A)[None, :]  # (1, state)
        u = jnp.matmul(x, Bm)     # (B, state)
        h_new = dA * h_prev + u   # (B, state)
        y = jnp.matmul(h_new, Cm) + Dm * x  # (B, dim)
        return nn.gelu(y), h_new


class RecurrentLSTMBackbone(nn.Module):
    hidden: int = 128

    @nn.compact
    def __call__(self, x, mem):
        # x: (B, 64, 64, 3), mem: (B, 2, hidden) onde [:, 0] e' c e [:, 1] e' h
        enc = ClassicCNN()
        f = nn.relu(nn.Dense(self.hidden)(enc(x)))  # (B, hidden)
        c_prev = mem[:, 0]
        h_prev = mem[:, 1]
        (c_new, h_new), _ = nn.LSTMCell(self.hidden)((c_prev, h_prev), f)
        new_mem = jax.lax.stop_gradient(jnp.stack([c_new, h_new], axis=1))
        out = nn.relu(nn.Dense(512)(h_new))
        return out, new_mem


class RecurrentS5Backbone(nn.Module):
    dim: int = 128
    state: int = 64

    @nn.compact
    def __call__(self, x, mem):
        # x: (B, 64, 64, 3), mem: (B, 2, state) para 2 camadas de S5
        enc = ClassicCNN()
        f = nn.relu(nn.Dense(self.dim)(enc(x)))  # (B, dim)
        h0_prev = mem[:, 0]
        h1_prev = mem[:, 1]
        y0, h0_new = S5StepLayer(self.dim, self.state, name="s5_0")(f, h0_prev)
        y1, h1_new = S5StepLayer(self.dim, self.state, name="s5_1")(y0, h1_prev)
        new_mem = jax.lax.stop_gradient(jnp.stack([h0_new, h1_new], axis=1))
        out = nn.relu(nn.Dense(512)(y1))
        return out, new_mem


RECURRENT_STEP = {
    "recurrent_lstm": (RecurrentLSTMBackbone, (2, 128)),
    "recurrent_s5": (RecurrentS5Backbone, (2, 64)),
}
