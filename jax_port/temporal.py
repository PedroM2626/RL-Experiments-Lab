"""Zoo temporal — bake-off de memoria sobre frames empilhados (EXTENSAO).

Todos recebem (B,64,64,12) = 4 frames NHWC (ou (B,64,64,3*K)) e devolvem
512D, plugaveis no loop PPO (frame_stack via StackGym3/train --stack).
mlp: mean-pool 16x16 + MLP[256,256] (baseline sem tempo).
cnn1d: encoder/frame + Conv1D k3 + mean.
tcn: encoder/frame + Conv1D causal dilatada d=1,2 + residual, last-step.
lstm/gru: encoder/frame + recorrencia real sobre a pilha (carry so
  dentro da pilha; entre steps stateless — upgrade fiel do fake-repeat
  do estudo) + last hidden.
transformer: encoder/frame + pos + Transformer x2 + mean.
transformer_xl: como transformer + memoria de segmento (mems do stack
  anterior, carry no loop de treino) — unico com estado entre steps.
mamba: selective scan simplificado (B,C,Δ dependentes de x, SiLU gate,
  sem conv1d-prelude do paper) sobre Dense(128) dos frames.
s4: S4D diagonal real (A=-exp, FFT conv) por canal.
s5: SSM MIMO (HiPPO-diag, associative scan) + skip.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp

from jax_port.backbones import ClassicCNN, _TransformerBlock


def _frames(x, k=4):
    """(B,64,64,3K) -> (B,K,64,64,3)."""
    b = x.shape[0]
    return x.reshape(b, 64, 64, k, 3).transpose(0, 3, 1, 2, 4)


def _encode_each(enc, x, k=4):
    b = x.shape[0]
    f = enc(_frames(x, k).reshape(b * k, 64, 64, 3))
    return f.reshape(b, k, -1)  # (B,K,1024)


class MlpStack(nn.Module):
    @nn.compact
    def __call__(self, x):
        s = x.reshape(x.shape[0], 16, 4, 16, 4, 12).mean(axis=(2, 4))
        s = s.reshape((s.shape[0], -1))
        s = nn.relu(nn.Dense(256)(s))
        s = nn.relu(nn.Dense(256)(s))
        return nn.relu(nn.Dense(512)(s))


class Cnn1D(nn.Module):
    @nn.compact
    def __call__(self, x):
        f = _encode_each(ClassicCNN(), x)  # (B,4,1024)
        h = nn.relu(nn.Conv(128, (3,), padding="SAME")(f))
        return nn.relu(nn.Dense(512)(h.mean(axis=1)))


class TCN(nn.Module):
    @nn.compact
    def __call__(self, x):
        f = nn.relu(nn.Dense(128)(_encode_each(ClassicCNN(), x)))  # (B,4,128)
        h = f
        for d in (1, 2):
            # causal: pad a esquerda, corta a direita (VALID apos pad)
            y = jnp.pad(h, ((0, 0), ((2 - 1) * d, 0), (0, 0)))
            y = nn.Conv(128, (2,), strides=1,
                        padding="VALID", kernel_dilation=(d,))(y)
            y = nn.relu(y)
            h = h + y if h.shape == y.shape else y
        return nn.relu(nn.Dense(512)(h[:, -1]))


class LSTMStack(nn.Module):
    hidden: int = 256

    @nn.compact
    def __call__(self, x):
        f = _encode_each(ClassicCNN(), x)
        cell = nn.LSTMCell(self.hidden)
        c = cell.initialize_carry(jax.random.PRNGKey(0), (f.shape[0],))
        hs = []
        for t in range(f.shape[1]):
            c, h = cell(c, f[:, t])
            hs.append(h)
        return nn.relu(nn.Dense(512)(hs[-1]))


class GRUStack(nn.Module):
    hidden: int = 256

    @nn.compact
    def __call__(self, x):
        f = _encode_each(ClassicCNN(), x)
        cell = nn.GRUCell(self.hidden)
        h = jnp.zeros((f.shape[0], self.hidden))
        for t in range(f.shape[1]):
            h, _ = cell(h, f[:, t])
        return nn.relu(nn.Dense(512)(h))


class TransformerStack(nn.Module):
    dim: int = 128
    layers: int = 2

    @nn.compact
    def __call__(self, x):
        f = nn.Dense(self.dim)(_encode_each(ClassicCNN(), x))
        pos = self.param("pos_emb", nn.initializers.normal(0.02),
                         (1, f.shape[1], self.dim))
        h = f + pos
        for _ in range(self.layers):
            h = _TransformerBlock(self.dim)(h)
        return nn.relu(nn.Dense(512)(h.mean(axis=1)))


class TransformerXL(nn.Module):
    dim: int = 128
    layers: int = 2

    @nn.compact
    def __call__(self, x, mem):
        # x: (B,K,64,64,3) frames; mem: (B,M,dim) segmento anterior.
        f = nn.Dense(self.dim)(_encode_each(ClassicCNN(), x))
        pos = self.param("pos_emb", nn.initializers.normal(0.02),
                         (1, f.shape[1], self.dim))
        h = f + pos
        for _ in range(self.layers):
            h = _TransformerBlockXL(self.dim)(h, mem)
        new_mem = jax.lax.stop_gradient(h)
        return nn.relu(nn.Dense(512)(h.mean(axis=1))), new_mem


class _TransformerBlockXL(nn.Module):
    dim: int
    heads: int = 4
    mlp: int = 256

    @nn.compact
    def __call__(self, x, mem):
        h = nn.LayerNorm()(x)
        full = jnp.concatenate([jax.lax.stop_gradient(mem), h], axis=1)
        h = nn.MultiHeadAttention(num_heads=self.heads)(h, full)
        x = x + h
        h = nn.LayerNorm()(x)
        h = nn.Dense(self.mlp)(h)
        h = nn.gelu(h)
        h = nn.Dense(self.dim)(h)
        return x + h


class MambaLayer(nn.Module):
    dim: int = 128
    state: int = 16

    @nn.compact
    def __call__(self, x):
        # x: (B,K,dim). Scan seletivo simplificado (paper sem conv1d).
        Bm = nn.Dense(self.state)(x)  # B(x)
        Cm = nn.Dense(self.state)(x)  # C(x)
        dt = nn.softplus(nn.Dense(self.dim)(x))  # Delta(x) > 0
        A = -jnp.exp(self.param("A_log", nn.initializers.uniform(),
                                (self.dim, self.state)))
        Dm = self.param("D_skip", nn.initializers.ones, (self.dim,))
        h = jnp.zeros((x.shape[0], self.dim, self.state))
        ys = []
        for t in range(x.shape[1]):
            dA = jnp.exp(dt[:, t, :, None] * A[None, :, :])
            dB = dt[:, t, :, None] * Bm[:, t, None, :]
            h = dA * h + dB * x[:, t, :, None]
            y = (h * Cm[:, t, None, :]).sum(-1)
            ys.append(y + Dm * x[:, t])
        y = jnp.stack(ys, axis=1)
        g = nn.silu(nn.Dense(self.dim)(x))
        return nn.LayerNorm()(x + y * g)


class MambaStack(nn.Module):
    dim: int = 128

    @nn.compact
    def __call__(self, x):
        f = nn.Dense(self.dim)(_encode_each(ClassicCNN(), x))
        h = MambaLayer(self.dim)(f)
        h = MambaLayer(self.dim)(h)
        return nn.relu(nn.Dense(512)(h[:, -1]))


class S4DLayer(nn.Module):
    dim: int = 128

    @nn.compact
    def __call__(self, x):
        # S4D diagonal real: nucleo exp(A t), conv via FFT por canal.
        A = -jnp.exp(self.param("A_log", nn.initializers.uniform(), (self.dim,)))
        B = self.param("B", nn.initializers.normal(), (self.dim,))
        C = self.param("C", nn.initializers.normal(), (self.dim,))
        K = x.shape[1]
        t = jnp.arange(K)
        kernel = (C[None, :] * jnp.exp(A[None, :] * t[:, None])).T  # (dim,K)
        Xf = jnp.fft.rfft(x.transpose(0, 2, 1), n=2 * K, axis=-1)
        Kf = jnp.fft.rfft(kernel[None, :, :], n=2 * K, axis=-1)
        y = jnp.fft.irfft(Xf * Kf, n=2 * K, axis=-1)[..., :K]
        y = y.transpose(0, 2, 1) + self.param(
            "D_skip", nn.initializers.zeros, (self.dim,)) * x
        return nn.gelu(y)


class S4Stack(nn.Module):
    dim: int = 128

    @nn.compact
    def __call__(self, x):
        f = nn.Dense(self.dim)(_encode_each(ClassicCNN(), x))
        h = S4DLayer(self.dim)(f)
        h = S4DLayer(self.dim)(h)
        return nn.relu(nn.Dense(512)(h[:, -1]))


class S5Layer(nn.Module):
    dim: int = 128
    state: int = 64

    @nn.compact
    def __call__(self, x):
        # SSM MIMO: h = A h + B x, y = C h + skip; scan associativo.
        A = -jnp.exp(self.param("A_log", nn.initializers.uniform(),
                                (self.state,)))
        Bm = self.param("B", nn.initializers.normal(), (self.dim, self.state))
        Cm = self.param("C", nn.initializers.normal(), (self.state, self.dim))
        Dm = self.param("D_skip", nn.initializers.zeros, (self.dim,))

        def binary(op_a, op_b):
            Aa, ba = op_a
            Ab, bb = op_b
            return Aa * Ab, Ab * ba + bb

        B, T_ = x.shape[0], x.shape[1]
        dA = jnp.exp(A)[None, None, :]
        Bx = jnp.einsum("btd,dp->btp", x, Bm)
        dAt = jnp.broadcast_to(dA, (B, T_, self.state))
        # scan sobre o tempo: poe tempo no eixo 0
        _, ys_t = jax.lax.associative_scan(
            binary, (dAt.transpose(1, 0, 2), Bx.transpose(1, 0, 2)))
        ys = ys_t.transpose(1, 0, 2)
        y = jnp.einsum("btp,pd->btd", ys, Cm) + Dm * x
        return nn.gelu(y)


class S5Stack(nn.Module):
    dim: int = 128

    @nn.compact
    def __call__(self, x):
        f = nn.Dense(self.dim)(_encode_each(ClassicCNN(), x))
        h = S5Layer(self.dim)(f)
        h = S5Layer(self.dim)(h)
        return nn.relu(nn.Dense(512)(h[:, -1]))


TEMPORAL = {
    "mlp": MlpStack,
    "cnn1d": Cnn1D,
    "tcn": TCN,
    "lstm": LSTMStack,
    "gru": GRUStack,
    "transformer": TransformerStack,
    "transformer_xl": TransformerXL,
    "mamba": MambaStack,
    "s4": S4Stack,
    "s5": S5Stack,
}

MEM_DIM = 128
MEM_LEN = 4


def make_mem_fns(model, optimizer, clip_range=0.2, vf_coef=0.5,
                 ent_coef=0.01):
    """Irmaos com memoria de ppo.make_update_fn (só p/ transformer_xl).

    rollout/update/forward levam (obs, mem) e devolvem new_mem; a
    matematica PPO e identica. Mems guardadas no rollout e reusadas
    nos updates (padrao RecurrentPPO/SB3-contrib).
    """
    import jax
    import jax.numpy as jnp
    import optax

    def loss_fn(params, obs_u8, mem, act, old_logp, adv, ret):
        obs = obs_u8.astype(jnp.float32) / 255.0
        logits, value, _ = model.apply(params, obs, mem)
        logp_all = jax.nn.log_softmax(logits)
        logp = jnp.take_along_axis(logp_all, act[:, None], axis=1).squeeze(1)
        ratio = jnp.exp(logp - old_logp)
        pg = -jnp.mean(jnp.minimum(
            ratio * adv, jnp.clip(ratio, 1 - clip_range, 1 + clip_range) * adv))
        v = jnp.mean(jnp.maximum((value - ret) ** 2, (ret + jnp.clip(
            value - ret, -clip_range, clip_range) - ret) ** 2)) / 2.0
        ent = -jnp.mean(jnp.sum(jax.nn.softmax(logits) * logp_all, axis=1))
        return pg + vf_coef * v - ent_coef * ent

    @jax.jit
    def update(state, obs_u8, mem, act, old_logp, adv, ret):
        params, opt_state = state
        loss, grads = jax.value_and_grad(loss_fn)(params, obs_u8, mem, act,
                                                  old_logp, adv, ret)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss

    @jax.jit
    def rollout(params, obs_u8, mem, key):
        obs = obs_u8.astype(jnp.float32) / 255.0
        logits, value, new_mem = model.apply(params, obs, mem)
        key, ks = jax.random.split(key)
        act = jax.random.categorical(ks, logits)
        logp = jax.nn.log_softmax(logits)[jnp.arange(logits.shape[0]), act]
        return act, logp, value, new_mem, key

    @jax.jit
    def forward(params, obs, mem):
        return model.apply(params, obs, mem)

    return update, rollout, forward
