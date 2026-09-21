"""SPR (Self-Predictive Representations) — EXTENSION beyond the original ProcGen study.

Honest label: the study `4f84ed3` DOES NOT have SPR (SPR/CURL/CPC/ACL were from
the removed Craftax phase). Here SPR runs as an extra `spr` suite, with the
same rigor, but marked as an extension — never mixed into the core conclusions
of Sections 1-12 / parity.

Method (Schwarzer et al. 2021, adapted to PPO loop):
  ONLINE encoder = policy backbone (SHARED: auxiliary gradient
    flows into the backbone) + target encoder = EMA copy (tau=0.99/iter);
  MLP transition model (512+15)->512->512 predicts z_{t+1} from (z_t, a_t);
  loss = MSE(pred, target) over L2-normalized latents, with augmented
    views (own crop, p=1.0) of o_t and o_{t+1};
  auxiliary Adam optimizer 1e-4 (precedent ICM/RND from study), spr_coef=1.0
  (added to PPO loss in SPR phase, in pairs minibatches).
Requires pixel CNN backbone (not mlp/vae).
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from jax_port.augment import make_augment

SPR_COEF = 1.0
SPR_LR = 1e-4
SPR_TAU = 0.99


class TransitionMLP(nn.Module):
    n_actions: int = 15

    @nn.compact
    def __call__(self, z_a):
        h = nn.relu(nn.Dense(512)(z_a))
        return nn.Dense(512)(h)


def _norm(x, eps=1e-6):
    # sqrt(sum+eps): denominator is never exactly zero (same as contrast.py).
    n = jnp.sqrt((x ** 2).sum(axis=-1, keepdims=True) + eps)
    return x / n


def make_spr(backbone_cls, n_actions=15):
    """Returns dict with init/step/ema over the shared backbone."""
    backbone = backbone_cls()
    trans = TransitionMLP(n_actions=n_actions)
    aug = make_augment("crop", p=1.0)
    opt_b = optax.adam(SPR_LR)
    opt_h = optax.adam(SPR_LR)

    def encode(bb_params, x_f, key=None):
        del key
        return backbone.apply({"params": bb_params}, x_f)

    @jax.jit
    def step(bb_p, bb_os, h_p, h_os, tgt_p, ot_u8, act, on_u8, key):
        k1, k2, k3 = jax.random.split(key, 3)
        xt = aug(ot_u8.astype(jnp.float32) / 255.0, k1)
        xn = aug(on_u8.astype(jnp.float32) / 255.0, k2)
        a_oh = jax.nn.one_hot(act, n_actions)

        def loss_fn(bb, tr):
            z = encode(bb, xt)
            pred = trans.apply(tr,
                               jnp.concatenate([z, a_oh], axis=1))
            tgt = backbone.apply({"params": tgt_p}, xn)
            return ((_norm(pred) - _norm(tgt)) ** 2).mean()

        (l2, grads) = jax.value_and_grad(
            lambda b, t: loss_fn(b, t), argnums=(0, 1))(bb_p, h_p)
        upd_b, bb_os = opt_b.update(grads[0], bb_os, bb_p)
        upd_h, h_os = opt_h.update(grads[1], h_os, h_p)
        bb_p = optax.apply_updates(bb_p, upd_b)
        h_p = optax.apply_updates(h_p, upd_h)
        return bb_p, bb_os, h_p, h_os, l2 * SPR_COEF

    @jax.jit
    def ema(tgt_p, bb_p):
        return jax.tree.map(lambda t, o: SPR_TAU * t + (1.0 - SPR_TAU) * o,
                            tgt_p, bb_p)

    def init_head(key):
        return trans.init(key, jnp.zeros((1, 512 + n_actions)))

    return {"step": step, "ema": ema, "init_head": init_head,
            "opt_b": opt_b, "opt_h": opt_h, "backbone": backbone}
