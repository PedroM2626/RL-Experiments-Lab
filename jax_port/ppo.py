"""JIT-compiled PPO update (hyperparameters faithful to study, §1.4 + SB3 defaults).

Study (``compare_suite.py:26``): lr 3e-4, n_steps 256, batch 64,
n_epochs 3, gamma 0.99, gae_lambda 0.95, clip 0.2 (+ SB3 defaults:
vf_coef 0.5, ent_coef 0.01, max_grad_norm 0.5, normalize_advantage True).

Throughput adjustment (documented): study uses n_envs=1 (single DummyVecEnv)
with minibatches of 64 — optimal for head-to-head comparison, poor for GPU
(hundreds of tiny grad-steps per iteration). Here parallelism changes
(N envs x T steps, minibatch >= 1024) while PPO mathematical objective stays
intact: identical clipped surrogate, value clipping, and entropy bonus.
"""

import jax
import jax.numpy as jnp
import optax


def make_optimizer(lr=3e-4, max_grad_norm=0.5, kind="adam"):
    # SB3: PPO uses Adam; A2C uses RMSprop (default). Study: A2C lr 3e-4.
    if kind == "rmsprop":
        return optax.chain(optax.clip_by_global_norm(max_grad_norm),
                           optax.rmsprop(lr, eps=1e-5))
    return optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(lr, eps=1e-5),
    )


def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    """GAE in numpy (T,N) -> advantages, returns. Inexpensive: T~128, N~64."""
    import numpy as np
    T, N = rewards.shape
    adv = np.zeros_like(rewards)
    lastgaelam = np.zeros(N, dtype=np.float32)
    for t in reversed(range(T)):
        nonterm = 1.0 - dones[t].astype(np.float32)
        nv = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * nv * nonterm - values[t]
        lastgaelam = delta + gamma * lam * nonterm * lastgaelam
        adv[t] = lastgaelam
    return adv, (adv + values).astype(np.float32)


def make_update_fn(model, optimizer, clip_range=0.2, vf_coef=0.5, ent_coef=0.01):
    """Returns (update_jit, rollout_jit, forward_jit, norm).

    Every value that can live on device stays on device: update receives
    contiguous uint8 obs (preprocess inside JIT) and rollout samples
    action+logp inside JIT (one dispatch+sync per env step).
    MEASURED NOTE: slicing minibatch with device index
    (``d_obs[d_mb]``) costs ~457ms/call in this setup; host slicing +
    contiguous H2D costs ~12ms/call (~40x speedup). Minibatch loop thus uses
    numpy slicing + per-call H2D (see ``train.py``).
    ``key`` is always passed (stochastic backbones, e.g. VAE,
    sample fresh z per forward — matching the study); deterministic backbones
    ignore it.
    """

    def loss_fn(params, obs_u8, act, old_logp, adv, ret, key):
        obs = obs_u8.astype(jnp.float32) / 255.0
        logits, value = model.apply(params, obs, key)
        logp_all = jax.nn.log_softmax(logits)
        logp = jnp.take_along_axis(logp_all, act[:, None], axis=1).squeeze(1)
        ratio = jnp.exp(logp - old_logp)
        if clip_range is None:
            # A2C: pure policy gradient, no clipping (SB3 default).
            pg_loss = -jnp.mean(ratio * adv)
        else:
            pg1 = ratio * adv
            pg2 = jnp.clip(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv
            pg_loss = -jnp.mean(jnp.minimum(pg1, pg2))
        if clip_range is None:
            v_loss = (jnp.mean((value - ret) ** 2)) / 2.0
        else:
            v_clipped = ret + jnp.clip(value - ret, -clip_range, clip_range)
            v_loss = jnp.mean(jnp.maximum((value - ret) ** 2,
                                          (v_clipped - ret) ** 2)) / 2.0
        entropy = -jnp.mean(jnp.sum(jax.nn.softmax(logits) * logp_all, axis=1))
        return pg_loss + vf_coef * v_loss - ent_coef * entropy

    @jax.jit
    def update_real(state, obs_u8, act, old_logp, adv, ret, key):
        params, opt_state = state
        loss, grads = jax.value_and_grad(loss_fn)(params, obs_u8, act, old_logp, adv, ret, key)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss

    @jax.jit
    def rollout_step(params, obs_u8, key):
        key, kz, ks = jax.random.split(key, 3)
        obs = obs_u8.astype(jnp.float32) / 255.0
        logits, value = model.apply(params, obs, kz)
        act = jax.random.categorical(ks, logits)
        logp = jax.nn.log_softmax(logits)[jnp.arange(logits.shape[0]), act]
        return act, logp, value, key

    @jax.jit
    def forward(params, obs, key):
        return model.apply(params, obs, key)

    @jax.jit
    def normalize_adv(adv):
        return (adv - adv.mean()) / (adv.std() + 1e-8)

    return update_real, rollout_step, forward, normalize_adv
