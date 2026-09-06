"""CPU: zoo temporal shapes + StackVec (sem GPU)."""
import jax
import jax.numpy as jnp
import numpy as np


def test_temporal_shapes():
    from jax_port.networks import ActorCritic, ActorCriticXL
    from jax_port.temporal import MEM_DIM, MEM_LEN, TEMPORAL, make_mem_fns
    from jax_port.ppo import make_optimizer
    key = jax.random.PRNGKey(0)
    for name, cls in TEMPORAL.items():
        Dun = (2, 64, 64, 12)
        dummy = jnp.zeros(Dun, jnp.float32)
        if name == "transformer_xl":
            m = ActorCriticXL(backbone=cls(), n_actions=15)
            key, k = jax.random.split(key)
            p = m.init(k, dummy, jnp.zeros((2, MEM_LEN, MEM_DIM), jnp.float32))
            opt = make_optimizer()
            up, ro, fw = make_mem_fns(m, opt)
            key, k = jax.random.split(key)
            (p2, _), loss = up(
                (p, opt.init(p)), jnp.zeros((4, 64, 64, 12), jnp.uint8),
                jnp.zeros((4, MEM_LEN, MEM_DIM), jnp.float32),
                jnp.zeros((4,), jnp.int32), jnp.zeros((4,)),
                jnp.zeros((4,)), jnp.zeros((4,)))
            assert bool(jnp.isfinite(loss)), name
        else:
            m = ActorCritic(backbone=cls(), n_actions=15)
            key, k = jax.random.split(key)
            p = m.init(k, dummy, None)
            logits, value = m.apply(p, dummy, None)
            assert tuple(logits.shape) == (2, 15), (name, logits.shape)
            assert tuple(value.shape) == (2,), name
        n = sum(x.size for x in jax.tree_util.tree_leaves(p))
        print(f"{name:14s} params={n:>9,} OK", flush=True)
    print("TEMPORAL_OK")


def test_stack():
    from jax_port.stack_env import StackVec
    v = StackVec("coinrun", num_envs=4, num_levels=200, seed=0, k=4)
    s = v.reset()
    assert s.shape == (4, 64, 64, 12) and s.dtype == np.uint8, s.shape
    v.act(np.zeros(4, np.int32))
    rew, d, first = v.observe()
    assert set(d.keys()) == {"rgb_stacked"}
    assert np.asarray(d["rgb_stacked"]).shape == (4, 64, 64, 12)
    print("STACK_OK")


if __name__ == "__main__":
    test_temporal_shapes()
    test_stack()
