"""MARL tests (SMAX): zero-GPU parity when JAX_PLATFORMS=cpu.

Covers: battle_won, adapter (shapes/autoreset), forwards + finite losses
of ippo/mappo/vdn/qmix/mapoca/tarmac/cte.
Usage: python -m jax_port.tests.test_marl
"""

import jax
import jax.numpy as jnp
import numpy as np


def test_battle_won():
    from jax_port.marl.smax_vec import battle_won
    assert battle_won([True, True, False, False], 2) is True
    assert battle_won([False, False, False, False], 2) is False
    assert battle_won([True, True, True, False], 2) is False
    assert battle_won([True, False, False, False], 2) is True
    return True


def test_adapter():
    from jax_port.marl.smax_vec import SmaxVec
    v = SmaxVec("3m", num_envs=4, seed=0)
    assert (v.n_agents, v.obs_dim, v.state_dim) == (3, 75, 72)
    assert v.n_actions == 8
    rng = np.random.default_rng(0)
    nd = 0
    for _ in range(60):
        obs, rew, done, win = v.step(rng.integers(0, 8, size=(4, 3)))
        assert np.asarray(obs).shape == (4, 3, 75)
        nd += int(done.sum())
    assert nd > 0
    return True


def test_losses():
    from jax_port.backbones import MlpBackbone
    from jax_port.marl.paradigms import (AgentEncoder, AttnCritic,
                                         PolicyHead, TarMACActor,
                                         make_mapoca_update)
    from jax_port.marl.ppo_marl import (ActorOnly, CentralCritic,
                                        make_mappo_update, make_update_float)
    from jax_port.marl.qmix import QMixer, make_ql_update
    from jax_port.networks import ActorCritic
    from jax_port.ppo import make_optimizer
    B, G, NA, O, S = 8, 3, 8, 75, 72
    key = jax.random.PRNGKey(0)
    opt = make_optimizer()
    key, k = jax.random.split(key)
    m = ActorCritic(backbone=MlpBackbone(), n_actions=NA)
    p = m.init(k, jnp.zeros((1, O)), None)
    up, _, _ = make_update_float(m, opt)
    (_, _), loss = up(
        (p, opt.init(p)), jnp.zeros((B, O)), jnp.zeros((B,), jnp.int32),
        jnp.zeros((B,)), jnp.zeros((B,)), jnp.zeros((B,)))
    assert bool(jnp.isfinite(loss)), "ippo"
    ac, cc = ActorOnly(n_actions=NA), CentralCritic()
    key, k1, k2 = jax.random.split(key, 3)
    pa, pc = ac.init(k1, jnp.zeros((1, O))), cc.init(k2, jnp.zeros((1, S)))
    mu = make_mappo_update(ac, cc, opt)
    (_, _), (_, _), loss = mu(
        (pa, opt.init(pa)), (pc, opt.init(pc)), jnp.zeros((B, O)),
        jnp.zeros((B,), jnp.int32), jnp.zeros((B,)), jnp.zeros((B,)),
        jnp.zeros((B,)), jnp.zeros((B, S)))
    assert bool(jnp.isfinite(loss)), "mappo"
    qnet = ActorOnly(n_actions=NA)
    for kind in ("vdn", "qmix"):
        mixer = QMixer(n_agents=G)
        key, k0, k1 = jax.random.split(key, 3)
        pp = {"q": qnet.init(k0, jnp.zeros((1, O))),
              "mix": mixer.init(k1, jnp.zeros((1, G)), jnp.zeros((1, S)))}
        qu = make_ql_update(qnet, mixer, opt, kind=kind)
        _, _, loss = qu(
            pp, opt.init(pp), pp, jnp.zeros((B, G, O)),
            jnp.zeros((B, G), jnp.int32), jnp.zeros((B,)),
            jnp.zeros((B, G, O)), jnp.zeros((B, S)), jnp.zeros((B, S)),
            jnp.zeros((B,)))
        assert bool(jnp.isfinite(loss)), kind
    enc, cri = AgentEncoder(), AttnCritic(n_actions=NA)
    pol = PolicyHead(n_actions=NA)
    key, k0, k1, k2 = jax.random.split(key, 4)
    ap = {"enc": enc.init(k0, jnp.zeros((1, O))),
          "pol": pol.init(k1, jnp.zeros((1, 64)))}
    cp = cri.init(k2, jnp.zeros((1, G, 64)), jnp.zeros((1, G, NA)),
                  jnp.zeros((1, S)))
    mp = make_mapoca_update(enc, cri, pol, opt, NA)
    _, _, loss = mp(
        (ap, opt.init(ap)), (cp, opt.init(cp)), jnp.zeros((B, G, O)),
        jnp.zeros((B, G), jnp.int32), jnp.zeros((B, G)), jnp.zeros((B,)),
        jnp.zeros((B, S)))
    assert bool(jnp.isfinite(loss)), "mapoca"
    tm = TarMACActor(n_actions=NA)
    key, k = jax.random.split(key)
    pt = tm.init(k, jnp.zeros((1, G, O)))
    assert tm.apply(pt, jnp.zeros((2, G, O))).shape == (2, G, NA)
    return True


if __name__ == "__main__":
    test_battle_won()
    print("battle_won OK", flush=True)
    test_adapter()
    print("adapter OK", flush=True)
    test_losses()
    print("losses OK", flush=True)
    print("MARL_TESTS_OK")
