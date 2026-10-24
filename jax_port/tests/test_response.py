"""Response-measure tests for the JAX port (jax_port.response).

The TV helper runs on numpy alone, so it executes in the CI port job (which has
jax but not procgen); the full measure needs the port venv's procgen and SKIPs
without it -- the same contract as every other registered case.
"""
import numpy as np


def test_tv_max_pairwise():
    from jax_port.response import tv_max_pairwise
    assert tv_max_pairwise([[0.5, 0.5], [0.5, 0.5]]) == 0.0
    assert abs(tv_max_pairwise([[1.0, 0.0], [0.0, 1.0]]) - 1.0) < 1e-12
    # three rows: the max pair is the two that swap -- TV 0.8
    p = np.array([[0.1, 0.9], [0.9, 0.1], [0.2, 0.8]])
    assert abs(tv_max_pairwise(p) - 0.8) < 1e-12
    # one row: no pairs, no variation
    assert tv_max_pairwise([[0.3, 0.7]]) == 0.0
    # for normalized rows TV is bounded by 1 (unnormalized rows are not a
    # distribution and the helper does not pretend otherwise)
    p = np.random.rand(5, 15)
    p /= p.sum(axis=1, keepdims=True)
    assert 0.0 <= tv_max_pairwise(p) <= 1.0


def test_cross_game_response_keys():
    try:
        import procgen  # noqa: F401
    except ImportError as e:
        raise ImportError(f"procgen missing: {e}")
    import jax
    import jax.numpy as jnp
    import optax
    from jax_port.backbones import BACKBONES
    from jax_port.networks import ActorCritic
    from jax_port.ppo import make_optimizer, make_update_fn
    from jax_port.response import cross_game_response

    model = ActorCritic(backbone=BACKBONES["classic"](), n_actions=15)
    key = jax.random.PRNGKey(0)
    params = model.init(key, jnp.zeros((1, 64, 64, 3), jnp.float32), None)
    opt = make_optimizer(lr=3e-4, kind="adam")
    _, _, forward_fn, _ = make_update_fn(model, opt, clip_range=0.2, ent_coef=0.01)
    state = (params, opt.init(params))
    row = cross_game_response(state, forward_fn, "pixels",
                              games=("maze", "coinrun"))
    assert set(row) >= {"games", "observation", "max_pairwise_tv",
                        "distinct_argmax_across_games", "v_std_across_games",
                        "responsive"}
    assert row["games"] == ["maze", "coinrun"]
    assert 0.0 <= row["max_pairwise_tv"] <= 1.0
    assert row["distinct_argmax_across_games"] in (1, 2)
    assert isinstance(row["responsive"], bool)


def run_all():
    test_tv_max_pairwise()
    test_cross_game_response_keys()
    return {"tv_max_pairwise": True, "cross_game_response_keys": True}
