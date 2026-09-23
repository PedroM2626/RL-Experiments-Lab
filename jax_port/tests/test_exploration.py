"""Bookkeeping tests for jax_port/exploration.py (ICM/RND/NGU).

The SB3 side of this project published an exploration benchmark where two arms were really
PPO runs: the wrapper raised on every step, the error was swallowed, and nothing recorded
that no bonus had been added. The port has the same failure modes available and had no way to
detect either of them. These cases pin that every arm (a) injects a bonus, (b) reports what it
injected, and (c) reports zero when beta is zero — which is what makes "null by scale" a
measurable statement instead of an assumption.
"""

import numpy as np


def _rand(n, key):
    rng = np.random.RandomState(key)
    return rng.randint(0, 256, size=(n, 64, 64, 3)).astype(np.uint8)


def run_all():
    from jax_port.exploration import Exploration

    N = 4
    for kind in ("icm", "rnd", "ngu"):
        obs, obs2 = _rand(N, 1), _rand(N, 2)
        act = np.zeros(N, np.int32)
        rew = np.ones(N, np.float32)
        done = np.zeros(N, bool)

        exp = Exploration(kind, N, seed=42)
        exp.reset(obs)
        aug = exp.step(obs, act, rew, obs2, done)
        st = exp.stats()

        assert st["steps"] == 1 and st["bonus_cells"] == N, (kind, st)
        assert np.isfinite(st["bonus_mean"]) and st["bonus_mean"] > 0.0, (kind, st)
        assert st["bonus_max"] >= st["bonus_mean"] > 0.0, (kind, st)
        # the reward actually handed to PPO must match what stats() claims was injected
        assert abs(float(np.mean(np.asarray(aug) - rew)) - st["effective_bonus"]) < 1e-6, \
            (kind, st, aug)
        assert np.all(np.asarray(aug) >= rew - 1e-6), (kind, aug)

        # beta = 0 is the control arm: identical rewards, and the bookkeeping says so
        exp0 = Exploration(kind, N, seed=42, beta=0.0)
        exp0.reset(obs)
        aug0 = exp0.step(obs, act, rew, obs2, done)
        assert np.allclose(aug0, rew), (kind, aug0)
        assert exp0.stats()["effective_bonus"] == 0.0, exp0.stats()
        assert exp0.stats()["bonus_mean"] > 0.0, (
            f"{kind}: the mechanism itself stopped computing a bonus")

        # multiple steps accumulate rather than overwrite
        for _ in range(3):
            exp.step(obs, act, rew, obs2, done)
        assert exp.stats()["steps"] == 4, exp.stats()

    return "EXPLORATION_TESTS_OK"


if __name__ == "__main__":
    print(run_all())
