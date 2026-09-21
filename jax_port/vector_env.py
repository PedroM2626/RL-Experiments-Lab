"""CPU-env pipeline of the JAX port (PA1).

Boundary faithful to SB3 study: same levels/seeds/preprocessing of
``procgen_wrapper.py`` (frame_stack=1, HWC 64x64x3 uint8 -> CHW 3x64x64),
without torch/sb3/cv2/gymnasium dependencies — only gym+procgen+numpy
on the CPU side. The JAX (GPU) side receives the batch via ``jnp.asarray`` + JIT.

Fidelity reference:
  study: ``procgen_wrapper.py:29`` (reset HWC->CHW), ``:41`` (step),
  ``:91`` (factory num_levels/distribution_mode/rand_seed).
"""

import gym
import numpy as np

import procgen  # noqa: F401  (registers procgen:procgen-*-v0)


def make_single_env(game="coinrun", num_levels=200, distribution_mode="easy", rand_seed=0):
    """Raw ProcGen env (legacy gym API, HWC uint8 obs)."""
    return gym.make(
        f"procgen:procgen-{game}-v0",
        num_levels=num_levels,
        start_level=0,
        distribution_mode=distribution_mode,
        rand_seed=rand_seed,
    )


class ProcgenVectorEnv:
    """N synchronous ProcGen envs on CPU, stacked CHW uint8 obs.

    Semantics identical to N x ``ProcgenGymWrapper(frame_stack=1)`` from study:
    reset returns ``(N,3,64,64)`` uint8; step receives ``(N,)`` int and returns
    ``obs (N,3,64,64) uint8, rew (N,) float32, done (N,) bool``.
    SB3-style autoreset on done (bench measures throughput, not episodes).
    """

    def __init__(self, game="coinrun", num_envs=4, num_levels=200,
                 distribution_mode="easy", seed=42):
        self.envs = [
            make_single_env(game, num_levels, distribution_mode, seed + i)
            for i in range(num_envs)
        ]
        self.num_envs = num_envs
        self.action_space_n = self.envs[0].action_space.n

    def reset(self):
        batch = [np.transpose(e.reset(), (2, 0, 1)) for e in self.envs]
        return np.stack(batch).astype(np.uint8)

    def step(self, actions):
        obs, rew, done = [], [], []
        for e, a in zip(self.envs, actions):
            o, r, d, _ = e.step(int(a))
            if d:
                o = e.reset()
            obs.append(np.transpose(o, (2, 0, 1)))
            rew.append(r)
            done.append(d)
        return (np.stack(obs).astype(np.uint8),
                np.asarray(rew, dtype=np.float32),
                np.asarray(done, dtype=bool))

    def sample_actions(self, rng):
        return rng.integers(0, self.action_space_n, size=self.num_envs)
