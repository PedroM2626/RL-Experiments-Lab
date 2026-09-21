"""Frame stacking over ProcgenGym3Env (faithful to study's frame_stack).

StackVec: stores the last K frames per env (numpy ring buffer), reset clears.
API compatible with training loop (act/observe gym3-style):
  reset() -> stacked; act(a) buffers; observe() -> (rew, {"rgb_stacked"},
  first). K=1 equals no stack (study baseline).
"""

import numpy as np


class StackVec:
    def __init__(self, game, num_envs, num_levels=200, seed=42, k=4,
                 distribution="easy"):
        from procgen import ProcgenGym3Env
        self.env = ProcgenGym3Env(num=num_envs, env_name=game,
                                  num_levels=num_levels,
                                  distribution_mode=distribution,
                                  rand_seed=seed)
        self.n, self.k = num_envs, k
        _, d, _ = self.env.observe()
        o = d["rgb"] if isinstance(d, dict) else d
        self.hist = np.zeros((num_envs, k, 64, 64, 3), np.uint8)
        self.hist[:, -1] = o
        self._pending = np.zeros(num_envs, np.int32)

    def _stacked(self):
        n, k = self.n, self.k
        return self.hist.transpose(0, 2, 3, 4, 1).reshape(n, 64, 64, 3 * k)

    def reset(self):
        _, d, _ = self.env.observe()
        o = d["rgb"] if isinstance(d, dict) else d
        self.hist[:] = 0
        self.hist[:, -1] = o
        return self._stacked()

    def act(self, a):
        self._pending = np.asarray(a, np.int32)

    def observe(self):
        self.env.act(self._pending)
        rew_d, d, first_d = self.env.observe()
        o = d["rgb"] if isinstance(d, dict) else d
        first = np.asarray(first_d)
        self.hist = np.concatenate([self.hist[:, 1:], o[:, None]], axis=1)
        self.hist[first, :-1] = 0
        return (np.asarray(rew_d, np.float32), {"rgb_stacked": self._stacked()},
                first)

    def step(self, act):
        self.act(act)
        return self.observe()
