"""Vectorized SMAX adapter (JaxMARL) — device-resident, zero round-trip.

N envs in vmap (JIT-compiled reset+step); masked autoreset per env on
done['__all__']; obs (N,A,O) and world_state (N,S) assembled on device;
only dones/rewards travel to host (small). Win-rate derived from
unit_alive on done (enemy indices [A,) all dead AND ally alive;
draw = defeat, SMAC standard).
"""

import jax
import jax.numpy as jnp
import numpy as np


def battle_won(unit_alive, n_allies):
    """Pure numpy (testable without env). Allies = first n_allies."""
    alive = np.asarray(unit_alive, bool)
    return bool((not alive[n_allies:].any()) and alive[:n_allies].any())


class SmaxVec:
    def __init__(self, map_name="3m", num_envs=32, seed=0, **kw):
        from jaxmarl import make
        from jaxmarl.environments.smax import map_name_to_scenario
        scenario = map_name_to_scenario(map_name)
        self.env = make("HeuristicEnemySMAX", scenario=scenario, **kw)
        self.agents = self.env.agents
        self.n_agents = len(self.agents)
        self.n_actions = self.env.action_space(self.agents[0]).n
        self.N = num_envs
        self.vreset = jax.jit(jax.vmap(self.env.reset))
        self.vstep = jax.jit(jax.vmap(self.env.step))
        key = jax.random.PRNGKey(seed)
        obs, state = self.vreset(jax.random.split(key, num_envs))
        self.obs, self.state = obs, state
        self.obs_dim = int(obs[self.agents[0]].shape[-1])
        self.state_dim = int(obs["world_state"].shape[-1])
        self.key = key

    def obs_batch(self):
        return jnp.stack([self.obs[a] for a in self.agents], axis=1)

    def state_batch(self):
        return self.obs["world_state"]

    def _masked_replace(self, old, new, mask):
        def sel(a, b):
            if isinstance(a, (jnp.ndarray, np.ndarray)) and \
                    np.shape(a) and np.shape(a)[0] == self.N:
                m = np.asarray(mask).reshape(
                    (self.N,) + (1,) * (np.ndim(a) - 1))
                return jnp.where(m, jnp.asarray(b), jnp.asarray(a))
            return a
        if isinstance(old, dict):
            return {k: sel(old[k], new[k]) for k in old}
        return jax.tree.map(sel, old, new)

    def reset_where(self, done):
        self.key, kr = jax.random.split(self.key)
        o2, s2 = self.vreset(jax.random.split(kr, self.N))
        self.obs = self._masked_replace(self.obs, o2, done)
        self.state = self._masked_replace(self.state, s2, done)

    def step(self, acts):
        """acts: (N,A) int device/host. Returns (obs, rew, done, win);
        obs stays on device; rew/done/win in numpy (small)."""
        self.key, ks = jax.random.split(self.key)
        dacts = {a: jnp.asarray(acts[:, i]) for i, a in enumerate(self.agents)}
        obs, state, rew, done, _ = self.vstep(
            jax.random.split(ks, self.N), self.state, dacts)
        self.obs, self.state = obs, state
        done_all = np.asarray(done["__all__"])
        r0 = np.asarray(rew[self.agents[0]], np.float32)
        win = np.zeros(self.N, bool)
        if done_all.any():
            st = state.state if hasattr(state, "state") else state
            alive = np.asarray(st.unit_alive)
            for i in np.where(done_all)[0]:
                win[i] = battle_won(alive[i], self.n_agents)
            self.reset_where(done_all)
        return self.obs_batch(), r0, done_all, win
