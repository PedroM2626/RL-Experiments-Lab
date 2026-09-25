"""Intrinsic-reward environment wrappers for the exploration benchmark (README section 3.6).

Moved out of compare_maze_heist.py on 23/09/2026 so they can be imported and tested.
The previous version wrapped the whole bonus computation in `except Exception: pass`,
which meant a shape or dtype problem degraded `icm`/`rnd`/`ngu` into plain PPO without
any error — consistent with the exact three-way tie reported in README section 3.6.
Failures now propagate, and every wrapper counts how many steps it actually added a
bonus so a run can assert that the mechanism was live (`stats()`).
"""

import collections

import gymnasium as gymn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def to_chw_float(obs):
    """Procgen hands CHW through ProcgenGymWrapper; accept HWC too, reject anything else."""
    if not isinstance(obs, np.ndarray) or obs.ndim != 3:
        raise ValueError(f"intrinsic wrapper expected a 3D image observation, got "
                         f"{type(obs).__name__} with ndim={getattr(obs, 'ndim', None)}")
    if obs.shape == (64, 64, 3):
        obs = np.transpose(obs, (2, 0, 1))
    elif obs.shape != (3, 64, 64):
        raise ValueError(f"intrinsic wrapper expected (3,64,64) or (64,64,3), got {obs.shape}")
    return torch.from_numpy(np.ascontiguousarray(obs)).unsqueeze(0).float() / 255.0


class RunningMeanStd:
    """Welford-like running mean and variance estimator for bonus normalization (Burda et al., 2018)."""

    def __init__(self, epsilon: float = 1e-4, shape=()):
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if len(x.shape) > 0 else 1
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        self.mean = new_mean
        self.var = m2 / tot_count
        self.count = tot_count


class _IntrinsicWrapper(gymn.Wrapper):
    """Bookkeeping and optional running variance normalization shared by all bonus wrappers."""

    def __init__(self, env, beta=0.01, normalize=False, clip=5.0):
        super().__init__(env)
        self.beta = beta
        self.normalize = normalize
        self.clip = clip
        self.rms = RunningMeanStd() if normalize else None
        self.n_steps = 0
        self.n_bonus = 0
        self.intrinsic_sum = 0.0

    def _bonus(self, obs):
        raise NotImplementedError

    def _scale_bonus(self, intrinsic: float) -> float:
        if self.normalize:
            self.rms.update(np.array([intrinsic]))
            std = float(np.sqrt(max(1e-8, float(self.rms.var))))
            return float(np.clip(intrinsic / std, 0.0, self.clip))
        return intrinsic

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.n_steps += 1
        intrinsic = float(self._bonus(obs))
        self.n_bonus += 1
        self.intrinsic_sum += intrinsic
        scaled_intrinsic = self._scale_bonus(intrinsic)
        return obs, reward + self.beta * scaled_intrinsic, terminated, truncated, info

    def stats(self):
        return {
            "wrapper": type(self).__name__,
            "steps": self.n_steps,
            "bonus_applied": self.n_bonus,
            "bonus_ratio": self.n_bonus / max(1, self.n_steps),
            "intrinsic_mean": self.intrinsic_sum / max(1, self.n_bonus),
            "normalized": self.normalize,
        }


class ICMWrapper(_IntrinsicWrapper):
    """ curiosity via forward-model prediction error in a learned feature space.

    The inverse model is constructed and registered in the optimizer to mirror the
    reference implementation this benchmark replicates, but the loss contains only the
    forward term — it therefore receives no gradient. See jax_port/exploration.py, which
    documents the same quirk.
    """

    def __init__(self, env, beta=0.01, n_actions=15, **kwargs):
        super().__init__(env, beta, **kwargs)
        self.n_actions = n_actions
        self.phi = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(), nn.Flatten())
        with torch.no_grad():
            n_flat = self.phi(torch.zeros(1, 3, 64, 64)).shape[1]
        self.forward_model = nn.Sequential(nn.Linear(n_flat + n_actions, 512), nn.ReLU(), nn.Linear(512, n_flat))
        self.inverse_model = nn.Sequential(nn.Linear(n_flat * 2, 512), nn.ReLU(), nn.Linear(512, n_actions))
        self.opt = torch.optim.Adam(
            list(self.phi.parameters()) + list(self.forward_model.parameters()) + list(self.inverse_model.parameters()),
            lr=1e-4)
        self.prev_phi = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        with torch.no_grad():
            self.prev_phi = self.phi(to_chw_float(obs))
        return obs, info

    def step(self, action):
        if self.prev_phi is None:
            raise RuntimeError("ICMWrapper.step() called before reset() — prev_phi is undefined")
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.n_steps += 1
        x = to_chw_float(obs)
        phi_next = self.phi(x)
        a_onehot = F.one_hot(torch.tensor([int(action)]), num_classes=self.n_actions).float()
        pred_phi = self.forward_model(torch.cat([self.prev_phi, a_onehot], dim=1))
        loss_forward = F.mse_loss(pred_phi, phi_next)
        intrinsic = float(loss_forward.item())
        self.opt.zero_grad()
        loss_forward.backward()
        self.opt.step()
        self.prev_phi = phi_next.detach()
        self.n_bonus += 1
        self.intrinsic_sum += intrinsic
        scaled_intrinsic = self._scale_bonus(intrinsic)
        return obs, reward + self.beta * scaled_intrinsic, terminated, truncated, info

    def _bonus(self, obs):  # pragma: no cover - ICM overrides step for its state machine
        raise NotImplementedError


class RNDWrapper(_IntrinsicWrapper):
    """Random Network Distillation: prediction error against a fixed random target network.

    The projection after the convolutional stem is sized by probing the stem, as ICMWrapper
    does. compare_maze_heist.py hardcoded nn.Linear(1024, 512) over a two-convolution stem
    that flattens to 2304, so every RND/NGU step raised RuntimeError and was swallowed by
    the wrapper's `except Exception: pass` — those two arms were plain PPO runs. Fixing the
    geometry changes what the arm is, so README section 3.6 must be re-measured.
    """

    def __init__(self, env, beta=0.01, hidden_dim=512, **kwargs):
        super().__init__(env, beta, **kwargs)
        self.hidden_dim = hidden_dim
        self.target = self._build_net()
        self.predictor = self._build_net()
        for p in self.target.parameters():
            p.requires_grad = False
        self.opt = torch.optim.Adam(self.predictor.parameters(), lr=1e-4)

    def _build_net(self):
        stem = nn.Sequential(nn.Conv2d(3, 32, 8, stride=4), nn.ReLU(),
                             nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(), nn.Flatten())
        with torch.no_grad():
            n_flat = stem(torch.zeros(1, 3, 64, 64)).shape[1]
        stem.add_module("proj", nn.Linear(n_flat, self.hidden_dim))
        return stem

    def _bonus(self, obs):
        x = to_chw_float(obs)
        with torch.no_grad():
            t = self.target(x)
        p = self.predictor(x)
        loss = F.mse_loss(p, t)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss.item())


class NGUWrapper(RNDWrapper):
    """RND novelty scaled by an episodic-memory term (nearest-neighbour distance in phi)."""

    def __init__(self, env, beta=0.01, memory_size=1000, knn=100, sample=5, **kwargs):
        super().__init__(env, beta, **kwargs)
        self.memory = collections.deque(maxlen=memory_size)
        self.knn = knn
        self.sample = sample

    def _bonus(self, obs):
        x = to_chw_float(obs)
        with torch.no_grad():
            t = self.target(x)
        p = self.predictor(x)
        rnd_loss = float(F.mse_loss(p, t).item())
        phi = p.detach().cpu().numpy().flatten()
        if len(self.memory) > 10:
            dists = [np.linalg.norm(phi - m) for m in list(self.memory)[-self.knn:]]
            episodic = float(np.mean(sorted(dists)[:self.sample]))
        else:
            episodic = 1.0
        self.memory.append(phi)
        loss = F.mse_loss(p, t)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return rnd_loss * episodic
