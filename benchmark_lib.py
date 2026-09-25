"""Unified benchmark infrastructure library for Procgen evaluations.

Provides standardized environment factories and the canonical 3-tier evaluation protocol
(unseen stochastic, unseen deterministic, and train stochastic) used across all benchmarks.
Ensures zero protocol drift across evaluation scripts.
"""
import os
from typing import Dict, Tuple, Optional, Any
import numpy as np
import torch
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from procgen_wrapper import make_procgen_env


def make_eval_env(
    game: str,
    num_levels: int,
    seed: int,
    vector: bool = False,
    distribution_mode: str = "easy",
) -> DummyVecEnv:
    """Create a standardized single-monitor DummyVecEnv for policy evaluation."""
    return DummyVecEnv([
        lambda: Monitor(
            make_procgen_env(
                game,
                num_levels=num_levels,
                distribution_mode=distribution_mode,
                seed=seed,
                vector=vector,
            )
        )
    ])


def evaluate_model_protocol(
    model: BaseAlgorithm,
    game: str,
    seed: int,
    n_stoch: int = 100,
    n_det: int = 100,
    n_train: int = 15,
) -> Dict[str, float]:
    """Execute the canonical evaluation protocol:
    
    1. n_stoch episodes on unseen levels (num_levels=0, seed+1000, deterministic=False)
    2. n_det episodes on unseen levels (num_levels=0, seed+1000, deterministic=True)
    3. n_train episodes on training levels (num_levels=200, seed, deterministic=False)
    
    Returns:
        dict with keys: 'stoch_unseen', 'det_unseen', 'stoch_train', 'gen_gap'
    """
    vector = len(model.observation_space.shape) == 1

    unseen_env = make_eval_env(game, num_levels=0, seed=seed + 1000, vector=vector)
    train_env = make_eval_env(game, num_levels=200, seed=seed, vector=vector)

    try:
        m_st, _ = evaluate_policy(model, unseen_env, n_eval_episodes=n_stoch, deterministic=False)
        m_dt, _ = evaluate_policy(model, unseen_env, n_eval_episodes=n_det, deterministic=True)
        m_tr, _ = evaluate_policy(model, train_env, n_eval_episodes=n_train, deterministic=False)
    finally:
        unseen_env.close()
        train_env.close()

    m_st, m_dt, m_tr = float(m_st), float(m_dt), float(m_tr)
    return {
        "stoch_unseen": round(m_st, 3),
        "det_unseen": round(m_dt, 3),
        "stoch_train": round(m_tr, 3),
        "gen_gap": round(m_tr - m_st, 3),
    }
