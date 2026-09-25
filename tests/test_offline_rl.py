"""Unit tests for compare_offline_rl.py architectures and dataset."""

import numpy as np
import pytest
import torch

from compare_offline_rl import (
    BCPolicy,
    CQLAgent,
    DecisionTransformer,
    IQLAgent,
    NatureCNNEncoder,
    OfflineDataset,
)


def test_nature_cnn_encoder():
    encoder = NatureCNNEncoder(in_channels=3, features_dim=512)
    # Test uint8 input (B, H, W, C)
    x = torch.randint(0, 256, (4, 64, 64, 3), dtype=torch.uint8)
    out = encoder(x)
    assert out.shape == (4, 512)
    assert not torch.isnan(out).any()


def test_bc_policy():
    policy = BCPolicy(in_channels=3, n_actions=15, features_dim=512)
    x = torch.randint(0, 256, (2, 64, 64, 3), dtype=torch.uint8)
    logits = policy(x)
    assert logits.shape == (2, 15)
    action_det = policy.act(x, deterministic=True)
    assert action_det.shape == (2,)
    assert (action_det >= 0).all() and (action_det < 15).all()


def test_iql_agent():
    agent = IQLAgent(in_channels=3, n_actions=15, features_dim=512)
    x = torch.randint(0, 256, (2, 64, 64, 3), dtype=torch.uint8)
    v = agent.value(x)
    assert v.shape == (2,)
    q = agent.q_values(x)
    assert q.shape == (2, 15)
    logits = agent.policy_logits(x)
    assert logits.shape == (2, 15)
    act = agent.act(x, deterministic=True)
    assert act.shape == (2,)


def test_cql_agent():
    agent = CQLAgent(in_channels=3, n_actions=15, features_dim=512)
    x = torch.randint(0, 256, (2, 64, 64, 3), dtype=torch.uint8)
    q = agent.q_values(x)
    assert q.shape == (2, 15)
    act = agent.act(x, deterministic=True)
    assert act.shape == (2,)


def test_decision_transformer():
    dt = DecisionTransformer(in_channels=3, n_actions=15, hidden_dim=64, n_layers=1, n_heads=2, max_len=5)
    B, K = 2, 5
    states = torch.randint(0, 256, (B, K, 64, 64, 3), dtype=torch.uint8)
    actions = torch.randint(0, 15, (B, K), dtype=torch.long)
    rtgs = torch.randn(B, K, 1)
    timesteps = torch.arange(K).unsqueeze(0).repeat(B, 1)

    preds = dt(states, actions, rtgs, timesteps)
    assert preds.shape == (B, K, 15)


def test_offline_dataset():
    N = 100
    obs = np.zeros((N, 64, 64, 3), dtype=np.uint8)
    next_obs = np.zeros((N, 64, 64, 3), dtype=np.uint8)
    act = np.zeros(N, dtype=np.int64)
    rew = np.zeros(N, dtype=np.float32)
    done = np.zeros(N, dtype=np.float32)
    rtg = np.zeros(N, dtype=np.float32)
    timesteps = np.zeros(N, dtype=np.int64)

    ds = OfflineDataset(obs, act, rew, next_obs, done, rtg, timesteps)
    assert len(ds) == N
    item = ds[0]
    assert len(item) == 7
