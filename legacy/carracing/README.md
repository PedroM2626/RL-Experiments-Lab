# Legacy Continuous Control Prototypes: CarRacing-v2 SAC

This directory contains the initial Phase 1 exploratory codebase developed prior to the project's pivot to discrete-action procedural benchmarks (Procgen):

- `sac_trainer.py`: Custom Soft Actor-Critic (SAC) implementation with continuous Box action space, Twin Q-networks, target entropy auto-tuning, and experience replay buffer.
- `compare_architectures.py`: Training harness benchmarking Classic 3-layer CNN vs. Dual Attention (CBAM) CNN on `CarRacing-v2` / `CarRacing-v3`.

### Motivation for Archiving

1. **Domain Transition**: The primary research focus of this repository is high-throughput procedural generalization using Procgen (discretized action space, 15 actions, vectorized rollouts via CleanRL and JAX/PureJaxRL).
2. **Action Space Incompatibility**: Procgen uses `spaces.Discrete(15)`, whereas CarRacing requires `spaces.Box(low=-1.0, high=1.0, shape=(3,))`. Isolating continuous SAC ensures the root workspace strictly reflects the active Procgen benchmark suite.
3. **Reproducibility**: The code is preserved here for historical transparency and reproducibility of the early continuous vision-RL experiments.
