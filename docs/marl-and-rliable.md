# MARL Buffer Alignment and the RLiable Protocol

> Part of the study's documentation — start at the [README](../README.md). Sections 17 and 18. Section numbers are the README's original ones, so every `§x.y` reference in the code, tests and other pages still names the same section.

## 17. Multi-Agent RL Architecture & Sequential Buffer Alignment (`jax_port/marl/`)

The multi-agent module (`jax_port/marl/`) provides cooperative and competitive multi-agent learning architectures implemented in JAX/Flax.

### 17.1. Temporal Causality and Buffer Partitioning
In recurrent multi-agent Q-learning, agents update their parameters through Truncated Backpropagation Through Time (TBPTT) over trajectories of length $L$.
- **Legacy Artifact:** Circular buffers indexed linearly across parallel environments mix observations from disparate environments ($e_0, e_1, \dots, e_{N-1}$), violating temporal causality.
- **`MARLSequentialBuffer`:** Contiguous histories are allocated and stored per parallel environment independently:
  $$\text{Obs Buffer Shape}: (N_{\text{envs}}, C_{\text{env}}, \dots)$$
  Sampling selects an environment index $e \sim \mathcal{U}(0, N_{\text{envs}}-1)$ and a valid temporal window $[t, t+L]$ within that specific environment's ring buffer, guaranteeing that sampled sequences represent genuine, contiguous Markovian trajectories ($s_t^e, a_t^e, r_t^e, s_{t+1}^e$).

### 17.2. Bellman Target Terminal Masking
The recurrent Q-learning Bellman target is given by:
$$y_t = r_t + \gamma \max_{a'} Q(s_{t+1}, a'; \theta^-) \cdot (1 - d_t)$$
In `jax_port/marl/train_ql.py`, the terminal array `done` is converted to a JAX array (`jnp.asarray(bd)`) and passed into `make_ql_seq_update`, strictly preventing bootstrapped value propagation across episode terminations.

## 18. RLiable Evaluation Protocol & Canonical Benchmark Normalization

To eliminate evaluation pathologies (such as outliers skewing the arithmetic mean, or uninformative point estimates lacking confidence intervals), the benchmark suite incorporates the evaluation methodology developed by **Agarwal et al. (NeurIPS 2021)** via `rliable_metrics.py` and `run_rliable_eval.py`. Those two files are a self-contained implementation on top of NumPy — the `rliable` package is **not** a dependency of this repository and is not pinned in `requirements.txt`.

### 18.1. Empirical Random Baselines across Procgen
Rather than assuming artificial minimum bounds, uniform random policies were empirically evaluated across 50 episodes per game on unseen levels (`num_levels=0`, `distribution_mode='easy'`, evaluation seed `1042` — the same protocol used to evaluate trained models). The measurements are produced by `random_baselines.py` (added 23/09/2026; previously the JSON existed with no script able to regenerate it) and can be reproduced with `py -3.10 random_baselines.py --episodes 50`:
- `bossfight`: $0.00 \pm 0.00$
- `starpilot`: $1.78 \pm 1.82$
- `dodgeball`: $0.96 \pm 1.35$
- `maze`: $4.40 \pm 5.01$
- `heist`: $2.80 \pm 4.54$

Per-episode returns and the protocol record are serialized in `results/random_baselines.json`. Standard deviations are sample standard deviations ($ddof=1$).

> ⚠️ **Consequence for `maze` and `heist` (must be read together with section 3.12):** under this protocol a uniform random policy solves `maze` in 22 of 50 episodes and `heist` in 14 of 50, while every trained arm on those two games lands at `maze` $2.76$–$2.80$ and `heist` $0.72$ — i.e. **below the random point estimate**. Taking the standard errors of the anchors ($4.40 \pm 0.71$ S.E. for `maze`, $2.80 \pm 0.64$ S.E. for `heist`, $n=50$), `heist` is clearly worse than random (95% CI $[1.54, 4.06]$ does not reach $0.72$) whereas `maze` is borderline (95% CI $[2.97, 5.83]$ just overlaps $2.80$). Procgen `maze`/`heist` in `easy` mode use small levels with a 600-step budget, so a random walk reaches the goal often enough to score $9$–$10$ more frequently than these policies. Two conclusions follow: (i) the `maze`/`heist` rows of the section 3.6 and 3.12 tables must **not** be read as "ICM beats PPO" — all four arms are within $0.04$ of each other *and* at or below the random anchor; (ii) these two games are excluded from canonical normalization (section 3.14 item 4). The `ppo`/`icm`/`rnd`/`ngu` arms are also where the broken-wrapper defect of section 3.6 was found: the exact `ppo`=`rnd`=`ngu` equality across `maze` and `heist` was not a null result but a swallowed `RuntimeError`, now fixed and covered by `tests/test_exploration_bonuses.py`.

### 18.2. Canonical Normalization Formula
Per Agarwal et al., game returns are normalized against empirical random baselines and maximum task performance:
$$\bar{z}_{m, g} = \frac{R_{m, g} - R_{\text{random}, g}}{R_{\text{expert}, g} - R_{\text{random}, g}}$$
Both Canonical Agarwal normalization and empirical min-max spans are computed and recorded in `results/rliable_scorecard.json`.

### 18.3. Scorecard and Statistical Profiles
The consolidated scorecard across top architectures (`results/rliable_profile.png`) presents three complementary panels:
1. **Aggregated Interquartile Mean (IQM):** Computed across the middle 50% of normalized scores, averaged over the three suite games, with 95% Stratified Bootstrap Confidence Intervals ($B = 10{,}000$ replicates, taken from `rliable_metrics.NUM_BOOTSTRAPS`).
2. **Empirical Performance Profiles with Shaded Bootstrap Bands:** Cumulative distribution functions $\hat{F}(\tau)$ plotted with pointwise 95% bootstrap confidence bands (`fill_between`), showing the fraction of runs exceeding threshold $\tau$ (same $B = 10{,}000$).
3. **Multi-Metric Forest Plot:** Comparative visualization of IQM, 5% Trimmed Mean, Median, and Arithmetic Mean over the top 5 architectures.
4. **Canonical normalization scope:** `results/rliable_scorecard.json` now records `canonical_games` and `canonical_exclusions` with an explicit reason per excluded game, instead of silently substituting a placeholder reference mean.
