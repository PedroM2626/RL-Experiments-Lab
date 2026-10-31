# The JAX Port of ProcGen

> Part of the study's documentation — start at the [README](../README.md). Sections 13–16: scope note, trajectory, status, parity, the full grid, MARL and temporal results, process references. Section numbers are the README's original ones, so every `§x.y` reference in the code, tests and other pages still names the same section.

## 13. Scope and Restoration Note (04/09/2026) — Why this README was Restored

### 13.1. Background Context

This repository transitioned through three distinct phases, all preserved within Git commit history for auditing:

1. **ProcGen/SB3 Phase (this study, commit `4f84ed3`).** The original systematic benchmark on real `ProcGen` using `Stable-Baselines3`/`PyTorch` — sections 1–12 above. This constitutes the scientific core of the project.
2. **JAX/Craftax/Brax/MARL Phase (commits `5e88c16`–`6d450fa`).** An exploration rewriting environments in `JAX`/`Flax`/`Optax` featuring `Craftax`, `Brax`, `MPE`, offline boxing, and combinatorial grids. Protocols, budgets, metrics, and dynamics differed substantially from the ProcGen study (e.g., symbolic Craftax episodic return vs pixel ProcGen reward; `8M` steps vs `100k`; throughput measured on non-comparable workloads).
3. **JAX-Port-of-ProcGen Phase (Sep/2026).** A systematic initiative to reconstruct the **exact** study in `JAX` to reproduce the identical experimental design at substantially higher throughput (detailed in section 14).

On `04/09/2026`, by explicit design decision, **all phase 2–3 exploration scaffolding outside the study scope was removed from the working tree**, and **the repository was restored to the exact state of commit `4f84ed3`**. Sections 1–12 above are, therefore, **the byte-for-byte documentation of the original ProcGen study** — no numbers, tables, or conclusions were altered in this restoration.

### 13.1.1. Methodological Rationale for Returning to ProcGen

Through empirical profiling, training wall-clock time in Craftax was found to be **virtually identical to ProcGen** (runs requiring millions of steps, multi-hour grids), yet with **significantly reduced academic depth** — lacking the published generalization protocol (`200` train vs `0` unseen), lacking the 16 architectures × 5 games × 5 seeds grid, and without CI/Cohen's d/AUC/dual-eval/budget-scaling rigor. Computational resources were being expended for a fraction of the scientific rigor. Consequently, the optimal methodological path was **returning to ProcGen and accelerating it** — maintaining the rigorous study while resolving its sole bottleneck (throughput) via a faithful JAX port, rather than compromising on experimental depth. Sections 14–16 document this work.

### 13.2. Academic Justification for Scaffolding Removal

Removal is framed not as lost effort, but as an essential measure of internal validity:

- **Construct Validity.** Aggregating two benchmarks with divergent reward semantics, episode horizons, and budgets (`Craftax 8M` vs `ProcGen 100k`) within a single `README`/`results/` invites invalid comparisons between incommensurable quantities. Generalization literature in RL (Cobbe et al., ProcGen) mandates a fixed protocol per study.
- **Reproducibility.** The ProcGen study relies strictly on `Python ≤3.10` + `procgen 0.10.7` + `gym 0.26.2` + `torch cu121` (section 5 and `requirements.txt`), whereas the exploratory JAX suite required `Python 3.12` + `jax 0.11` + `craftax`/`brax`. Maintaining both within the same virtual environment breaks dependency resolution on both sides (the `numpy<2` vs `numpy 2.x` conflict in section 14.3 is a direct example).
- **Traceability.** All excised code remains fully retrievable via Git history (`git log --oneline`, `git show 5e88c16:...`, `git show 6d450fa:...`). History was not rewritten; the working tree was simply aligned with the study's scope.

### 13.3. Inventory of Restored and Archived Components

**Archived from the working tree (preserved in Git history):** `src/` (JAX trainers: `ppo.py`, `dqn.py`, `marl_*`, `continuous_rl.py`, `offline_rl.py`, `recurrent_ppo.py`, `eval_utils.py`, `procgen_parity_modules.py`, `procgen_env.py`, `procgen_ppo.py`), `experiments/` (12 JAX/Craftax/Brax/MARL benchmarks), `figures/` (9 JAX figures), `results/` JAX artifacts (`*_benchmarks_results.json`, `boxing_*`, `dataset_boxing_expert.npz`, `procgen_parity_master_results.json`, `combinatorial_grid_results.json`, `results/logs/`, `boxing_final_results.txt`), standalone scripts (`run_all_procgen_combinations.py`, `run_combinatorial_grid.py`, `run_convergence_*.sh`, `run_full_benchmark.py`, `run_smoke_test.py`, `bench_*.py`, `smoke_procgen.py`, `diag_env.sh`, `probe_procgen.sh`, `setup_procgen_env.sh`, `_verify_pg.py`, `_probe_pg3.py`), and build caches (`.jax_compile_cache/`, `__pycache__/`).

**Restored to state `4f84ed3`:** All root `compare_*.py` scripts, `models/` (`sb3_extractors.py`, `cnn_attention.py`, `cnn_classic.py`, `combined_extractors.py`, `world_model_extractors.py`), `procgen_wrapper.py`, `*_analysis.py`, `re_eval_*.py`, `visualize_*.py`, pinned `requirements.txt` (ProcGen), original `.gitignore`, and original `results/` (JSONs + PNGs for sections 3–12).

> **Accuracy note (23/09/2026):** "preserved in Git history" holds for the bulk of the inventory — `git log --all --diff-filter=A --name-only` confirms 29 `src/` modules, 20+ `experiments/` benchmarks and the JAX `results/*.json` artifacts are retrievable (e.g. `git show 5e88c16:src/ppo.py`). It does **not** hold for every name in the list above: the throwaway helper scripts `setup_procgen_env.sh`, `probe_procgen.sh`, `diag_env.sh`, `smoke_procgen.py`, `_verify_pg.py` and `_probe_pg3.py`, plus `boxing_final_results.txt` and `combinatorial_grid_results.json`, were never committed and are gone. They were scratch tooling and one duplicate of `combinatorial_matrix_results.json`, so nothing in sections 1–19 depends on them.

### 13.4. Purpose of Sections 14–16

Sections 1–12 remain frozen as the scientific record of the completed study. Sections 14–16 serve as the **methodological research log**: documenting, with identical academic rigor, **the trajectory, decisions, architectural choices, and adaptations** encountered while porting this study to JAX — including verified milestones (Gates PA0 and PA1 cleared, §14.3 and §15.1), discarded paths, and open objectives. No metrics in sections 1–12 are altered here; this constitutes meta-documentation of process, not new study outcomes.

## 14. Trajectory of the JAX-Port-of-ProcGen — Objective, Decisions, and Gate PA0

### 14.1. Core Objective and Selected Route (Path A)

The objective was never "accumulate disparate benchmarks in JAX", but rather **"the ProcGen study, executed faster"**. Among possible alternatives, **Path A — faithful porting of the ProcGen study to JAX** — was explicitly chosen:

> Reconstruct the **exact** study in JAX: Authentic ProcGen (`envpool`/CPU) + **16 architectures × 5 games × 5 seeds × 100k steps** + **95% CI / Cohen's d / AUC / dual-evaluation (stoch+det) / budget-scaling**. Expected speedup: ~**10×** over SB3 throughput (~`300 FPS` legacy on `cuda`), preserving identical experimental design.

Rejected alternatives included: (B) substituting ProcGen with JAX/Craftax — rejected because symbolic/pixel Craftax and pixel ProcGen possess divergent observation spaces, transition dynamics, and generalization regimes, meaning "PPO ≫ DQN on Craftax" fails to answer "which visual extractor generalizes best on ProcGen"; and (C) reimplementing ProcGen natively in pure JAX — rejected as intractable (ProcGen is C++/OpenGL with proprietary procedural generation; rewriting it introduces a different environment rather than accelerating the existing benchmark).

Path A necessitated a hybrid architecture: **CPU environment pipeline + JAX GPU learner** — vectorized ProcGen environments hosted on CPU feeding a batched PPO learner implemented in JAX on the GPU.

### 14.2. Phased Roadmap and Resource Constraints

Prior to implementing learner logic, a phased plan with strict viability gates was established:

- **PA0 (Compatibility):** Prove that `procgen 0.10.7` and `jax[cuda12]` coexist and operate (env simulation + GPU acceleration) within the same interpreter environment. Exit criterion: `PROCGEN_JAX_OK` — initialize headless `coinrun` and execute a JAX kernel on `cuda:0` within the same virtual environment.
- **PA1 (Throughput):** Construct the vectorized CPU→GPU pipeline and measure empirical FPS against the ~`300 FPS` SB3 legacy baseline.
- **PA2+ (Fidelity):** Implement the 16 extractors and the full evaluation suite (CI/Cohen/AUC/dual-eval/budget-scaling), validating numerical parity on a subset before launching the full grid.

As a prerequisite for PA0, running background workloads were safely terminated (caching intermediate JSONs) to release GPU VRAM, and the host environment was profiled with `diag_env.sh`. Maintaining a single concurrent job on the 8 GB GPU prevents VRAM out-of-memory errors.

### 14.3. Gate PA0 — Diagnostics, Roadblocks, Probing, and Resolution

**Initial Diagnostic (Informative Failure).** The active environment had `Python 3.12` + `JAX 0.11.1` + operational `cuda:0`, but `pip` could not locate **any** compatible wheel for `procgen`: `procgen 0.10.7` distributes wheels up to `cp310`. The host lacked `Python 3.10` or conda toolchains, and ProcGen requires compiled C++ extensions. Compiling from source without `cmake` was error-prone. Resolving environment compatibility was a non-negotiable prerequisite.

**Feasibility Probing (`probe_procgen.sh`).** Two empirical questions were addressed: (1) Does a valid `manylinux` wheel for `procgen 0.10.7` exist for `cp310`? Yes — `procgen-0.10.7-cp310-cp310-manylinux...whl` installs cleanly without local compilation. (2) Can `Python 3.10` be acquired on `Ubuntu 24.04` via reputable packaging channels? Yes, via the `deadsnakes` PPA. Verdict: **VIABLE**.

**Environment Setup (`setup_procgen_env.sh`).** Executed pipeline: `Python 3.10` (deadsnakes) + dedicated virtual environment + `procgen` (wheel) + `jax[cuda12]`. Interpreter and venv configuration completed rapidly; JAX CUDA wheel installation was the primary I/O bottleneck. Initial validation revealed two specific issues, resolved and documented below:

| # | Diagnostic Symptom | Root Cause | Implemented Resolution |
|---|---|---|---|
| 1 | `numpy` ABI conflict: `procgen` requires `numpy<2.0`, but default JAX installation pulled `numpy 2.2.6` | Divergent dependency bounds (legacy C++ extension vs modern JAX ecosystem) | Pinned `numpy==1.26.4` in the port venv; residual `ml-dtypes` warning is harmless (GPU test kernel executed successfully) |
| 2 | Initial test script targeted `gymnasium`, but ProcGen registers within `gym 0.26.2` (legacy 4-tuple `step` API) | Original study bridges this via wrapper (`procgen_wrapper.py`); probe script attempted direct lookup in the wrong namespace | Updated verification script (`_verify_pg.py`) to legacy `gym` API; explored vectorized C++ interface in `_probe_pg3.py` (`gym3`) |

**Validation (Gate Cleared, Re-verified on 04/09/2026).** Following adjustments, `PROCGEN_JAX_OK` was achieved in headless WSL2 under `py3.10`: `coinrun` initializes with observation shape `(64,64,3)` `uint8` and `Discrete(15)` action space, and JAX executes on `cuda:0` (verified: `jax 0.6.2`, `devices=[CudaDevice(id=0)]`, matmul reduction produces `1073741824.0`). Verified stack: `JAX 0.6.2` + `flax 0.10.7` + `optax 0.2.8` + `numpy 1.26.4` + `procgen 0.10.7` (with `gym 0.26.2`, `gymnasium 1.3.0`, `gym3 0.3.3`).

### 14.4. Summary of Design Decisions and Technical Adjustments

| Decision / Adjustment | Alternative Considered | Criteria and Outcome |
|---|---|---|
| Path A (faithful port) vs B (Craftax substitute) vs C (native JAX rewrite) | Path B was computationally cheaper; C was "pure JAX" | Experimental fidelity took priority: only Path A addresses the research question without altering environment dynamics |
| Terminate Craftax grid prior to PA0 | Run setup concurrently | 8 GB VRAM cannot safely sustain concurrent GPU training and compilation; clean termination with serialized JSON |
| `deadsnakes` + dedicated `py3.10` venv | Compile ProcGen under `py3.12`, or conda/pyenv | `cp310` wheel is pre-built; dedicated venv isolates the `numpy` version conflict from the SB3 study environment |
| Pin `numpy 1.26.4` | Force `numpy 2.x` | Strict requirement of ProcGen C++ extension (`<2`); JAX operates nominally on `1.26` |
| Validate via `gym 0.26.2` (and probe `gym3`) rather than direct `gymnasium` | Uniformly enforce `gymnasium` | ProcGen environment registry is natively `gym`; the port preserves the established wrapping boundary |
| Archive exploratory JAX scaffolding from working tree (section 13) | Keep setup scripts alongside main study | Cleanliness of repository and dependencies: SB3 study remains reproducible with a single `requirements.txt`; scaffolding is preserved in Git history |

## 15. Current Status, Boundaries, and Full Grid Completion (PA2-Speed Complete; Parity Complete)

**Current Status.** The primary working tree remains the fully reproducible ProcGen/SB3 study (sections 1–12; reproduction commands in section 5). The JAX port, co-located on `main` under `jax_port/` (without modifying root study files), spans the full experimental design: 13 backbones, PPO/A2C, DQN/QR-DQN, ICM/RND/NGU, data augmentations, 4-arm HRL, stoch+det+gap evaluation, statistical metrics (CI/Cohen/AUC), and the full 615-cell multi-seed parity grid — completely executed and serialized in `jax_port/analysis_full.json` (§15.4.2).

> **Evidence note (23/09/2026):** the raw per-cell grid outputs are git-ignored, so `jax_port/cells_summary.json` (897 cells = 865 evaluated + 32 MARL) is the committed record they were reduced to. `python -m jax_port.analyze_grade --from_summary jax_port/cells_summary.json` regenerates `analysis_full.json` from that record alone, and `tests/test_grid_analysis.py` fails if the two drift apart. `analyze_grade.py` resolves its own paths and refuses to overwrite a richer analysis with a thinner one (previously a fresh clone silently rewrote the aggregate into an empty file). The top-1-vs-top-2 `overlap` flag in `jax_port/stats.py` is also corrected: its condition was inverted in a way that reported non-overlap almost always, overstating significance — the rankings themselves are unchanged and the regenerated aggregate matches the published one.

### 15.1. PA1 — Pipeline and Measured Throughput (04/09/2026, venv `/root/procgen-jax`, WSL2, `coinrun`, Random Actions, 3000 Steps)

Reproducible command (executed via WSL, without torch/sb3/cv2 dependencies in the port venv):

```bash
wsl -e env PYTHONPATH=/mnt/c/Users/Acer/Downloads/MLE \
  /root/procgen-jax/bin/python \
  /mnt/c/Users/Acer/Downloads/MLE/jax_port/bench_throughput.py \
  --game coinrun --num-envs 1 4 16 --steps 3000 --seed 42 \
  --out /mnt/c/Users/Acer/Downloads/MLE/jax_port/pa1_throughput.json
```

| `num_envs` | Raw Env FPS (CPU, autoreset) | Env→GPU FPS (Transfer + `JIT uint8→float32 /255` with step synchronization) |
|---:|---:|---:|
| 1 | 12,618.2 | 301.5 |
| 4 | 13,272.9 | 910.1 |
| 16 | 18,695.5 | 3,023.8 |

Source: `jax_port/pa1_throughput.json`. Standard `gym`/`np.bool8` deprecation warnings during execution are expected due to legacy ProcGen bindings.

> **Empirical Interpretation:** (1) Raw CPU simulation exhibits sub-linear scaling from 1→16 envs (12.6k→18.7k) due to Python single-process synchronous step loops; multi-process C++ wrappers (`gym3`) offer headroom. (2) Step-by-step synchronization (`asarray` + JIT + `block_until_ready` per step) dominates overhead: with 1 env, the pipeline sustains ~`300 FPS`, matching the legacy SB3 training baseline (~`300 FPS`, section 1.4) — but here **without network updates**. (3) Vectorized batching amortizes host-to-device synchronization: 16 envs yield ~`3k FPS` pre-learner, representing a ~`10×` headroom margin before accounting for PPO backpropagation. Path A throughput was subsequently measured with active gradient updates (§15.2), surpassing this 3k benchmark.

**Grid Execution Status.** Full execution of `run_grade.py` across 10 suites with 5 seeds and `--eval-full` was completed on 06/09/2026 (615/615 cells completed), benchmarking ranking consistency against sections 3–12 and confirming statistical parity within inter-seed variance (§15.4.2).

### 15.2. PA2-Speed — Active PPO Training at 7.0–8.6k SPS (04/09/2026, `/root/procgen-jax`, WSL2, RTX 4070 Laptop)

Learner components: `jax_port/networks.py` (NatureCNN matching `models/sb3_extractors.py:8`, NHWC layout), `jax_port/ppo.py` (clipped surrogate objective, value clipping, entropy bonus; lr 3e-4, γ 0.99, λ 0.95, clip 0.2, 3 epochs, vf 0.5, ent 0.01, grad clip 0.5, adv norm — matching study hyperparameters in `compare_suite.py:26`), `jax_port/train.py` (64 C++ `gym3` environments, rollout 128, minibatch 1024).

```bash
wsl -e env PYTHONPATH=/mnt/c/Users/Acer/Downloads/MLE \
  /root/procgen-jax/bin/python \
  /mnt/c/Users/Acer/Downloads/MLE/jax_port/train.py \
  --game coinrun --timesteps 100000 --seed 42 --num-envs 64
```

| Game | Steps | Training Wall-Clock | **Training SPS** | Train Return | Eval Unseen (10 eps) | SB3 Baseline |
|---|---|---:|---:|---:|---:|---|
| `coinrun` seed 42 | 106,496 | 12.3 s | **8,647** | 7.50 | 3.0 | ~120–300 (≈7 min/50k, §2) → **~30–70×** |
| `starpilot` seed 42 | 106,496 | 14.6 s | **7,281** | 2.05 | 1.9 | ~167 (≈10 min/100k, §1.4) → **~40×** |

Sources: `jax_port/pa2_coinrun_100k.json`, `jax_port/pa2_starpilot_100k.json`.

> **Empirical Interpretation:** (1) The initial 3–5k SPS target was exceeded during full gradient updates — executing 100k steps in ~12–15 s vs ~7–10 min in single-env SB3. (2) Controlled comparison against equally parallelized SB3 is documented in §15.3, attributing ~1.1× to framework execution. (3) One-time startup costs: XLA compilation and kernel autotuning (~3.5 min, cached under `/tmp/jax_port_cache`) + GPU context allocation (~4 s). (4) Returns demonstrate active policy convergence. (5) Host-side minibatch slicing avoids device gather latency (~40× faster).

**Parity Considerations.** Acceptance criteria for the JAX port: **reproduce ranking orders and statistical error bounds within inter-seed variance**, rather than maximizing throughput in isolation.

**Artifact Organization.** Port implementation resides within `jax_port/`: core modules (`vector_env.py`, `bench_throughput.py`, `networks.py`, `backbones.py`, `ppo.py`, `dqn.py`, `augment.py`, `exploration.py`), trainers (`train.py`, `train_dqn.py`, `train_hrl.py`), verification suite (`stats.py`, `run_grade.py`, `tests/`), controlled benchmark scripts (`bench_sb3_paired.py` at root + `paired_*.json`), and empirical outputs (`pa1_*`, `pa2_*`). Root study files (`models/`, `results/`) remain intact.

### 15.3. Controlled Benchmark Comparison — Identical System, Same Day (05/09/2026, `coinrun`, 100k Steps, Seed 42)

Benchmarking conditions: Training loop wall-clock time isolated (excluding environment initialization, evaluation callbacks, and disk serialization); identical PPO hyperparameters (lr 3e-4, n_steps 256, 3 epochs, γ 0.99, λ 0.95, clip 0.2); sequential execution on the same RTX 4070 Laptop. Configurations: **A** SB3 matching original study (`DummyVecEnv` n=1, batch 64, `bench_sb3_paired.py`); **B** Parallel SB3 (`SubprocVecEnv` n=64, batch 1024); **C** JAX (`jax_port/train.py`, 64 envs `gym3`, batch 1024).

| Arm | Steps | Training Wall-Clock | **SPS** | Source |
|---|---|---:|---:|---|
| A: SB3 n=1 / batch 64 (Original Study) | 100,096 | 175.5 s | **570** | `jax_port/paired_sb3_n1.json` |
| B: SB3 n=64 / batch 1024 | 114,688 | 16.8 s | **6,838** | `jax_port/paired_sb3_n64.json` |
| C: JAX n=64 / batch 1024 | 106,496 | 14.2 s | **7,506** | `jax_port/pa2_coinrun_100k_rerun.json` |

> **Throughput Decomposition (~30–70× vs ~120–170 SPS reported):** (1) ~4× stems from removing logging/callback overhead present during full benchmark runs (570 paired vs ~120–170 reported in config A). (2) ~12× is driven by environment parallelism — transitioning A→B (570→6,838), attainable in either framework. (3) **~1.1× represents framework and environment bindings** — comparing B→C (6,838→7,506), PyTorch/`SubprocVecEnv` vs JAX/`gym3` C++ bindings. While JAX is ~15× faster than the original single-env study configuration under controlled conditions (7,506 vs 570), it is only ~10% faster than an equally parallelized SB3 setup — demonstrating that batch parallelization accounts for the vast majority of throughput gains.

### 15.4. Comprehensive Port — Coverage Map and Test Evidence (05/09/2026)

> **Scope note (11/10/2026):** the study proper is Procgen (sections 1-14). Everything under §15.4 is the
> JAX port and its appendices — including the multi-agent work in §15.4.5, which is a negative result
> about this code path rather than a claim about SMAX or about cooperative MARL in general.

Each component of the original study corresponds to a JAX module in `jax_port/`, with test coverage:

| Primary Study (sections 1–12) | JAX Port Implementation | Executed Verification Test |
|---|---|---|
| 16-Architecture Suite (`compare_suite/new_archs/combined`, §3) | `backbones.py` (13 extractors) + `train.py --extractor` | Zoo 13/13 shapes+parameters (§15.4.1); smoke training 8/8; mini-grid 16 cells OK |
| `mlp_vector` (`procgen_wrapper.py:55`, §3.12) | `--obs vector` mode (luminance + mean pooling, no cv2 dependency) | Smoke training verified; `test_smoke` suite uses this pipeline |
| Augmentations: crop/color/noise (§3.3/3.11) | `augment.py` (p=0.5/float) + `--augment` | Smoke crop and contrastive+noise verified |
| ICM/RND/NGU (`compare_maze_heist.py:16`, §3.6) | `exploration.py` (beta=0.01, online/step, inverse dynamics behavior documented) | Smoke maze-ICM/RND/NGU + heist-RND verified |
| LSTM-Attention stateless (§3.5) | `LSTMAttention` (repeat-4 + BiLSTM + MHA) | Output shapes and smoke training verified |
| VAE/AE/Recon/Contrastive (§3.2) | `VAEBackbone` (reparameterization) + twins `ae`/`recon` ≡ classic + `contrastive` ≡ classic+noise | Smoke vae/ae/contrastive+noise verified |
| PPO/A2C/DQN/QR-DQN (§12) | `train.py --algo`, `dqn.py` + `train_dqn.py` (100k buffer, eps 1→0.05 over 25%, lr 1e-4, QR-200, `--lr` sensitivity) | Smoke dqn/qrdqn/a2c verified |
| HRL flat/skip4/hrl/hrl_learned (§11) | `train_hrl.py` (exact skill mapping, DUR=4, frame-based budget, low-level `π(a\|obs, z)`) | Smoke across all 4 arms verified |
| Evaluation: 100+100+15, generalization gap (§3.10–12) | `--eval-eps/--eval-det-eps/--eval-train-eps` + `gen_gap` across 3 trainers | Regression tested across PPO/DQN/HRL |
| Statistics: CI/Cohen/AUC (§3.8–3.9) | `stats.py` (Student's t, Cohen's d, trapezoidal integration) | `test_stats` validated against analytical values |
| Grids + budget scaling + hard + pilot + extensions (§2–3.13) | `run_grade.py` (suites: main/exploration/algo/hrl/budget/hard/pilot/spr/gnn/aux, `--distribution`, resume via `master.json`) + `spr.py` + `contrast.py` | Mini-grid 26/26 cells verified (§15.4.1) |
| Controlled SB3 vs JAX Benchmark (§15.3) | `bench_sb3_paired.py` + `paired_*.json` | Arms A/B/C empirically profiled |
| Cross-game response measure (§3.6.2, §0.1) | `jax_port/response.py` + `train.py --response` (added 24/10/2026) | `test_response`: TV helper + full-measure smoke on a built policy (§19); in-venv smoke: `classic`/maze 20k steps → TV `0.139`, `2` argmax, responsive |

#### 15.4.1. Empirical Test Evidence (05/09/2026, `/root/procgen-jax`, RTX 4070)
- `test_stats`: `STATS_OK` across both environments (Win py3.10 and Linux venv).
- `test_parity`: `gym3` batching matches single-instance `gym` step-for-step across 200 steps with 0 numerical discrepancies (`PARITY_OK`).
- `test_zoo`: 13/13 forward passes pass shape checks; classic CNN parameter count matches at 608,944 (~600k in study); identical backbones verified (`ZOO_OK`).
- `test_smoke`: End-to-end MLP training loop verified (`SMOKE_OK`).
- Addressed bugs and edge cases: Flax default SAME padding inflating parameters to 2.1M (corrected to VALID); `stop_gradient` syntax; DQN entrypoint hook; device-side gather latency; `np.trapezoid` compatibility under `numpy 1.26`.
- Mini-grid execution: 26/26 cells (HRL 4 + main 16 + algo 6) completed in ~6 min; resume functionality verified via `master.json`.
- Test execution: `wsl -e env PYTHONPATH=... /root/procgen-jax/bin/python -m jax_port.tests.run_tests` (stats+parity+zoo+smoke, ~3 min); full grid executed via `run_grade.py --suite <s> --games <g> --seeds 42-46 --timesteps 100000 --eval-full` (10 suites, 615/615 cells completed, documented in `analysis_full.json`).
- **Path conventions (updated 23/09/2026):** every `jax_port/` module resolves its own package directory, so `run_grade.py` writes into `jax_port/results_grade/` and `analyze_grade.py` / `analyze_temporal_hard.py` read and write inside `jax_port/` regardless of the current working directory. No source file hardcodes `/mnt/c/...` or `C:\Users\...` any more (the literal commands quoted in sections 14-15 are kept as the historical record of what was executed); a WSL invocation is just `PYTHONPATH=<checkout> <venv>/bin/python -m jax_port.<module>`. The orchestration scripts take overrides instead of literals: `marl_10m_run.sh` / `marl_2s3z_run.sh` honor `REPO`, `PY`, `SEEDS`, `LR`, `ENT`; `continue_grade.ps1` honors `REPO` and `PROCGEN_JAX_VENV` (default `/root/procgen-jax`). `.gitattributes` pins `*.sh` to LF because those scripts are executed inside WSL from the Windows checkout.
- **Figures:** `jax_port/figures/01-06.png` (global CI, paired speedup, budget scaling, HRL, algorithm families, top-10 n=10) generated programmatically by `make_figures.py` directly from serialized JSON results.
- **`hrl_learned` at 500k Steps:** On `plunder` across 3 seeds: learned achieves 4.19 vs fixed 3.42 (at 100k: 4.23 vs 3.19). Advantage persists stably; co-training sustains performance without divergence under 5× budget.
- **Dreams Reconstruction (VAE/AE):** `jax_port/dream.py` (mirrored decoders with SAME padding for exact 64×64 reconstruction) — 20k frames on `bossfight`: VAE BCE 0.37 / KL 0.02, AE 0.32 (~30 s each); generated `jax_port/dreams/dreams_panel.png` + `bossfight_dreams.gif` (300 steps, structured visual dreams: pixel std 44–60 vs 57 ground truth). Rendered as GIF due to upstream `imageio-ffmpeg` audio argument incompatibility in the venv.
- **Full Dreamer 1M Steps:** `dreamer.py` + `train_dreamer.py` — `coinrun` seed 42, 1M frames in 656 s (**1,524 SPS**, ~10× SB3; lower than PPO-JAX due to sequence-level world model updates and imagination rollouts). World model converged (`dreamer_imagined.gif` visual std 61.7 vs 0.7 at initialization), while policy actor plateaued at 0.0 return without reward symlog normalization.
- **Dreamer Actor Tuning (Sweep B/C/D):** Baseline A (raw rewards, entropy 3e-4) returned 0.0; **B (symlog rewards, entropy 3e-4) reached 3.0 return**; C (symlog, entropy 1e-3) reached 2.5; D (raw, entropy 1e-3) reached 0.5. Verdict: **Symlog reward transformation resolves the actor bottleneck**; default updated in `train_dreamer.py`. Artifacts in `sweep.json` and `jax_port/dreams/`.
- **Dreamer reached by the grid runner (27/09/2026):** the `dreamer` suite was wired into `run_grade.py`'s cell builder on 25/09 but never added to `--suite`'s argparse `choices`, so `--suite dreamer` exited with `invalid choice` and the branch had executed **zero** cells — `results_grade/dreamer/` did not exist. The published Dreamer rows above come from `train_dreamer.py` directly, which is why the dead branch stayed invisible while looking integrated. With the choice registered, one cell was smoked end-to-end through the grid path (`JAX_PLATFORMS=cpu`, `coinrun` seed 42, 20k frames, 8 envs, 3 eval eps): 424 s at 47 SPS, `train_ret_mean20` 3.0, `eval_unseen` 3.33, and `analyze_grade.py`'s `Loader` parses the cell it wrote (`_cfg=dreamer`, `eval_unseen.mean` present, 0 unreadable, 0 no-eval). The artifact is deliberately parked in `results_grade/dreamer_smoke/` rather than `results_grade/dreamer/`: `dreamer` is in `RANKING_SUITES`, so a 20k-frame plumbing cell left in the graded directory would be folded into `cells_summary.json` and into everything downstream of it.

#### 15.4.2. Full Grid Evaluation Verdict (05–06/09/2026, 615 Cells, Eval 100+100+15)

Full execution: main 240 + exploration 40 + algo 70 + hrl 40 + budget 60 + hard 55 + pilot 20 + spr 30 + gnn 15 + aux 45. Academic analysis:

- **Parity Confirmed on Core Findings.** Global top-6 clustered at 1.32–1.24 with overlapping confidence intervals (`d=0.27`) — replicating the primary study finding (top tier statistically indistinguishable, §3.12). Algorithm families replicate closely: policy-based > value-based across all 3 games, `qrdqn` leads `bossfight`, `a2c ≈ ppo`, learning rate sensitivity is minimal. On `plunder`, `hrl_learned` leads (4.23), matching the study. Exploration shows an empirical tie: `ppo ≈ icm ≈ rnd ≈ ngu`, replicating the definitive protocol.
- **Top-1 per Game Variations within Expected Variance.** Port top configurations (`aug_noise` global, `contrastive` starpilot, `resnet18` bossfight) differ from SB3 study top models (`mlp_vector`, `mlp`, `aug_crop`), but all pairwise comparisons show `d ≤ 0.4` with overlapping CIs — reinforcing section 3.8: with 5 seeds, top-spot swaps reflect sample variance rather than methodological divergence.
- **Absolute Discrepancy Disentanglement.** Controlled isolation experiments showed: (1) JAX with SB3-sized batches (4 envs, minibatch 64) produced identical returns to large-batch JAX; (2) SB3 scaled to large batches (32 envs, minibatch 1024) increased `maze` performance to **5.33** (vs 2.80 in single-env SB3). Conclusion: Discrepancies stem from **parallel rollout batch dynamics**, not framework differences. Documented in `paired_sb3_big_maze.json`.
- **Environment Reproducibility.** Pinned via `jax_port/requirements-jax.txt` and documented in `jax_port/SETUP_VENV.md`.
- **Budget Scaling (§3.13):** `mlp` exhibits modest gains (2.04→2.74 in `starpilot`), while `resnet18` remains flat — confirming that expanded budgets do not alter architectural conclusions.
- **Top Cluster Expanded Evaluation (n=10 Seeds):** Sample size doubled across top 5 configurations (seeds 42–51, full evaluation): `bossfight` `impoola` 0.20 > `mlp` 0.09 (`d=0.73`, overlapping CIs); `starpilot` `aug_noise` 2.63 ≈ `impoola` 2.61 (`d=0.06`); `dodgeball` `impoola` 1.29 ≈ `mlp` 1.28 (`d=0.05`). Verdict: **Even with n=10, the top cluster remains statistically inseparable** — reinforcing section 3.8.
- **Extensions (No Baseline in Original Study):** `gat` achieves ~1.3 global return; `spr_aug > spr`; `acl > curl/cpc` across all 3 games; `hard` difficulty suppresses all returns to ~0.05.
- **Generalization Gap Re-Ranking (`gen_gap = eval_train − eval_unseen`):** Sorting by minimal generalization gap preserves relative order — top models: `aug_noise` (−0.47), `lstm_attention` (−0.45), `resnet18` (−0.44), `contrastive` (−0.41), with overlapping CIs throughout (`analysis_full.json`). High return correlates with strong generalization; models do not succeed via training-set memorization.

#### 15.4.3. Sample Budget Viability: 50k vs 100k Steps (Probe 05/09/2026, Classic/MLP × Coinrun/Starpilot × Seeds 42–43, Eval 30/30/15)

| Game | Extractor | 50k (Eval Unseen) | 100k (Eval Unseen) | Δ |
|---|---|---:|---:|---:|
| `coinrun` | `classic` | 4.50 | 5.00 | +0.50 |
| `coinrun` | `mlp` | 2.50 | 3.00 | +0.50 |
| `starpilot` | `classic` | 2.02 | 3.05 | +1.03 |
| `starpilot` | `mlp` | 1.93 | 2.53 | +0.60 |

> **Verdict:** 50k steps preserves relative ranking order (`classic > mlp` across all 4 pairs), and deltas (+0.5–1.0) are smaller than seed-to-seed variance. Combined with §3.13 (stagnation from 100k→500k), the conclusion is: **50k steps suffices for relative ranking** (the study's objective), while **100k provides higher absolute scores and safety margin**. The full grid runs at 100k steps.

#### 15.4.4. Measured Runtime Reductions (Identical RTX 4070 Laptop)
- `coinrun` 50k, 1 model: SB3 ~7 min (§2) → JAX ~8 s training (**~50×**).
- 100k cell: SB3 ~10 min → JAX ~14 s training + ~40 s full evaluation ≈ 1 min end-to-end (**~10×**).
- 15 h SB3 run (~8M steps): → ~18 min at 7.5k SPS.
- Complete grid (615 cells, 10 suites × 5 seeds, full evaluation): SB3 estimated at multiple weeks → JAX completed in ~7–8 h end-to-end (**completed 06/09, `master_full.json` 615/615 verified**; serialized in `analysis_full.json`). Cumulative measured training time: **5.0 h** vs ~90–100 h estimated in original study → **~18× in training, ~12× end-to-end with comprehensive evaluation**.
- **SPR Extension:** `spr.py` — shared encoder + target EMA (τ=0.99) + transition MLP, MSE loss on normalized latents with crop augmentation, auxiliary Adam lr 1e-4. Smoke test verified.
- **GAT Extension:** `GATPatch` — graph over 64 8×8 patches, 2× GAT with 4 heads + residual/LayerNorm, mean-pool → FC512 (141k params). Smoke test on `coinrun`: return 6.0, verified.
- **CURL/CPC/ACL Extensions:** `contrast.py` — shared encoder + target EMA, InfoNCE τ=0.1 with minibatch negatives. Resolved gradient singularity on exact zero vectors via epsilon regularization `sqrt(sum + eps)`.
- **MARL/SMAX Integration (`jaxmarl 0.1.0`):** `jax_port/marl/` — 7 algorithms (ippo, mappo, vdn, qmix, mapoca, cte, tarmac) on map `3m` × 3 seeds × 1M steps (21/21 runs, 1.5–3.3k SPS). Outcome: **Win-rate evaluated at 0.0 across all flat feedforward implementations** (`analysis_full.json`) — flat feedforward policies fail to coordinate in `3m` within 1M steps. Diagnosed that JaxMARL literature requires **recurrent GRU-128 + 10M steps**, implemented in `recurrent.py` with sequential BPTT over window T.
- **Exploration $\beta=0$ Control Suite (`jax_port/exploration_control_summary.json`):** Evaluated all 30 control cells (3 arms × 2 games × 5 seeds at 100k steps with `--explore-beta 0.0` where the bonus network trains but nothing is injected into the reward). Outcome: the chaotic drift floor ($|arm@0 - ppo|$) exceeds the published delta ($|arm@0.01 - ppo|$) across all 6 groups (`heist_icm`: drift 0.22 vs effect -0.18; `heist_rnd`: drift 0.38 vs effect -0.18; `maze_ngu`: drift 2.50 vs effect -0.58). **Scientific Verdict:** The published $\pm 0.5$ exploration spread in the JAX port is entirely within chaotic run-to-run drift and not attributable to intrinsic curiosity mechanisms.

#### 15.4.5. Recurrent MARL at 10M Steps with GRU-128 — a Negative Result about this Port's Q-learning Path (06/09/2026, re-measured 24 and 27/09/2026)

> **What this section is, and what it is not.** It is a measured negative result about *the JAX port's
> off-policy cooperative Q-learning path*: after the 21/09/2026 fixes (per-environment sequential
> buffer, terminal mask in the bootstrap, mixer target scaled by $r+\gamma(1-d)Q_{tot}^{\text{target}}$),
> **all 12** Q-learning cells — 9 on `3m` plus the 3 `2s3z` recurrent cells re-measured on 27/09 — reach
> their full budget and still return win-rate 0.000, with `delta_vs_published` 0.000. It is **not** a SMAX
> benchmark and must not be compared against published JaxMARL numbers: the recipe differs on the two
> axes the literature is sensitive to (32 parallel envs with 2 epochs here vs 8 envs with 4 epochs upstream, and
> full-episode TD windows vs length-32 sequence chunks). The on-policy arms (`ippo`, `mappo`, `mapoca`,
> `cte`, `tarmac`, `ippo-recurrent`) were never invalidated by those fixes and stand on their own.

Evaluated protocol: IPPO-rec and QMIX-rec on map `3m`, seeds 42–44, 10M steps each (~1h15 for IPPO at ~2.1k SPS; ~1h55 for QMIX at ~1.4k SPS per run). **All 6 runs converged to 0.0 evaluation win-rate across all seeds.**

| Algorithm | Mean Training Return | Eval Win-Rate (32 eps) | Eval Return |
|---|---:|---:|---:|
| IPPO-rec (3 seeds) | 1.6–1.9 | 0.0 | 1.6–1.9 |
| QMIX-rec (3 seeds) | ~0–0.3 | 0.0 | 0.0–0.3 |

**Ruled-out Failure Modes:** Pipeline correctness verified (hidden state carry, agent-specific done masking, sequence slicing). Extended budget (10M) and map configuration adjustments (`--no-walls`) yielded identical 0.0 win-rates across seeds.

**Empirical Diagnosis:** IPPO-rec acquires partial credit (eliminating 1–2 opposing units before defeat — stable return ~1.7), while QMIX acquires minimal credit (~0). Symmetric 3v3 combat rewards strict focus-firing and damage avoidance; partial-credit gradients form a local optimum. Discrepancies relative to JaxMARL baseline literature stem from rollout batch composition (upstream uses 8 parallel envs with 4 epochs/rollout, vs 32 envs with 2 epochs here) and full-episode TD windows vs length-32 sequence chunks.

**Positive Control on `2s3z` (09/09, QMIX-rec 10M × 3 seeds):** Tested on `2s3z` (the standard JaxMARL benchmark map). Outcome: **0.0 win-rate across all 3 seeds** (returns 0.0–0.2). This confirms that the behavior is governed by specific off-policy Q-learning hyperparameter recipes rather than environment mechanics or hidden state handling. SMAX investigation concluded without further resource allocation. Checkpoint restoration functionality implemented under `--ckpt`/`--resume`. *(Status 27/09/2026: these three cells predated the 21/09 fixes by twelve days, so this sentence was originally an inference from a run whose mixer target never saw the external reward. They have since been re-measured under the corrected code with identical hyperparameters and the outcome did not move — see the re-measurement block below.)*

**Algorithmic Root Cause Resolution (21/09/2026):**
A deep algorithmic audit of `jax_port/marl/recurrent.py` identified a critical deficiency in `make_ql_seq_update`: the mixer target computation omitted the environment reward and discount terms during sequence Bellman target construction:
```python
# Previous buggy formulation:
ttot = mixer.apply(target_mixer_params, target_qs, states)
# Mathematically corrected Bellman target:
target_tot = rew + gamma * (1.0 - done) * mixer.apply(target_mixer_params, target_qs, next_states)
```
Without scaling by $r + \gamma(1-d) Q_{tot}^{target}$, the mixer loss $L = (Q_{tot} - target)^2$ was deprived of the external reward signal. In addition to repairing the Bellman target equation, we integrated Polyak soft target updates (`--tau`) and configurable update frequencies (`--target-update-interval`) into `jax_port/marl/train_ql.py`, verified via unit tests in `jax_port/tests/test_marl.py` (`MARL_TESTS_OK`). Correction (23/09/2026): this paragraph previously stated a `--tau` default of `0.005`; the actual default is `tau = 0.0`, which selects the hard target copy every `500` gradient steps, and no published MARL cell passed `--tau` — so every MARL result in section 15.4.5 was produced with hard target updates. `run_grade.py` now exposes both flags so the grid can reach them.

**Definitive Empirical Re-Measurement under Corrected Code (24/09/2026, `jax_port/marl_remeasure_summary.json`):**  
To definitively resolve whether the 0.0 win-rate was caused by the code defects or fundamental optimization dynamics, all 9 Q-learning cells were re-measured from scratch via `jax_port/marl_ql_remeasure.sh` on map `3m` under the mathematically corrected codebase (verified contiguous sequential buffer, terminal masking, and target mixer scaling):
- `vdn` flat (1M steps × seeds 42–44): **0.000 eval win-rate** (returns 0.50–0.56)
- `qmix` flat (1M steps × seeds 42–44): **0.000 eval win-rate** (returns 0.00–0.49)
- `qmix-recurrent` (10M steps × seeds 42–44): **0.000 eval win-rate** (returns 0.00)

**The re-measurement was incomplete; it is now (27/09/2026, `jax_port/marl_remeasure_2s3z_summary.json`):** the 24/09 sweep paid for 9 cells, but the published grid carries **12** invalidated Q-learning cells — the `qmix-recurrent` 10M positive control on `2s3z` above ran on 09/09 under all three defects, and its 0.0 win-rate was the evidence for the claim that the failure is about hyperparameter recipes rather than map mechanics or hidden-state handling. Those 3 cells were re-measured from scratch under the corrected code with otherwise identical settings (`MAPS=2s3z RUN_FLAT=0`, so no unpublished flat `2s3z` cells were invented alongside them; no `--tau`, hard target copy every 500 gradient steps), into their own directory and their own summary so the frozen `3m` numbers stay untouched: 10M steps × seeds 42–44 → **0.000 eval win-rate** (returns `0.00`, `0.00`, `0.12`), all three cells reaching their full budget and marked usable, `delta_vs_published` = 0.000 on each, 6.9 h wall (`10602 s` / `7416 s` / `6948 s`, at 943–1439 SPS).

> 🔬 **Definitive Scientific Verdict:** Re-measurement confirms that the 0.0 win-rate in SMAX symmetric 3v3 combat is **not an artifact of the mixer target omission or buffer bugs** — and that now holds on both maps, `3m` (9 cells) and `2s3z` (3 cells), i.e. every Q-learning cell the study publishes was produced by code whose Bellman target actually sees $r + \gamma(1-d)Q_{tot}^{\text{target}}$. Rather, off-policy cooperative Q-learning without domain-specific credit assignment shaping or dense reward curricula fails to converge to coordinated focus-firing in either map. This settles the methodological integrity of the SMAX benchmark.

#### 15.4.6. Temporal Architecture Bake-Off — 10 Memory Models, 100k Steps, 5 Seeds (06–07/09/2026, Suite `temporal`)

Research question: In ProcGen with frame stack $k=4$, does explicit recurrent memory yield measurable advantage? Evaluated on `heist`, `maze`, and `jumper`. Full suite completed at **150/150 runs** (`master_temporal.json`):

| Game | 1st | 2nd | 3rd | Lowest | Distribution |
|---|---|---|---|---|---|
| `heist` | `s4` **3.10** | `cnn1d` 2.78 | `s5` 2.76 | `transformer_xl` 1.42 | Top-7 show overlapping 95% CIs |
| `jumper` | `s5` **3.56** | `mlp` 3.54 | `mamba` 3.44 | `transformer` 2.72 | Top-4 within 0.12 of each other |
| `maze` | `cnn1d` **5.14** | `tcn` 5.06 | `lstm` 4.94 | `transformer` 3.94 | Top-6 within ~0.4 of each other |

**Findings:** (1) *No recurrent memory architecture separates from simple 1D temporal convolution or stateless MLP* within 100k steps — replicating section 3.8: differences reflect sample variance rather than systematic superiority. (2) Consistent underperformers: `transformer` and `transformer_xl` — higher parameter counts (~1M) with self-attention struggle under a 100k step budget. (3) State space models (`s4`/`s5`/`mamba`) perform competitively without dominating feedforward baselines.

#### 15.4.7. Memory Models under Extended Budget (500k Steps) on Hard Regimes (07/09/2026, Suite `temporal_hard`)

Evaluated the same 10 architectures on `heist` (hard mode) and `bossfight` across 500k steps × 5 seeds with full evaluation (100/100 runs completed):

| `heist` hard | Mean Return | `bossfight` hard | Mean Return |
|---|---:|---|---:|
| `s5` | 2.02 | `s5` | 0.20 |
| `tcn` | 1.98 | `gru` | 0.17 |
| `s4` | 1.92 | `s4` | 0.15 |
| `gru` | 1.70 | `cnn1d` | 0.14 |
| `mlp` | 1.66 | `tcn` | 0.13 |
| `cnn1d` | 1.60 | `mamba` | 0.07 |
| `lstm` | 1.56 | `lstm` | 0.05 |
| `mamba` | 1.50 | `mlp` | 0.05 |
| **`transformer`** | **1.50** | `transformer` | 0.03 |
| `transformer_xl` | 1.46 | `transformer_xl` | 0.01 |

**Analysis:**
1. **Transformer models do not scale with budget:** Relative to the 100k baseline, `transformer` remains at the bottom tier in both environments. The hypothesis that "Transformers merely lacked budget" is rejected up to 500k steps.
2. **State space models (`s5`, `s4`, `mamba`) lead in hard heist,** consistent with 100k trends, though confidence intervals overlap with top convolutional baselines.
3. **`bossfight` suppresses all architectures** (0.0–0.2), offering no meaningful temporal differentiation.
4. **Conclusion:** Explicit temporal memory does not provide significant advantage over feedforward baselines in standard Procgen environments under 100k or 500k budgets.

#### 15.4.8. Single-Frame Regime (`stack=1`): Audit and Paired Recurrent Memory Benchmark (20/09/2026, `starpilot`, 100k Steps, Seed 42)

A key methodological critique of the preceding bake-offs (§15.4.6 and §15.4.7) is their reliance on `stack=4`. Stacking 4 frames provides a first-order Markovian approximation (velocity via finite differences), potentially masking the necessity of explicit recurrent memory.

**Pre-Experiment Infrastructure Audit:**
Auditing `temporal.py` and `train.py` revealed that early temporal models (`LSTMStack`, `S5Stack`, etc.) operated recurrence **strictly within the 4 stacked frames**, remaining *stateless* across environment transition steps ($t \to t+1$, resetting hidden state carry per observation). Executing those implementations with `stack=1` would reduce them to sequence length 1 (effectively feedforward linear layers).

To rigorously evaluate memory under strict partial observability (POMDP) without frame stacking, true inter-step recurrence was implemented (`jax_port/recurrent_step.py`: `RecurrentLSTMBackbone` and `RecurrentS5Backbone`), with hidden state carries maintained across rollouts and reset strictly on `done`, evaluated via `jax_port/bench_temporal_stack1.py` on `starpilot` (where $k=1$ removes direct velocity observation):

| Model | Architecture | SPS | Wall (s) | Train (Last 20 eps) | Eval Unseen (20 eps) | 95% CI | Source |
|---|---|---:|---:|---:|---:|---|---|
| **`classic`** | NatureCNN feedforward (*stateless*, $k=1$) | **7,566** | **14.1 s** | 2.65 | **3.10** | [1.73, 4.47] | `results_stack1_bench_100k.json` |
| **`recurrent_s5`** | ClassicCNN + S5 SSM step-carry ($k=1$) | 5,088 | 20.9 s | 2.25 | **2.20** | [1.29, 3.11] | `results_stack1_bench_100k.json` |
| **`recurrent_lstm`** | ClassicCNN + LSTM cell step-carry ($k=1$) | 4,982 | 21.4 s | **3.20** | **1.45** | [0.64, 2.26] | `results_stack1_bench_100k.json` |

**Scientific Findings:**
1. **Simple feedforward policy (`classic`) superior in generalization:** Even under forced partial observability ($k=1$), stateless NatureCNN achieved the highest return on unseen levels (3.10), outperforming both recurrent architectures.
2. **Latent State Overfitting in LSTM:** `recurrent_lstm` attained the highest training return (3.20) but collapsed to 1.45 on unseen levels (generalization gap of −1.75). In small budgets (100k), recurrent capacity primarily memorized training trajectories across the 200 levels.
3. **Stability of State Space Models (S5):** `recurrent_s5` exhibited near-zero generalization gap (2.25 train $\to$ 2.20 unseen eval), though it did not outperform the feedforward baseline.
4. **Computational Throughput:** Step-by-step recurrent state tracking incurred a ~33–34% throughput penalty (~7.5k $\to$ ~5.0k SPS).
5. **Final Verdict:** Even without frame stacking, explicit sequential memory does not offset optimization complexity in the 100k step Procgen regime; feedforward policies remain more sample-robust.

**Architectural Feasibility Taxonomy of 10 Models at `stack=1`:**

| Category | Model at `stack=4` | Mechanism at $k=4$ | Structural Behavior at $k=1$ | Status at `stack=1` |
|---|---|---|---|---|
| **Convolutional & Stateless** | `mlp` (`MlpStack`) | MLP over concatenated frames | Degenerates to standard MLP over 1 frame. No memory. | Redundant with `classic`. |
| | `cnn1d` | 1D temporal conv over 4 frames | At $T=1$, 1D conv degenerates to dense linear layer. No memory. | Requires explicit FIFO frame buffer. |
| | `tcn` | Causal dilated conv over 4 frames | Causal conv at $T=1$ degenerates to linear projection. No memory. | Requires explicit FIFO frame buffer. |
| **Pure Attention** | `transformer` | Self-attention across 4 frames | 1-token self-attention is trivial identity ($QK^T = 1$). Degenerates to feedforward. | Inapplicable without sequence length. |
| | `transformer_xl` | Self-attention with segment cache | Could maintain FIFO cache of latents, but requires 1-frame input adaptation. | Adaptable via step-by-step cache. |
| **Recurrent & SSMs** — $h_t = f(h_{t-1}, x_t)$ | `lstm` | LSTMCell over 4 frames | Requires $(c_t, h_t)$ carry across environment steps, reset on `done`. | **Implemented** (`recurrent_lstm`). |
| | `s5` | Associative scan over 4 frames | Scan reduces to recursive step $h_t = \bar{A}h_{t-1} + Bu_t$. | **Implemented** (`recurrent_s5`). |
| | `gru` | GRUCell over 4 frames | Requires carry $h_t$ across environment steps, reset on `done`. | Viable (direct extension of LSTM). |
| | `s4` | FFT convolution over 4 frames | Step-by-step operates as discrete SSM. | Viable (SSM variant). |
| | `mamba` | Selective scan over 4 frames | Operates as input-dependent recurrent step. | Viable (time-varying SSM). |

> **Engineering Conclusion:** Temporal convolutions (`cnn1d`, `tcn`) possess no latent state and require an external observation buffer (which merely reconstructs frame stacking). Only models with recurrent latent state equations $h_t = f(h_{t-1}, x_t)$ constitute valid memory architectures at $k=1$. `recurrent_lstm` and `recurrent_s5` represent the gated recurrent and state-space families, respectively.

#### 15.4.9. Single-Frame POMDP Occlusion Benchmark on Caveflyer & Memory Regularization (21/09/2026)

Following the finding in §15.4.8 that feedforward policies outperformed recurrent models in `starpilot` (where velocity is latent but all entities remain on-screen), we conducted a rigorous follow-up addressing two key methodological factors:
1. **Memory State Transition Alignment:** Auditing the rollout buffer in `jax_port/bench_temporal_stack1.py` revealed that memory carries were saved *after* stepping the environment (`b_mem[t] = next_mems`). For correct recurrent policy gradient backpropagation through time (BPTT), the pre-transition carry $h_{t-1}$ must be stored at step $t$ (`b_mem[t] = mems`), aligning the state that conditioned action $a_t$ with observation $s_t$.
2. **Hidden Carry Regularization (`RegularizedRecurrentLSTMBackbone`):** To prevent LSTM hidden state drift and memorization overfitting under low-data budgets, we implemented `RegularizedRecurrentLSTMBackbone` in `jax_port/recurrent_step.py`, introducing `LayerNorm` on the recurrent hidden carry $(h_t, c_t)$ along with AdamW weight decay ($1\times 10^{-4}$).
3. **Authentic Spatial Occlusion Benchmark (`caveflyer`):** Unlike `starpilot` where all adversaries are visible in the viewport, `caveflyer` features winding subterranean caverns and occluded target goals outside the camera frame — constituting an authentic Partial Observability (POMDP) environment where path memory is structurally required.

We benchmarked `classic` (feedforward NatureCNN), `recurrent_lstm`, `regularized_recurrent_lstm`, and `recurrent_s5` under `stack=1`, 57k steps, across a 5-seed evaluation ($n=5$, seeds 42–46) with 20 episodes on unseen levels per seed (`seed+1000`), persisted in `results/caveflyer_multiseed_bench.json`:

| Model | Architecture | Mean Return ± Std | 95% Student's $t$ CI | Cohen's $d$ vs Classic | Per-Seed Breakdown (42–46) | Source |
|---|---|:---:|:---:|:---:|:---:|---|
| **`recurrent_lstm`** | CNN + LSTM pre-transition carry ($k=1$) | **3.90 ± 1.14** | **[2.48, 5.32]** | **+0.57** (medium-to-large) | [2.5, 5.0, 4.0, 3.0, 5.0] | `caveflyer_multiseed_bench.json` |
| `recurrent_s5` | CNN + S5 SSM step-carry ($k=1$) | 3.60 ± 1.14 | [2.19, 5.01] | +0.33 (small-to-medium) | [2.5, 3.5, 5.5, 3.0, 3.5] | `caveflyer_multiseed_bench.json` |
| `classic` | NatureCNN feedforward (*stateless*, $k=1$) | 3.20 ± 1.30 | [1.58, 4.82] | — (baseline) | [1.0, 4.0, 4.0, 3.0, 4.0] | `caveflyer_multiseed_bench.json` |
| `regularized_recurrent_lstm` | CNN + LayerNorm carry + AdamW ($k=1$) | 3.20 ± 0.91 | [2.07, 4.33] | +0.00 (neutral) | [3.0, 4.5, 3.0, 2.0, 3.5] | `caveflyer_multiseed_bench.json` |

> **Scientific Conclusion (Definitive Multi-Seed Re-measurement):**
> Across $n=5$ seeds, **`recurrent_lstm` retains the highest mean unseen return ($3.90$ vs $3.20$, $+22\%$ relative advantage) with a positive effect size ($d = +0.57$)**. However, due to high inter-level variance in procedural generation, the 95% Student's $t$ confidence intervals overlap ($[2.48, 5.32]$ vs $[1.58, 4.82]$). 
> This contextualizes the exploratory single-seed pilot: the single-seed point estimate ($4.00$ vs $2.50$, $+60\%$) substantially exaggerated the true advantage. While temporal state-carry provides a consistent, measurable directional benefit in occluded POMDP environments ($d=+0.57$), procedural variance necessitates multi-seed aggregation to prevent over-claiming statistical separation. S5 SSMs also achieve a positive effect size ($d=+0.33$), whereas hidden-state LayerNorm regularization (`regularized_recurrent_lstm`) reduces return variance ($\pm 0.91$ vs $\pm 1.14$) at the expense of peak return.

## 16. Process References (Beyond Section 10)

- Gate PA0 and Environment: PPA `deadsnakes` (Python 3.10 on Ubuntu 24.04); wheel `procgen-0.10.7-cp310-cp310-manylinux`; `jax[cuda12]`; pins: `numpy==1.26.4`, `gym==0.26.2`, `flax==0.10.7`, `optax==0.2.8`, `JAX==0.6.2`.
- Gym/Gymnasium API Boundaries: `procgen_wrapper.py:6,55,85` (original study) serving as the baseline interface specification.
- GPU Workload Discipline: Single active process on 8 GB VRAM; graceful termination with serialized state checkpoints (established in `compare_suite_retrain.py`).
- Reference Hardware Environment: `WSL2 Ubuntu 24.04`, `NVIDIA RTX 4070 Laptop 8 GB`, `cuda:0`, `Python 3.10.11` (original study) / dedicated `py3.10` venv (JAX port).
