# Verification, Test Suites and Legacy Prototypes

> Part of the study's documentation — start at the [README](../README.md). Sections 19 and 20. Section numbers are the README's original ones, so every `§x.y` reference in the code, tests and other pages still names the same section.

## 19. Architecture Verification & Test Suites

The repository contains automated unit tests across both PyTorch and JAX backends:

- **PyTorch Extractor Test Suite (`tests/test_pytorch_extractors.py`):**
  - Validates forward pass output dimensions $(B, 512)$ for all 12 extractor configurations (`ClassicCNNExtractor`, `AttentionCNNExtractor` with CBAM / Spatial, `ImpalaCNNExtractor`, `ImpoolaCNNExtractor`, `ResNet18Extractor`, `LSTMAttentionExtractor`, `ViTExtractor`, `VAEExtractor`, `AEExtractor`, `ReconExtractor`, `ContrastiveExtractor`).
  - Verifies autograd backward pass and ensures non-zero parameter gradients.
  - Tests compatibility across Procgen HWC $(64, 64, 3)$ and standard CHW $(3, 64, 64)$ observation formats, that `features_dim` is actually honoured, and that a non-contiguous channels-first *view* yields the same features as a contiguous copy.
  - Previously this module fell back to a `DummyPytest` shim when pytest was absent, which turned every `@pytest.mark.parametrize` into a no-op and reported green without running the cases; the shim is gone and pytest is pinned in `requirements.txt`.
- **SB3 Integration Tests (`tests/test_sb3_integration.py`):** build a `PPO("CnnPolicy", ...)` on each representative extractor and call `predict()`, over both a CHW space (what `ProcgenGymWrapper` yields) and an HWC space. This caught a real defect: every extractor decided its tensor layout from the observation *space*, so under SB3 — which already permutes HWC to CHW in `preprocess_obs` — a channels-last environment was permuted twice and crashed in the first convolution, despite the class docstrings advertising HWC support. Layout is now inferred from the tensor (`models/layout.py`), and the CHW path used by every published number is unchanged.
- **Statistical Aggregation Tests (`tests/test_rliable_metrics.py`):** pin the per-game-then-average aggregation semantics, the single bootstrap constant, profile monotonicity and probability-of-improvement antisymmetry.
- **Legacy Data Provenance Tests (`tests/test_legacy_records.py`):** assert that `results/legacy_records.json` still reproduces the published `mean`/`std`/`ci95` of section 3.8 and the section 3.7 ranking used by `retrain_analysis.py`.
- **World-Model Extractor Tests (`tests/test_world_model_extractors.py`):** `dream()` must return the input's spatial resolution, must share one encoder with `forward()`, and must react to encoder weights; the augmentation hook is only applied in training mode.
- **Intrinsic Reward Tests (`tests/test_exploration_bonuses.py`):** every bonus wrapper must add a bonus on every step, accept HWC and CHW observations, raise on a malformed observation instead of degrading to plain PPO, keep the RND target frozen, and scale linearly with `beta`.
- **Grid Analysis Tests (`tests/test_grid_analysis.py`):** `jax_port/cells_summary.json` must still regenerate the published `analysis_full.json` (every cell, ranking and metric), which is what makes the committed summary an adequate stand-in for the git-ignored raw grid logs.
- **Figure Provenance Tests (`tests/test_report_figures.py`):** the set of figures embedded in this README must equal the `PROVENANCE` map, each entry must name a script that still exists and a real data source, and the figures claimed rebuildable must have their per-seed arrays present (5 seeds each) in `results/legacy_records.json`.
- **Symbolic Regression Tests (`tests/test_symbolic_regression.py`):** eleven cases guarding section 21. The important one is not about search quality but about validity: it perturbs every measured return and asserts the feature matrix is bit-identical afterwards, which is what proves `gen_gap` (train − unseen) never leaked into the inputs. Also covered: an unmapped configuration raises instead of being silently dropped, Wilson intervals behave and widen with fewer trials, the complexity-vs-error and inversion helpers, and **a real gplearn fit** — the guard that would have caught the `scikit-learn` 1.7 incompatibility at CI time rather than twenty minutes into a study run.

- **JAX Port Test Suite (`python -m jax_port.tests.run_tests`)** — seven registered modules, each SKIP-ponly on a missing optional dependency:
  - `test_stats`: Student-t CI, Cohen's d, AUC normalization and the top-1-vs-top-2 CI-overlap predicate (both the separated and the indistinguishable case).
  - `test_parity`: old-gym Procgen vs the JAX env on observations, rewards and autoreset over 200 steps.
  - `test_zoo`: all 13 backbones forward-pass with shape and parameter-count checks.
  - `test_smoke`: end-to-end PPO training run; its arguments are built by `train.build_parser()` so a flag rename cannot leave a stale hand-built namespace behind.
  - `test_temporal`: memory-backbone shapes and `StackVec` stacking — previously orphaned, never invoked by the runner.
  - `test_marl`: `run_all()` executes **all five** cases — `MARLSequentialBuffer` temporal integrity and per-environment isolation, recurrent Q-learning Bellman target masking and gradient updates, battle metrics and win-rate accounting, adapter autoreset, and the finite-loss sweep over the seven algorithms. Before 23/09/2026 the runner registered only `test_losses`, while this section cited the other four as validation evidence.
  - `test_response` (added 24/10/2026): the port of the audit's cross-game response measure (`jax_port/response.py`, the §3.6.2/§0.1 instrument, so a port grid can carry response rows the way `audit_suite_validity.py` does). The numpy TV helper — the scorecard threshold's own arithmetic — runs without procgen and is therefore exercised in the CI port job; the full measure on a freshly built `classic` policy needs the port venv's procgen and SKIPs without it (verified in-venv on 24/10/2026, both cases green; a 20k-step `--response` smoke on `maze` reads its observation, TV `0.139`).

## 20. Legacy Continuous Control Prototypes (`legacy/carracing/`, removed 27/09/2026)

The early Phase 1 exploratory codebase developed prior to the project's transition to discrete-action Procgen benchmarks was dropped from the working tree on 27/09/2026 as out of scope for this study. It is **not** lost: every file below is readable from any commit up to `16916f2` (`git show 16916f2:legacy/carracing/sac_trainer.py`), which is where the removal records it from.
- `legacy/carracing/sac_trainer.py`: Custom continuous Soft Actor-Critic (SAC) implementation with Box action space.
- `legacy/carracing/compare_architectures.py`: Training harness benchmarking Classic CNN vs CBAM CNN on `CarRacing-v2` / `CarRacing-v3`.
- `legacy/carracing/README.md`: Architectural motivation and historical context for the transition to Procgen.

> Unrelated to the above, and still committed: `results/legacy_records.json` holds the per-seed arrays of the pre-`rliable` study and is study data, not prototype code — the name collision is unfortunate but the two must not be confused during cleanups.
