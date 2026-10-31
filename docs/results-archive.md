# Archived Results — Executed Benchmarks, Return Tables and Per-Seed Data

> Part of the study's documentation — start at the [README](../README.md). Sections 2, 3 (except 3.6) and 7. These tables rank architectures by *return* under the published single-stream recipe; §3.6.2 established that those policies do not read the screen, so they are kept for provenance, not as architecture results (the architecture result is the README's §0.1 scorecard). Section numbers are the README's original ones, so every `§x.y` reference in the code, tests and other pages still names the same section.

## 2. Executed Benchmarks

| # | Script | Game(s) | Timesteps | Seeds | Configs | Time | Log |
|---|---|---|---|---|---|---|---|
| 1 | `compare_procgen.py` | `coinrun` | `50k` | `5` | `classic/cbam/spatial/mlp_vector` | `~35 min` `5×50k` | `logs_procgen/comparison_coinrun_20260827_204023` |
| 2 | `compare_world_models.py` | `bossfight` | `100k` | `5` | `vae/ae/recon/contrastive` | `~67 min` `5×100k` | `logs_world_models/comparison_bossfight_20260827_193327` |
| 3 | `compare_suite.py` | `bossfight+starpilot+dodgeball` | `100k` | `5` | `4 WM + 4 CNN + 3 Augment = 11` per game | `~8h` `16.5M steps` `20:41→04:47` | `logs_suite/suite_bossfight_starpilot_dodgeball_20260827_204109` |
| 4 | `compare_bossfight_hard.py` | `bossfight hard` | `100k` | `5` | `11` | `~2.5h` `07:21→09:57` | `logs_bossfight_hard/comparison_bossfight_hard_20260828_072148` |
| 5 | `compare_augment_contrastive.py` | `bossfight` | `100k` | `5` | `crop/color/noise` | `~80 min` (embedded in suite) | `logs_suite` `*_aug_*` |
| 6 | `compare_new_archs.py` | `bossfight+starpilot+dodgeball` | `100k` | `5` | `impala/impoola/lstm_attention/vit/resnet18` | `~12h` `13:45→01:39` `7.5M` | `logs_new_archs/new_archs_bossfight_starpilot_dodgeball_20260828_134545` |
| 7 | `compare_maze_heist.py` | `maze+heist` | `100k` | `5` | `ppo/icm/rnd/ngu` | `~6.5h` `01:48→08:23` `4M` | `logs_maze_heist/maze_heist_maze_heist_20260829_014802` |
| 8 | `compare_combined.py` | `suite+new_archs` aggregation | — | — | global ranking of `16 architectures` | immediate (no training) | `results/global_16.png` |
| 9 | `re_eval_scorecard.py` | re-eval of `new_archs+maze_heist` | — | — | `115 zips` × `30 eps` stoch+det + gap | `~70 min` (no training) | `results/re_eval_results.json` |
| 10 | `compare_suite_retrain.py` | `bossfight+starpilot+dodgeball` | `100k` | `5` | suite retrain (`11 configs`, new protocol) | `~28h` `30/08→31/08` `16.5M` | `logs_suite_retrain/suite_retrain_zips` + `results/retrain_results.json` |
| 11 | `re_eval_100.py` | definitive re-eval of `275 zips` | — | — | `100 eps` stoch+det + gap (no retrain) | `~5h` `31/08` | `results/eval100_results.json` |
| 12 | `compare_hrl.py` *(independent)* | `jumper+plunder` | `100k frames` | `5` | `flat` vs `skip4` vs `hrl` (section 11) | `~6-12h` `31/08` | `logs_hrl/hrl_zips` + `results/hrl_results.json` |
| 13 | `compare_hrl_learned.py` *(independent)* | `jumper+plunder` | `100k frames` | `5` | `hrl_learned` arm (latent skills, section 11) | `~4-8h`, sequential to #12 | idem (`*_hrl_learned_*`, `.pt`) |
| 14 | `compare_algo_families.py` *(independent)* | `starpilot+dodgeball+bossfight` | `100k` | `5` | `ppo`/`a2c` (policy) vs `dqn`/`qrdqn` (value), section 12 | `~9h` `01/09→02/09` | `logs_algo/algo_zips` + `results/algo_families_results.json` |

## 3. Official Results (5 Seeds, 10 Eps Eval)

### 3.1. Coinrun 50k — CNN vs MLP (Same Game With/Without CV)
`logs_procgen/comparison_coinrun_20260827_204023/statistics.json:1`
| Config | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|
| `classic_pixels` | **7.6** | 1.2 | 6.0 | 9.0 |
| `attention_cbam_pixels` | **8.0** | 0.0 | 8.0 | 8.0 |
| `attention_spatial_pixels` | 6.2 | 3.31 | 0.0 | 9.0 |
| `mlp_vector` (`16×16` `256D` without CV) | **8.0** | 0.89 | 7.0 | 9.0 |
> **Analysis:** Stable `CBAM 8.0` outperforms `classic 7.6`; `MLP 8.0` matches the best `CNN` — `coinrun easy 200` is purely reactive and does not require complex spatial vision (`downsample 256D` suffices). Pure `Spatial` attention proved unstable (`0.0` on 1 seed) without a residual connection.

### 3.2. Bossfight 100k — World Models
`logs_world_models/comparison_bossfight_20260827_193327/statistics.json:1`
| Config | Mean | Std |
|---|---:|---:|
| `vae` | 0.16 | 0.16 |
| `ae` | 0.30 | 0.50 |
| `recon` | 0.02 | 0.04 |
| `contrastive` | **0.36** | 0.62 |
> `contrastive` performs best, but all remain `<0.5` — `bossfight 100k` is insufficient when trained standalone.
>
> ⚠️ **Methodological & Architectural Clarification (Gradient Flow & Auxiliary Pretext Objectives):**
> In the root PyTorch / SB3 benchmark (`compare_world_models.py`), the modules in `models/world_model_extractors.py` served as custom `BaseFeaturesExtractor` architectures within standard Stable-Baselines3 PPO. During `model.learn()`, SB3 backpropagates gradients exclusively from the PPO policy and value heads through `forward()`. Because no custom policy was attached to compute auxiliary reconstruction loss (MSE) or KL divergence during the backward pass, **the decoders (`fc_dec`, `deconv1-3`, and `dream()`) never received gradients and remained at random initialization**. Consequently, the scores above reflect the inductive bias of an information/sampling bottleneck in the visual encoder rather than learned generative world modeling. Fully end-to-end recurrent dynamics modeling, latent imagination rollouts, and active reconstruction gradients are implemented in `jax_port/train_dreamer.py`.

### 3.3. Suite 100k — 3 Games (11 Configs/Game)
`logs_suite/suite_bossfight_starpilot_dodgeball_20260827_204109/suite_statistics.json:1`
| Game | Best | Mean | 2nd | Worst |
|---|---|---:|---|---|
| `bossfight` (`~0.5`) | `spatial 0.76±0.90` | `cbam 0.58`, `classic 0.54` | `contrastive 0.36` | `recon 0.02` |
| `starpilot` (`~2.0`) | `spatial 2.6±0.82` | `aug_crop 2.12±0.79`, `cbam 2.1`, `ae 2.08` | `color 1.94`, `noise 1.61` | — |
| `dodgeball` (`~1.2`) | `classic 1.48±0.69` | `cbam 1.31`, `spatial 1.28` | `vae 1.2` | `recon 0.88` |
> **Augment:** `crop` outperforms `color`/`noise` in `starpilot` (`+31%`) and `dodgeball`; `color` takes second in `bossfight`. **Overall:** `classic` wins `dodgeball`, `spatial` wins `starpilot` — **rankings invert across games**; a single game introduces significant inductive bias (the `Procgen` standard is `16` games; `3` is the bare minimum).

### 3.4. Bossfight HARD 100k — Stress Test
`logs_bossfight_hard/comparison_bossfight_hard_20260828_072148/statistics.json:1`
| Config | Mean | Std |
|---|---:|---:|
| `vae` | **0.43±0.54** | `ae 0.38`, `aug_crop 0.32`, `mlp 0.26` |
| `cnn` | `0.02±0.04` (`classic/cbam/spatial` collapse to zero) |
> `hard` compresses all returns to `0.0-0.4` (compared to `0.5-0.76` in `easy`); `CNN` collapses in `100k` — `hard` would require `200k` (`~6h`) and does not warrant a full `suite hard`; `easy` constitutes a fairer benchmark.

### 3.5. New Archs 100k — 5 New Architectures × 3 Games
`logs_new_archs/new_archs_bossfight_starpilot_dodgeball_20260828_134545/statistics.json:1`
| Game | Best | Mean | 2nd | Worst |
|---|---|---:|---|---|
| `bossfight` | `lstm 0.36±0.57` | `vit 0.30`, `impala 0.28` | `resnet 0.02` | `impoola 0.06` |
| `starpilot` | `lstm 2.44±0.56` | `resnet 2.28`, `impoola 2.2` | `vit 2.1`, `impala 1.78` | — |
| `dodgeball` | `resnet 1.72±0.65` | `vit 1.2`, `impoola 1.12` | `impala 1.08` | `lstm 0.80` |
> `ViT`/`ResNet` fail to surpass `spatial 2.6` in `starpilot` or `classic 1.48` in `dodgeball`; `lstm` wins `bossfight`/`starpilot` but underperforms in `dodgeball`.

> §3.6–§3.6.13 (the exploration arms and the diagnostic arc) live in [diagnostic-arc.md](diagnostic-arc.md).

### 3.7. Global 16 Architectures — 3-Game Average (Top 10 of 16 Displayed)

> ⚠ **Validity (11/10/2026; resolved 14/10/2026):** all `16` arms of this comparison were re-trained on each of its three games at the published budget, with the response measure attached (`48` cells, §3.6.2, `results/suite_validity_{starpilot,bossfight,dodgeball}.json`). Every arm that finished emits one argmax for five different games. Read as a ranking of visual front ends this table is **unverified**; read as a record of what the archived runs returned, it stands. **The re-run under the responsive recipe exists now** (§3.6.7, `240` cells, `results/suite_validity_nenv16_grid.json`): the ordering changes shape — the archived top-3 lands 12th–14th — and return still does not track whether the front end is read; the response table in §3.6.7 is the architecture result this table never was.

`logs_suite` + `logs_new_archs` aggregated (`16.5M + 7.5M` steps) — `mean` of `3` `means` per architecture:
| Rank | Architecture | Global Mean | Per-Game Breakdown (B/S/D) |
|---:|---|---:|---|
| 1 | `spatial` | **1.54** | 0.76 / 2.60 / 1.28 |
| 2 | `resnet18` | 1.34 | 0.02 / 2.28 / 1.72 |
| 3 | `classic` | 1.33 | 0.54 / 1.98 / 1.48 |
| 4 | `cbam` | 1.33 | 0.58 / 2.10 / 1.31 |
| 5 | `mlp_vector` | 1.25 | 0.58 / 2.07 / 1.11 |
| 6 | `lstm_attention` | 1.20 | 0.36 / 2.44 / 0.80 |
| 7 | `vit` | 1.20 | 0.30 / 2.10 / 1.20 |
| 8 | `aug_crop` | 1.16 | 0.54 / 2.12 / 0.84 |
| 9 | `impoola` | 1.12 | 0.06 / 2.20 / 1.12 |
| 10 | `ae` | 1.11 | 0.30 / 2.08 / 0.96 |
> The `Top 3` are pure `CNN` variants (`spatial`/`resnet`/`classic`); `World Models` (`vae 0.88`, `recon 0.90`) and `contrastive 0.97` rank below `MLP 1.25` in `suite 100k` `easy` — `CV` with attention still outperforms `World Models` in `Procgen` at `100k`.

### 3.8. Statistical Robustness — 95% CI + Effect Size (`scorecard_analysis.py`, `results/scorecard.json`)

With `n=5 seeds`, `95% CI` is computed via Student's `t` (`df=4`, critical value `2.776`); `Cohen's d` compares top-1 vs top-2 per game:
> **Data provenance:** the per-seed returns for the runs whose raw logs were pruned (`coinrun 50k`, `bossfight 100k` world models) are now read from `results/legacy_records.json`, the single machine-readable record of those runs — they used to be literals pasted inside `scorecard_analysis.py`. `tests/test_legacy_records.py` asserts that those arrays still reproduce the `mean`/`std`/`ci95` published below, and the same file supplies the section 3.7 ranking used by `retrain_analysis.py`.

| Game | Top-1 vs Top-2 | Cohen's d | CIs Overlap? | Conclusion |
|---|---|---:|---|---|
| `bossfight` (WM+New) | `lstm 0.36` vs `contrastive 0.36` | 0.0 | ✅ Yes | Statistical tie |
| `starpilot` | `lstm 2.44` `[1.66, 3.22]` vs `resnet 2.28` `[1.37, 3.19]` | 0.235 (small) | ✅ Yes | **Not statistically significant** |
| `dodgeball` | `resnet 1.72` `[0.81, 2.63]` vs `vit 1.2` `[0.90, 1.50]` | 0.956 (large) | ✅ Yes | Large effect size, but `n=5` underpowered |
| `maze` | `ppo 2.4` `[0.32, 4.48]` vs `icm 1.8` `[-0.24, 3.84]` | 0.36 (small) | ✅ Yes | ICM "worse" **not confirmed** |
| `heist` | `ppo 0.8` vs `icm 0.6` | 0.23 (small) | ✅ Yes | Statistical tie |
| `coinrun` | `cbam 8.0` `[8.0, 8.0]` vs `mlp 8.0` `[6.76, 9.24]` | 0.0 | — | Tie; `cbam` exhibits zero variance |
> **Critical takeaway:** No top-1 vs top-2 difference is statistically significant with `5 seeds` — the rankings in sections 3.x represent **empirical trends**, not definitive conclusions. Only `dodgeball resnet vs vit` (`d=0.956`) approaches a reliable effect. The exact ties between `RND`/`NGU` and `PPO` (`d=0`, identical per-seed returns) were not a null result — they were the broken wrapper of section 3.6, confirmed on 23/09/2026.
> ⚠️ **Superseded:** This ranking was measured under the legacy protocol (`10 eps`); retraining the suite under the new protocol (section 3.11) **inverts the top-2** (`mlp_vector` surpasses `spatial`).

### 3.9. Sample Efficiency (AUC) — Partial Scorecard (`rollout/ep_rew_mean` TensorBoard, 5 Seeds)

`AUC_norm = ∫reward·dsteps / 100k` (mean of 5 seeds) — computed exclusively for `new_archs`/`maze_heist` (legacy TensorBoard logs pruned):

| Game | Best AUC | Ranking by AUC | vs Final Evaluation Ranking |
|---|---|---|---|
| `bossfight` | `lstm 0.247` | `lstm > impoola 0.186 > impala 0.141 > resnet 0.107 > vit 0.064` | ≈ Identical to final ranking |
| `starpilot` | `impala 2.336` | `impala > resnet 2.325 > vit 2.297 > lstm 2.284 > impoola 2.23` | **Inverted**: `impala` has best AUC but worst final return (`1.78`) — learns fast, plateaus early |
| `dodgeball` | `resnet 1.175` | `resnet > impala 1.169 > impoola 1.151 > lstm 1.114 > vit 1.032` | ≈ Identical to final ranking |
| `maze` | `ppo=rnd=ngu 3.785` | all > `icm 3.658` | ICM trails in AUC as well (not only at convergence) |
| `heist` | `ppo=rnd=ngu 1.826` | all > `icm 1.792` | Identical pattern |
> `rnd`/`ngu` rows in this table are the disabled-wrapper arms of section 3.6, not curiosity results.
> **Finding:** `starpilot_impala` displays the highest learning rate curve but lowest final evaluation reward — proving that "fastest learner" ≠ "best asymptotic model" (the ⚡ AUC axis is orthogonal to the 🧠 performance axis).

### 3.10. Extended Re-Evaluation — `re_eval_scorecard.py` (Completed: `115/115`, `0 errors`)

`115 models` (`new_archs 75` + `maze_heist 40`) re-evaluated across `30 eps unseen` (stoch + det, levels `seed+1000`) and `15 eps` training levels (generalization gap). Full data in `results/re_eval_results.json`:
> ❓ **Why do absolute values differ from sections 3.4/3.6 when weights are identical?** Two independent causes: (a) **Different level distribution** — the new protocol samples unseen levels via `seed+1000` (original used an alternative seed offset), shifting sampled level difficulty; (b) **30 vs 10 episodes** — 10-episode averages had high sample variance. Hence, **only relative ranking within the same evaluation set is scientifically valid** ("vs 3.4/3.6" column), never absolute comparisons across protocols.

| Config | stoch unseen | det unseen | gen gap | vs 3.4/3.6 (10 eps) |
|---|---:|---:|---:|---|
| `bossfight_resnet18` | **0.39** | 0.03 | −0.12 | 4th → **1st** |
| `bossfight_impala` | 0.33 | 0.43 | −0.10 | Improves |
| `bossfight_lstm_attention` | 0.32 | 0.36 | −0.20 | 1st → 3rd |
| `bossfight_vit` | 0.29 | 0.16 | −0.27 | Stable |
| `bossfight_impoola` | 0.25 | 0.43 | −0.21 | Drops to last |
| `starpilot_lstm_attention` | **2.63** | 1.25 | +0.15 | Retains top-1 |
| `starpilot_resnet18` | 2.53 | 0.91 | −0.51 | Stable |
| `starpilot_impoola` | 2.44 | 1.29 | −0.19 | Stable |
| `starpilot_impala` | 2.00 | 1.29 | +0.31 | 5th → 4th |
| `starpilot_vit` | 1.88 | 0.41 | −0.07 | Drops to last |
| `dodgeball_resnet18` | **1.07** | 0.89 | +0.27 | Retains top-1 |
| `dodgeball_impoola` | 1.05 | 0.43 | −0.07 | Improves |
| `dodgeball_impala` | 1.04 | 0.44 | −0.11 | Improves |
| `dodgeball_lstm_attention` | 0.99 | 0.76 | +0.24 | 5th → 4th |
| `dodgeball_vit` | 0.91 | 0.40 | −0.21 | **2nd → 5th** |
| `maze_icm` | **2.80** | 0.47 | +0.80 | **3rd → 1st** |
| `maze_ppo`=`rnd`=`ngu` | 2.47 | 0.60 | +0.73 | Top-1 → tied 2nd |
| `heist_icm` | **0.73** | 0.33 | −0.47 | **2nd → 1st** |
| `heist_ppo`=`rnd`=`ngu` | 0.67 | 0.07 | 0.00 | Top-1 → tied 2nd |
> **3 Findings:** (1) **`ICM` leads in `maze`/`heist`** — inverting section 3.6; at `30 eps`, intrinsic curiosity bonuses become visible. (2) **`vit` fails to hold 2nd place in `dodgeball`** (`1.20` → `0.91`) — confirming section 3.8 (`n=5` underpowered). (3) **Deterministic evaluation collapses exploration policies** (`maze` `2.47 stoch` → `0.60 det`), while **gen gap is ≈ 0/negative** across most models — models do not overfit/memorize the `200` training levels; they simply remain sample-constrained (exceptions: `maze`/`heist`/`dodgeball resnet`, gap `+0.7~+0.8`).

### 3.11. Retraining the Suite under New Protocol — `compare_suite_retrain.py` (Completed: `160/165`; `5 NaN` in `wm_vae` Persisted Across 2 Retry Rounds)

The `165 models` of the original suite (`4 WM + 4 CNN + 3 augment × 3 games × 5 seeds`, hyperparameters identical to `compare_suite.py:26`) were **retrained from scratch** with the new protocol (`30 eps` stoch+det + gap, weights stored in `logs_suite_retrain/suite_retrain_zips`). Data in `results/retrain_results.json`, analysis in `results/retrain_analysis.json` (`retrain_analysis.py`):

> ❓ **Why do figures differ from the original suite (sections 3.3/3.7)?** Three independent factors: (a) unseen level evaluation seed offset (`seed+1000`), (b) `30 vs 10` episodes, and (c) **new model weights** — newly instantiated training runs subject to `CUDA`/`cuDNN` non-determinism (seen in the `5` divergent seeds of `wm_vae`). Disentanglement example: in `starpilot`, `spatial` scored `2.63` ≈ legacy `2.60` (difference ~0 → factors (a)+(b)+(c) negligible here), whereas in `bossfight` it dropped from `0.76 → 0.36` — since the protocol is identical across all configs, changes in *relative ranking* reflect genuine evaluation fidelity rather than random artifacts.

**Top 5 per game (stochastic unseen, 30 eps):**

| Game | 1st | 2nd | 3rd | 4th | 5th |
|---|---|---|---|---|---|
| `bossfight` | `aug_crop 0.68` | `contrastive 0.48` = `aug_noise 0.48` | `cbam 0.40` | `mlp 0.37` | `spatial 0.36` |
| `starpilot` | `spatial 2.63` | `mlp 2.49` | `classic 2.20` = `aug_crop 2.20` | `wm_ae 2.17` | `cbam 2.16` |
| `dodgeball` | `mlp 1.21` | `cbam 1.15` | `classic 1.07` = `spatial 1.07` = `aug_color 1.07` | `wm_ae 1.05` | `aug_crop 1.04` |

**New Global Ranking** (mean across 3 games, identical aggregation rule to section 3.7) **vs Legacy:**

| Rank | Architecture | New | Legacy (3.7) | Δ |
|---:|---|---:|---:|---:|
| 1 | `mlp_vector` | **1.36** | 1.25 (5th) | +0.11, **5th→1st** |
| 2 | `spatial` | 1.35 | **1.54 (1st)** | −0.19, **1st→2nd** |
| 3 | `aug_crop` | 1.31 | 1.16 | +0.15 |
| 4 | `cbam` | 1.24 | 1.33 (4th) | −0.09 |
| 5 | `classic` | 1.16 | 1.33 (3rd) | −0.17, **3rd→5th** |
| 6 | `wm_ae` | 1.13 | 1.11 | +0.02 |
| 7 | `wm_recon` | 1.05 | ~0.90 | Improves |
| 8 | `aug_noise` | 1.04 | — | — |
| 8 | `wm_contrastive` | 1.04 | ~0.97 | Improves |
| 10 | `aug_color` | 1.02 | — | — |
| 11 | `wm_vae` | 0.92 | — | n=2–3 (NaN) |

> **4 Findings:** (1) **`mlp_vector` dethrones `spatial`** — the shift stems from `bossfight` (`spatial` `0.76→0.36`), whereas `spatial` holds in `starpilot` (`2.63` ≈ legacy `2.60`) and `dodgeball`. (2) **World Models are not universally deficient**: weakness was concentrated in `bossfight`; in `starpilot` (`wm_ae 2.17`, `recon 1.97`) and `dodgeball` (`wm_ae 1.05`) they tie with standard `CNN`s. (3) **`aug_crop` is the superior data augmentation** (`bossfight 0.68`, game winner). (4) **`dodgeball` yields poor architectural separation** (`spread 1.21→0.88` vs `0.68→0.12` in `bossfight`) — low resolution benchmark. Combining re-evaluated `new_archs`/`maze_heist`, the **unified global ranking** is: `mlp_vector 1.36` > `spatial 1.35` > `lstm_attention 1.34` > `aug_crop 1.31` (all statistically indistinguishable per section 3.8). Total: `275 models` evaluated under the new protocol.
> ⚠️ **Update:** The figures above reflect `30 eps`; definitive `100 eps` evaluation (section 3.12) shifts the top-1 in `starpilot`/`dodgeball` and dissolves `ICM`'s advantage.

### 3.12. Definitive Protocol — `100 eps` Across All `275` Models (`re_eval_100.py`, Completed: `275/275`)

Re-evaluation of identical checkpoints across `100 eps unseen` stochastic + `100 eps` deterministic + `15 eps` training (no retraining). Data in `results/eval100_results.json`, comparative analysis `30 vs 100` in `results/eval100_analysis.json` (`eval100_analysis.py`):

**Top Configurations per Game @ 100 eps:**

| Game | 1st | 2nd | 3rd | vs @ 30 eps |
|---|---|---|---|---|
| `bossfight` | `aug_crop 0.68` | `aug_noise 0.54` = `contrastive 0.54` | `mlp 0.53` | Top-1 **stable** |
| `starpilot` | `mlp 2.67` | `spatial 2.45` | `resnet18 2.44` | `lstm` 1st → 4th (`2.43`) |
| `dodgeball` | `resnet18 1.16` | `cbam 1.08` = `mlp 1.08` | `wm_recon 1.02` | `mlp` 1st → 3rd; `recon` 12th → 4th |
| `maze` | `ngu 2.76` | `ppo 2.58` | `rnd 2.54` | `icm 2.36` |
| `heist` | `rnd 1.04` | `ngu 0.80` | `ppo 0.72` | `icm 0.60` |

**Global Suite Ranking @ 100 eps:** `mlp_vector 1.43` > `resnet18 1.34` > `spatial 1.28` > `lstm_attention 1.26` > `aug_crop 1.25`.

> **4 Findings from the 30→100 Upgrade:** (1) **`mlp_vector` consolidates global 1st place** (`1.25`@10eps → `1.36`@30 → `1.43`@100 — the only model consistently at the top across all protocols). (2) **Exploration arms re-measured (24/09/2026)**: In `maze`, `ngu` achieves `2.76`, `ppo` `2.58`, `rnd` `2.54`, and `icm` `2.36`; in `heist`, `rnd` achieves `1.04`, `ngu` `0.80`, `ppo` `0.72`, and `icm` `0.60`. All differences remain within overlapping 95% confidence intervals, confirming that while wrappers execute correctly, unnormalized intrinsic error ($10^{-7}$/step) is insufficient to fundamentally reshape policy exploration in Procgen. (3) **Top-1 in `starpilot`/`dodgeball` inverts again** (`lstm→mlp`, `mlp→resnet18`) while `bossfight` remains stable — the upper tier is solid, while mid-tier rankings remain fluid. (4) **Mean absolute delta |Δ| is `0.108`** between @30 and @100: variance converges, and residual position shifts reinforce section 3.8: with `5 seeds`, no strict ranking order is mathematically frozen; the study reports credible trends with confidence intervals.

### 3.13. Budget Scaling — `compare_budget_scaling.py` (Completed: `24/24`, `0 errors`)

> **Validity note (14/10/2026):** these runs trained on the single-stream recipe — the arms here are the state-independent policies of §3.6.2, so the plateau below is a plateau of *blind* policies. The same question under the responsive recipe is §3.6.8: there, `2.5×` budget changes return only by seed noise for the arms that read, regresses the `100k` seed-luck champion, and *deepens* the response measure instead.

Roadmap item #6: Does the advantage of top models persist under larger budgets? Scope: `resnet18` + `mlp_vector` × `starpilot` + `dodgeball` × `3 seeds` (42-44) × `250k`/`500k`; the `100k` baseline is established (definitive evaluation, identical seeds). Data in `results/budget_results.json`, analysis in `results/budget_analysis.json` (`budget_analysis.py`):

| Curve (stochastic unseen) | 100k | 250k | 500k | Verdict |
|---|---:|---:|---:|---|
| `starpilot_resnet18` | 2.30±0.58 | 2.73±0.27 | 2.52±0.49 | **Stagnant** |
| `starpilot_mlp_vector` | 2.67±0.23 | 3.23±0.55 | 2.64±0.41 | **Stagnant** |
| `dodgeball_resnet18` | 1.07±0.38 | 1.01±0.15 | 1.00±0.07 | **Stagnant** |
| `dodgeball_mlp_vector` | 1.17±0.35 | 0.91±0.15 | 0.91±0.15 | **Stagnant** |

> **3 Findings:** (1) **No learning curve scales monotonically with budget** — `250k` represents a minor inflection within empirical noise, and `500k` regresses or ties; a `5×` budget neither created nor eliminated an advantage. Answer to item #6: **larger budgets were not required** to differentiate these architectures; `100k` proved sufficient. (2) **Per-game relative advantages persist**: `mlp_vector` leads `starpilot` at `250k` (`3.23` vs `2.73`) and `resnet18` leads `dodgeball` across both budgets (`1.01/1.00` vs `0.91`) — confirming section 3.12. (3) **Generalization gap remains ≈ `0`/negative even at `500k`** — overfitting/memorization does not emerge with extended budgets, reinforcing section 3.10.

### 3.14. Robust Statistical Evaluation — NeurIPS 2021 RLiable Metrics (`rliable_metrics.py`, `run_rliable_eval.py`)

To resolve the statistical limitations of raw point estimates (sample means and standard deviations) across heterogeneous Procgen environments, we implemented the evaluation methodology proposed by **Agarwal et al. (NeurIPS 2021)** (*"Deep Reinforcement Learning at the Edge of the Statistical Precipice"*).

Using the definitive 100-episode evaluation dataset across all 275 models (`results/eval100_results.json`), scores are Min-Max normalized per game using empirical extremal bounds:
- `bossfight`: $[0.00, 1.51]$
- `dodgeball`: $[0.24, 1.78]$
- `heist`: $[0.10, 1.40]$
- `maze`: $[0.80, 3.80]$
- `starpilot`: $[0.28, 3.10]$

For each architecture evaluated across the 3 core suite games (`bossfight`, `starpilot`, `dodgeball`), we computed the **Interquartile Mean (IQM)**, **5% Trimmed Mean**, and **95% Stratified Bootstrap Confidence Intervals** (with $B = 10{,}000$ resamples), along with **Performance Profiles** ($\tau \in [0, 1]$) and **Pairwise Probability of Improvement**:

> **Aggregation correction (23/09/2026):** the aggregator previously applied the metric **once to the pooled array** of all games' scores, which lets any game with more seeds dominate the estimate. It now follows Agarwal et al. literally — the metric is computed **within each game** and then **averaged across games** (equal weight per game), with resampling stratified inside each game. Because the three suite games have equal seed counts, the point estimates shift only slightly, but the confidence intervals widen (they are no longer computed on a 45-element pooled sample), and the IQM ordering of `resnet18` and `cnn_mlp_vector` inverts. The single bootstrap constant is now `rliable_metrics.NUM_BOOTSTRAPS = 10000`; the forest and profile panels previously ran at $B = 2{,}000$.

| Architecture | Normalized IQM | 95% Bootstrap CI | 5% Trimmed Mean | 95% Trimmed CI |
|---|---:|:---:|---:|:---:|
| `resnet18` | **0.532** | [0.368, 0.736] | 0.546 | [0.410, 0.687] |
| `cnn_mlp_vector` | 0.529 | [0.420, 0.733] | **0.581** | [0.462, 0.710] |
| `cnn_cbam` | 0.520 | [0.363, 0.646] | 0.502 | [0.394, 0.607] |
| `aug_crop` | 0.494 | [0.374, 0.654] | 0.513 | [0.398, 0.632] |
| `cnn_spatial` | 0.488 | [0.383, 0.632] | 0.505 | [0.411, 0.605] |
| `lstm_attention` | 0.483 | [0.385, 0.635] | 0.499 | [0.406, 0.602] |
| `wm_vae` | 0.440 | [0.322, 0.574] | 0.443 | [0.324, 0.572] |
| `aug_noise` | 0.434 | [0.253, 0.543] | 0.395 | [0.273, 0.508] |
| `wm_contrastive` | 0.434 | [0.253, 0.543] | 0.395 | [0.273, 0.508] |
| `impala` | 0.419 | [0.311, 0.536] | 0.426 | [0.335, 0.516] |
| `impoola` | 0.413 | [0.319, 0.616] | 0.453 | [0.345, 0.583] |
| `cnn_classic` | 0.411 | [0.344, 0.522] | 0.432 | [0.362, 0.508] |
| `wm_recon` | 0.407 | [0.276, 0.598] | 0.438 | [0.320, 0.573] |
| `aug_color` | 0.399 | [0.255, 0.493] | 0.381 | [0.283, 0.467] |
| `wm_ae` | 0.374 | [0.311, 0.452] | 0.381 | [0.332, 0.433] |
| `vit` | 0.326 | [0.229, 0.538] | 0.372 | [0.257, 0.512] |

> **Key Robust Statistical Findings:**
> 1. **Overlapping 95% Confidence Intervals Across Top Tier:** The stratified bootstrap 95% CIs for the top 6 architectures (`resnet18` $[0.368, 0.736]$, `cnn_mlp_vector` $[0.420, 0.733]$, `cnn_cbam` $[0.363, 0.646]$, `aug_crop` $[0.374, 0.654]$, `cnn_spatial` $[0.383, 0.632]$, `lstm_attention` $[0.385, 0.635]$) substantially overlap. Under NeurIPS 2021 statistical guidelines, ranking these models by point IQM estimate alone is statistically unwarranted. The two leading architectures also disagree across aggregators (`resnet18` wins IQM, `cnn_mlp_vector` wins trimmed mean and unnormalized mean) — further evidence that the top tier is a single statistical cluster.
> 2. **Pairwise Probability of Improvement:** Agarwal et al.'s probability of improvement $P(X > Y)$ computes the probability that a randomly selected seed/run of architecture $X$ yields higher normalized return than architecture $Y$:
>    - $P(\text{mlp\_vector} > \text{resnet18}) = 0.587$ (95% CI: $[0.360, 0.800]$): Because the 95% CI spans $0.50$, neither model statistically dominates the other.
>    - $P(\text{mlp\_vector} > \text{cnn\_spatial}) = 0.640$ (95% CI: $[0.427, 0.840]$): also non-significant.
>    - $P(\text{mlp\_vector} > \text{aug\_crop}) = 0.633$ (95% CI: $[0.420, 0.827]$).
>    - $P(\text{resnet18} > \text{cnn\_cbam}) = 0.500$ (95% CI: $[0.267, 0.733]$): Exact parity.
>    Every pairwise comparison among the top 5 falls inside $[0.36, 0.64]$ with CIs crossing $0.5$ — no architecture in this benchmark is demonstrably superior to another.
> 3. **Performance Profiles:** The cumulative performance profile (see Figure 4.9, `results/rliable_profile.png`) demonstrates that `cnn_mlp_vector` and `resnet18` dominate higher normalized return thresholds ($\tau > 0.6$), whereas `lstm_attention`, `cnn_spatial`, and `cnn_cbam` show higher probability mass at moderate thresholds ($\tau \approx 0.5$). Data serialized in `results/rliable_scorecard.json`.
> 4. **Canonical (Agarwal) normalization covers only 2 of the 3 suite games:** `dodgeball` is excluded because the reference architecture `cnn_classic` does not beat the empirical random policy there (see section 18.1). `maze` and `heist` are excluded because no `cnn_classic` arm was ever trained on them. A previously published `classic_means` value of `1.0` for those two games was a silent code fallback, not a measurement; it has been removed.
   **Updated (24/10/2026):** `run_rliable_eval.py` now also computes a *best-anchored* canonical normalization — the same Agarwal formula with the best measured policy of this dataset as the per-game ceiling (`results/rliable_scorecard.json` → `best_anchored_architectures`), stated as a dataset-relative ceiling and never a human one: a normalized score above `1` means "beat the best measured arm of this study". It covers **all 3 suite games** — `dodgeball` enters with `resnet18` at `1.156` as its ceiling (against the random anchor's `0.960`) — while `maze`/`heist` remain excluded under a sharper reason: no measured architecture beats the random anchor there at all (`maze` best `ngu 2.760` vs random `4.400`; `heist` best `rnd 1.040` vs `2.800`), which is §18.1's finding expressed as an eligibility rule. The classic-anchored block is unchanged beside it.

## 7. Results per Seed (Complete) — 5 Seeds `42-46`, 10 Eps `deterministic=False`

**Coinrun 50k 5 Seeds** `logs_procgen/comparison_coinrun_20260827_204023/comparison_results.json:1`
| Config | 42 | 43 | 44 | 45 | 46 | Mean±Std |
|---|---:|---:|---:|---:|---:|---|
| `classic` | 7.0 | 6.0 | 7.0 | 9.0 | 9.0 | **7.6±1.2** |
| `cbam` | 8.0 | 8.0 | 8.0 | 8.0 | 8.0 | **8.0±0.0** |
| `spatial` | 6.0 | 0.0 | 7.0 | 9.0 | 9.0 | **6.2±3.31** |
| `mlp_vector` | 7.0 | 7.0 | 8.0 | 9.0 | 9.0 | **8.0±0.89** |

**Bossfight 100k 5 Seeds (World Models)** `logs_world_models/comparison_bossfight_20260827_193327/comparison_results.json:1`
| Config | 42 | 43 | 44 | 45 | 46 | Mean±Std |
|---|---:|---:|---:|---:|---:|---|
| `vae` | 0.0 | 0.0 | 0.1 | 0.4 | 0.4 | 0.16±0.16 |
| `ae` | 0.0 | 0.0 | 0.1 | 0.1 | 1.3 | 0.30±0.50 |
| `recon` | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 | 0.02±0.04 |
| `contrastive` | 0.0 | 0.0 | 0.0 | 0.2 | 1.6 | **0.36±0.62** |

**Bossfight HARD 100k 5 Seeds** `logs_bossfight_hard/comparison_bossfight_hard_20260828_072148/comparison_results.json:1`
| Config | 42 | 43 | 44 | 45 | 46 | Mean±Std |
|---|---:|---:|---:|---:|---:|---|
| `vae` | 0.0 | 0.0 | 0.4 | 1.2 | 0.6 | **0.43±0.54** |
| `ae` | 0.1 | 0.1 | 0.4 | 0.1 | 1.2 | 0.38±0.41 |
| `recon` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0±0.0 |
| `contrastive` | 0.0 | 0.0 | 0.0 | 0.0 | 0.3 | 0.06±0.12 |
| `classic` | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 | 0.02±0.04 |
| `cbam` | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 | 0.02±0.04 |
| `spatial` | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 | 0.02±0.04 |
| `mlp` | 0.0 | 0.0 | 0.0 | 0.1 | 1.2 | 0.26±0.47 |
| `aug_crop` | 0.0 | 0.0 | 0.0 | 0.3 | 1.3 | 0.32±0.49 |
| `aug_color` | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 | 0.02±0.04 |
| `aug_noise` | 0.0 | 0.0 | 0.0 | 0.0 | 0.3 | 0.06±0.12 |

**Suite 100k 3 Games (11 configs × 3 × 5 = 165 entries)** `logs_suite/suite_bossfight_starpilot_dodgeball_20260827_204109/suite_results.json:1` — summary `mean` in `suite_statistics.json:1`; e.g., `starpilot spatial: [1.6, 2.1, 2.2, 3.2, 3.5] 2.6±0.82`, `dodgeball classic: [0.8, 1.0, 1.4, 1.5, 2.7] 1.48±0.69`. Full `165` raw entries preserved in `suite_results.json` for LaTeX compilation.

**New Archs 100k 3 Games (5 × 3 × 5 = 75 entries)** `logs_new_archs/new_archs_bossfight_starpilot_dodgeball_20260828_134545/comparison_results.json:1`
| Config | 42 | 43 | 44 | 45 | 46 | Mean±Std |
|---|---:|---:|---:|---:|---:|---|
| `bossfight_impala` | 0.0 | 0.2 | 0.0 | 0.0 | 1.2 | 0.28±0.47 |
| `starpilot_lstm_attention` | 2.3 | 2.9 | 1.4 | 2.7 | 2.9 | **2.44±0.56** |
| `dodgeball_resnet18` | 1.2 | 1.6 | 1.4 | 3.0 | 1.4 | **1.72±0.65** |

**Maze+Heist 100k 2 Games (4 × 2 × 5 = 40 entries)** `results/exploration_remeasure.json:1` (re-measured 24/09/2026, 10 eps unseen)
| Config | 42 | 43 | 44 | 45 | 46 | Mean±Std |
|---|---:|---:|---:|---:|---:|---|
| `maze_ppo` | 2.0 | 2.0 | 1.0 | 1.0 | 3.0 | 1.80±0.84 |
| `maze_icm` | 3.0 | 2.0 | 0.0 | 1.0 | 5.0 | 2.20±1.92 |
| `maze_rnd` | 2.0 | 4.0 | 0.0 | 0.0 | 6.0 | **2.40±2.61** |
| `maze_ngu` | 2.0 | 2.0 | 0.0 | 1.0 | 6.0 | 2.20±2.28 |
| `heist_ppo` | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 0.80±0.45 |
| `heist_icm` | 1.0 | 0.0 | 2.0 | 0.0 | 1.0 | 0.80±0.84 |
| `heist_rnd` | 2.0 | 1.0 | 3.0 | 0.0 | 2.0 | **1.60±1.14** |
| `heist_ngu` | 1.0 | 0.0 | 3.0 | 0.0 | 2.0 | 1.20±1.30 |
