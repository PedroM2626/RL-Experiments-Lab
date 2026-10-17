# The Collapse of Observation-Conditioning in a Procgen Benchmark: Cause, Threshold, Anatomy

A distilled narrative of README §3.6.2–§3.6.13 — the dense-probe arc, 10–14/10/2026 — written to be read start to finish in about ten minutes. The full tables, provenance and operational notes live in the README sections cited; every number here is carried by a committed artifact and recomputed by a test (`tests/test_report_md.py`, plus the section tests it points at). Total cost of the arc: **~56 GPU-hours** across ~220 runs, of which the 500k budget round is 23.

---

## 1. The finding

The study's trained policies ignored their observations. Of `69/70` committed checkpoints, every one emits the same argmax action for the initial frames of five *different* Procgen games (median cross-game TV `4.8e-4` — §3.6.2). A re-run of the published recipe confirmed it, and the audit extended the check to the headline comparison itself: `47/48` cells of the 16-architecture grid trained blind (§3.6.2). Whatever those tables measured, it was not the visual front ends.

## 2. The cause: what one update sees

The recipe trained on a **single environment** — a `DummyVecEnv` of one. A dense probe (held-out frames, every 500 steps, §3.6.4) put the collapse on a timeline: the published arm's response dies at `9.5k` steps (seed 42; anywhere in `9k–36k` across five seeds), while the same recipe with `16` parallel envs — the *same per-update batch*, more distinct game levels per update — **never dips** (`101/101` probes alive, §3.6.5). The operative quantity is per-update **coverage**: transitions and distinct levels together. Three levers settled it at five seeds each — changing what one *step* sees (`action_repeat=4`: dead `4/5`) and pressing the objective's entropy term (`ent_coef=0.01`: dead `5/5`, the marginal held entropic throughout) do nothing; quadrupling the transitions of a single stream (`n_steps=1024`: alive `5/5`) does (§3.6.5).

## 3. The threshold

At the fixed batch of 256 transitions the dose-response has a **cliff, not a slope** (`0, 0, 0, 3, 3, 4, 5` responsive-of-5 as levels grow from 1 to 16): one to three distinct levels per update end dead on **15 of 15** seeds; four levels and above survive in **20 of 25**. Below the cliff, counting every arm and lever on that side, `24/25` seeds die — the single exception is a named `repeat4` seed (§3.6.5, evening pass). The recipe starved the update; that is the whole of the maze/heist and suite blindness.

## 4. What the fix repaired — and what it did not

Re-running the entire 240-cell grid under the wide recipe (`16×256`, §3.6.7) reshuffled everything: the archived top-3 fell to 12th–14th, and the new return leaders were still the arms that read *least* — `vit` (`1.907`) and `impoola` (`1.707`) respond on `1/15` and `3/15` cells. Return at this budget does not track the front end under either recipe (§3.6.7); the `100k` champion was a seed artifact that `250k` regresses (its `6.4` cell lands at `2.2`, a paired delta of `−4.2`; §3.6.8); and the budget axis is now measured at `100k`, `250k` and `500k` (§3.6.13) — at every point the return ranking stayed seed noise. What *does* separate the sixteen front ends — cleanly, at five seeds, deepening monotonically at every budget — is the **response measure**: readers deepen (`20/20` cells at `250k`; `30/30` at `500k`, `cbam` starpilot climbing `0.254 → 0.538 → 0.682`), non-readers do not. And `bossfight`, the game flat near zero at `100k`, leaves zero at `500k` with the study's sharpest dissociation in hand: on the same game and budget, `cbam` reads deeply and scores `1.14`, `classic` reads just as deeply and scores `0.02`, and `vit` reads nothing at all and scores `0.64`. The architecture result of this study is the response scorecard (§0.1): `cbam`, `classic`, `spatial`, `c_aug_color` read on `15/15` cells; `vit`, `impoola`, `mlp_vector` on `1–3`.

## 5. The anatomy: two blindnesses, two organs

Of the three non-readers, only four cells are truly blind — and by two different mechanisms (§3.6.9–§3.6.10). `vit`/bossfight **never ignites**: born near-dead and monotone decay, and the feature-level diagnostic shows why — its front end maps different screens to (nearly) the same 512-D vector (cross-frame range `0.154` against `classic`'s `1.31–3.10`; **zero** of 512 dimensions past scale), so the head never receives a signal. `mlp_vector`/dodgeball is born alive and **dies absorbing-ly** at `8500` — the maze death pattern — while its features keep varying: the head abandoned a live representation. The other four non-reader cells read *within* their games, a third to a tenth of `classic`'s depth, without generalizing across games.

## 6. One of the two was a bug

The anatomy predicted its own fixes, and both were tested (§3.6.11–§3.6.12). For the head: `mlp_vector` is the only arm fed **raw `uint8`** — its first `Tanh` is `91–98%` saturated *at initialization*. Scaling the input to `[0,1]` (the line every CNN arm already applies internally) rescues it on **every seed**: `5/5` runs never die on dodgeball, and on bossfight the scaled arm is a reader five times over, reaching cross-game TV `0.379` — past `classic`'s own `0.275`. For the front end: neither LayerNorm at the ViT's output (`3/3` dead) nor max-pooling over its 16 tokens (`0/3` ignite; one seed climbs to `0.053` and collapses anyway) restores the range. So the two blindnesses end with different statuses — **the head-side one was a preprocessing bug, the frontal one is an architecture-level property** at this budget.

## 7. What carried the conclusions

Three instruments, each cheap and each gated: the **cross-game response measure** (five games' initial frames — what separates the architectures), the **dense probe** (held-out random-walk frames every 500 steps — what put the collapse on a clock), and the **feature-vs-action diagnostic** (per-dimension cross-frame range over the run's own activation scale — what named the organ). On top of them, the process rules that made the results trustworthy: every claim recomputed from committed artifacts by CI on every push; single-seed claims promoted to five seeds before closing (and *revised* when the data said so — "never dips" became `4/5`, "dies at the same step" turned out to be one seed's, the "inert" `impoola` cell was its seed's); knife-edge disagreements published as findings rather than re-rolled (§3.6.10's two runs of the same cell, `0.016` vs feature-flat); one arm's error (`vae`/bossfight `NaN`, which reproduces under both recipes) recorded as an error, never smoothed to zero.

## 8. The one-line answers

- **Why do the SB3 policies ignore the observation?** The published recipe's single-stream rollout starves each update of coverage; below ~4 distinct levels per 256-transition batch, the policy's weak early response dies absorbing-ly (§3.6.4–§3.6.5).
- **Does it invalidate the architecture rankings?** Yes — return at no budget tested (`100k`–`500k`) measures either the front ends or the recipe's blindness; the response measure does, and by it the sixteen front ends separate into a stable cliff from `15/15` readers to `1–3/15` non-readers (§3.6.7–§3.6.8, §3.6.13, §0.1).
- **And the arms that stayed blind under the fix?** Two mapped defects: a front end that collapses its features to frame-invariance (`vit`, robust to every single-stage intervention tried), and a head killed by an input-scaling artifact that one division by 255 cures, replicated on every seed (§3.6.9–§3.6.12).
