"""Validity audit of the 16-architecture suite: does each trained arm read its observation?

README 3.6.2 measured that a fresh 100k run of the published recipe collapses to a
state-independent policy on starpilot, bossfight and dodgeball, while coinrun does not. That
was one extractor. This walks all sixteen arms of the headline comparison and, for each one,
reports the return AND whether the policy responds to the observation at all -- because a
ranking of policies that do not read the input cannot be attributed to the visual front end.

The training call is the published one: compare_suite.train_one / compare_new_archs.train_one
are imported, not copied, so the hyperparameters cannot drift from the arms being audited.
The response measure is diagnose_policy_saturation.cross_game_response: feed the trained policy
the initial frame of five different Procgen games and take the largest pairwise total variation
between its action distributions.

Each arm is written to the output file as it finishes, and an existing entry is skipped, so an
interrupted audit resumes rather than restarts. No checkpoints are saved: the audit needs the
response of each policy once, and 16 zips would be ~800 MB of weight in the repository for it.

Shards should stay short: two of the three shards in the first full run deadlocked in cffi
while creating the env of their 3rd/4th arm (see run_arm), after ~20 env creations in one
process. One arm per process is the safe shape, and run_arm forces a collection at each
boundary regardless.

Usage:
    py -3.10 audit_suite_validity.py --game starpilot --configs classic mlp_vector
    py -3.10 audit_suite_validity.py --game starpilot --out results/suite_validity_starpilot.json
    py -3.10 audit_suite_validity.py --merge "logs_audit/shards/sh-*.json"
"""
import argparse
import gc
import json
import os
import time

import compare_new_archs as CNA
import compare_suite as CS
from compare_augment_contrastive import ContrastivePixelDrop
from diagnose_policy_saturation import cross_game_response
from models.world_model_extractors import ReconBottleneckExtractor

BASE = os.path.dirname(os.path.abspath(__file__))
# (arm, family, extractor class, kwargs, vector, vector_scale) -- assembled from the
# producers themselves. The sixteen published arms carry vector_scale=False, which is
# exactly what 3.6.2/3.6.7 measured. mlp_scaled is the 3.6.12 input-scale lever as a
# full grid arm: the same 256-D vector observation, scaled to [0,1] float32 by the
# wrapper (procgen_wrapper.ProcgenVectorWrapper(scale=True)) -- it is opt-in, not part
# of the published sixteen, so the default audit and every merged artifact are unchanged.
ARMS = ([(k, "world_model", cls, kw, False, False) for k, cls, kw in CS.WORLD_CONFIGS]
        + [(k, "procgen_cnn", cls, kw, vec, False) for k, cls, kw, vec in CS.PROCGEN_CONFIGS]
        + [("mlp_scaled", "procgen_cnn", None, {}, True, True)]
        + [("recon_bottleneck", "world_model", ReconBottleneckExtractor,
            dict(features_dim=512, latent_dim=128), False, False)]
        + [(k, "augment", cls, kw, False, False) for k, cls, kw in CS.AUGMENT_CONFIGS]
        + [("c_aug_noise_drop", "augment", ContrastivePixelDrop, dict(features_dim=512), False, False)]
        + [(k, "new_arch", cls, kw, False, False) for k, cls, kw in CNA.NEW_ARCH_CONFIGS])
BY_NAME = {a[0]: a for a in ARMS}
# the three opt-in arms added after the audit: the 3.6.12 input-scale lever and the two
# genuinely-distinct variants of the same-program pairs (README 3.6.2). They are available
# by name so a future grid can measure 17 distinct front ends; the default audit and every
# merged artifact still cover exactly the published sixteen.
EXTRA_ARMS = {"mlp_scaled", "recon_bottleneck", "c_aug_noise_drop"}
PUBLISHED_ARMS = [a[0] for a in ARMS if a[0] not in EXTRA_ARMS]


def load(out):
    if os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            return json.load(f)
    return {"_provenance": {
        "produced_by": "audit_suite_validity.py",
        "question": "does each of the 16 published arms condition on its observation",
        "training": "compare_suite.train_one / compare_new_archs.train_one, recipe recorded in n_envs/n_steps",
        "response_measure": "max pairwise total variation across five games' initial frames "
                            "(diagnose_policy_saturation.cross_game_response)",
        "note": "TV near 0 means the policy emits the same action distribution for five "
                "different games; it is not a measure of skill",
    }, "arms": {}}


def save(out, data):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, out)


def steps_logged(log_dir):
    """How far the most recent run in log_dir got, in environment steps.

    When an arm dies mid-training the error message alone cannot say whether it died at step
    1 or at step 99k, and that difference is the finding (the vae arm's NaN on bossfight is a
    mid-run divergence, not a broken evaluation). The number is written into the artifact
    because a claim in the README has to be carried by the file it cites, not by a log dir
    that CI never sees.
    """
    import glob as _glob
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return None
    files = sorted(_glob.glob(os.path.join(log_dir, "PPO_*", "events*")), key=os.path.getmtime)
    if not files:
        return None
    ea = EventAccumulator(files[-1], size_guidance={"scalars": 5000})
    ea.Reload()
    tags = ea.Tags()["scalars"]
    if "rollout/timesteps" in tags and ea.Scalars("rollout/timesteps"):
        return int(ea.Scalars("rollout/timesteps")[-1].value)
    if "train/clip_fraction" in tags:  # one logged update per n_steps batch
        return len(ea.Scalars("train/clip_fraction")) * 256
    return None


def run_arm(name, game, timesteps, seed, device, n_envs=1, n_steps=256):
    _, family, cls, kw, vector, vector_scale = BY_NAME[name]
    # Safe point for gym3 finalizers. Each arm builds ~7 Procgen envs (train, eval, five
    # cross-game probes) at n_envs=1 -- 22 at the pilot's 16 -- and gym3's Libenv.__del__
    # closes the C handle through cffi; when the gc triggers that collector while the
    # main thread is already inside cffi's accessor-construction lock, the reentrant
    # acquire deadlocks -- observed on two shards at once, 0% CPU, hung in make_accessor
    # with a __del__/close frame in the middle of the stack. Collecting here runs those
    # finalizers where no FFI call is in progress.
    gc.collect()
    t0 = time.time()
    if family == "new_arch":
        mean, std, model = CNA.train_one(game, cls, kw, timesteps, seed, "logs_audit", device,
                                         n_envs=n_envs, n_steps=n_steps)
    else:
        mean, std, model = CS.train_one(game, cls, kw, timesteps, seed, "logs_audit",
                                        device, vector=vector, n_envs=n_envs, n_steps=n_steps,
                                        vector_scale=vector_scale)
    resp = cross_game_response(model, vector=vector, vector_scale=vector_scale)
    return {"family": family, "vector_obs": vector, "vector_scale": vector_scale,
            "return_10eps": round(float(mean), 3),
            "std_10eps": round(float(std), 3),
            "seconds": round(time.time() - t0, 1), "n_envs": n_envs, "n_steps": n_steps,
            **resp}


def merge(pattern, out, key_by_seed=False):
    """Combine shard files written by parallel --out runs into the canonical output.

    Several processes share one GPU, so they train concurrently but must not write the same
    JSON. Each keeps its own file; this one checks they cover the arms exactly once and then
    produces the single artifact the README cites.

    Shards are only the same measurement if they share game/budget/seed AND recipe -- and
    that is asserted, recipe included: otherwise a n_envs=1 shard and a pilot shard, or a
    bossfight row and a dodgeball row of one arm, silently overwrite each other and the
    artifact looks like a clean 16-arm table. A seed sweep is the legitimate exception --
    the same arm appears once per shard -- so --key-by-seed names those rows arm@seed.
    """
    import glob as _glob
    files = sorted(_glob.glob(pattern))
    assert files, f"no shard files match {pattern}"
    arms, prov, fixed = {}, None, None
    for f in files:
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        p = d.get("_provenance", {})
        stamp = (p.get("game"), p.get("timesteps"), p.get("seed"),
                 p.get("n_envs"), p.get("n_steps"))
        if prov is None:
            prov, fixed = p, stamp
        elif not key_by_seed:
            assert stamp == fixed, (f"{f}: provenance {stamp} differs from {fixed} -- shards of "
                                    "different games or budgets are not one artifact")
        else:
            # a seed sweep varies only the seed: game, budget and recipe must still agree,
            # or a bossfight shard would fuse into a starpilot merge unnoticed
            sans_seed = lambda t: tuple(x for i, x in enumerate(t) if i != 2)
            assert sans_seed(stamp) == sans_seed(fixed), (f"{f}: {stamp} vs {fixed} -- key-by-seed "
                                                          "shards must share game/budget/recipe")
        for name, row in d.get("arms", {}).items():
            key = f"{name}@{p.get('seed')}" if key_by_seed else name
            dup = arms.get(key)
            assert dup is None or dup == row, f"{key} measured twice with different values"
            arms[key] = row
    data = {"_provenance": {**prov, "shards_merged": files}, "arms": arms}
    if key_by_seed:
        data["_provenance"]["rows"] = "arm@seed, one shard per seed"
    save(out, data)
    print(f"{len(arms)} braços fundidos em {os.path.relpath(out, BASE).replace(os.sep, '/')}")
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="starpilot")
    ap.add_argument("--timesteps", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-envs", type=int, default=1,
                    help="parallel envs per run. The published recipe is 1 (the collapsing "
                         "one, 3.6.2-3.6.5); the pilot raises it to 16 -- each env on its "
                         "own level stream, per-update batch 4096 (3.6.6)")
    ap.add_argument("--n-steps", type=int, default=256,
                    help="rollout length per env per update, the published 256 by default")
    ap.add_argument("--configs", nargs="*", default=PUBLISHED_ARMS,
                    help="the sixteen published arms by default; mlp_scaled (the 3.6.12 "
                         "input-scale lever) is available by naming it")
    ap.add_argument("--out", default=None)
    ap.add_argument("--merge", default=None, metavar="GLOB",
                    help="fuse shard files into --out instead of training")
    ap.add_argument("--key-by-seed", action="store_true",
                    help="with --merge: name the rows arm@seed, for a sweep over seeds")
    args = ap.parse_args()
    out = args.out or os.path.join(BASE, "results", f"suite_validity_{args.game}.json")
    if args.merge:
        merge(args.merge, out, key_by_seed=args.key_by_seed)
        return
    data = load(out)
    data["_provenance"].update({"game": args.game, "timesteps": args.timesteps,
                                "seed": args.seed, "n_envs": args.n_envs,
                                "n_steps": args.n_steps})
    todo = [c for c in args.configs if c not in data["arms"]]
    print(f"{len(args.configs)} braços pedidos, {len(todo)} a medir, "
          f"{len(args.configs) - len(todo)} já no arquivo", flush=True)
    for i, name in enumerate(todo, 1):
        try:
            row = run_arm(name, args.game, args.timesteps, args.seed, args.device,
                          n_envs=args.n_envs, n_steps=args.n_steps)
            print(f"[{i}/{len(todo)}] {name:14s} ret={row['return_10eps']:6.2f}  "
                  f"TV={row['max_pairwise_tv']:.6f}  argmax_distintos="
                  f"{row['distinct_argmax_across_games']}/5  "
                  f"1oTanh={row['first_tanh_saturated_fraction']*100:5.1f}%  "
                  f"feat_max={row['feat_max']:7.2f}  {row['seconds']/60:5.1f} min", flush=True)
        except Exception as e:  # one OOM must not discard the arms already measured
            row = {"error": f"{type(e).__name__}: {e}",
                   "steps_logged": steps_logged("logs_audit")}
            print(f"[{i}/{len(todo)}] {name:14s} ERRO after {row['steps_logged']} steps: "
                  f"{row['error'][:80]}", flush=True)
        data["arms"][name] = row
        save(out, data)
    save(out, data)
    print(f"\nescrito em {os.path.relpath(out, BASE).replace(os.sep, '/')}")


if __name__ == "__main__":
    main()
