"""Run the 16-architecture grid under the fixed recipe (README 3.6.6, item (iii)).

240 cells: 16 arms x {starpilot, bossfight, dodgeball} x seeds 42-46, each trained by
audit_suite_validity.py at n_envs=16 x n_steps=256, 100k steps -- one process per cell,
because gym3's cffi finalizers deadlock when a process creates too many Procgen envs in
sequence (see audit_suite_validity.run_arm). On the pilot's measured rate the whole grid
is ~13.4 GPU-hours; 4 workers share the 8 GB GPU without OOM.

Resume-safe twice over: a cell is skipped if its arm is already in the shard file (the
audit skips it too), and error rows count as done -- a cell whose training failed mid-run
keeps its error as the result, exactly as in the first audit (the vae/bossfight NaN).
The starpilot seed-42 shard can be adopted from the pilot artifact, whose provenance is
the same cell spec.

The process holds ES_SYSTEM_REQUIRED for the duration: a multi-hour run must not be
suspended by the laptop's idle-sleep timer. The request is per-process and Windows
drops it when this script exits, so the power plan is never modified.

Usage:
    py -3.10 run_grid_nenv16.py --adopt-pilot          # full grid, starpilot/s42 from 3.6.6
    py -3.10 run_grid_nenv16.py --workers 4 --adopt-pilot
    py -3.10 run_grid_nenv16.py --status              # cell counts per shard, no training
"""
import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from datetime import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
SHARDS_DEFAULT = os.path.join(BASE, "logs_audit", "shards_nenv16_grid")
SHARDS = SHARDS_DEFAULT  # the 3.6.7 grid's dir; a budget sweep passes its own
PILOT = os.path.join(BASE, "results", "suite_validity_starpilot_nenv16.json")
GAMES = ("starpilot", "bossfight", "dodgeball")
SEEDS = (42, 43, 44, 45, 46)
ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001

# pilot-measured cell minutes, heaviest first so the queue drains the slow arms while
# every worker still has work; the ordering only shapes the tail of the run
HEAVY_FIRST = ("impala", "resnet18", "vit", "lstm_attention", "impoola", "cbam", "spatial",
               "vae", "c_aug_crop", "classic", "ae", "recon", "contrastive", "c_aug_noise",
               "c_aug_color", "mlp_vector")


def set_keep_awake(on):
    if os.name != "nt":
        return
    ctypes.windll.kernel32.SetThreadExecutionState(
        ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


def shard_path(spec, game, seed, arm):
    """One file PER CELL (the sweep's dir may differ from the 100k grid's). The audit's
    save() is a read-modify-write of the whole file -- two processes completing into the
    same --out clobber each other's rows (observed: two finished cells vanished when
    three workers landed within 30 s), which is why the merge docstring says concurrent
    processes must not share a JSON. One small shard per cell, each written by exactly
    one process, is the shape that cannot race."""
    return os.path.join(spec["shards"], f"sh-{game}-s{seed}-{arm}.json")


def arms_in(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("arms", {})


def adopt_pilot():
    """The 3.6.6 pilot IS the starpilot/seed-42 cells: same arms, same provenance
    (game, seed, timesteps, n_envs, n_steps). Splitting it into the 16 per-cell shards
    saves 16 cells and keeps one uniform resume path."""
    done = 0
    for arm in HEAVY_FIRST:
        dst = shard_path({"shards": SHARDS_DEFAULT}, "starpilot", 42, arm)
        if os.path.exists(dst):
            done += 1
            continue
        with open(PILOT, encoding="utf-8") as f:
            data = json.load(f)
        p = data["_provenance"]
        assert (p["game"], p["seed"], p["timesteps"], p["n_envs"], p["n_steps"]) == \
            ("starpilot", 42, 100000, 16, 256), "the pilot artifact is not this cell spec"
        row = data["arms"][arm]
        os.makedirs(SHARDS_DEFAULT, exist_ok=True)
        with open(dst, "w", encoding="utf-8") as f:
            json.dump({"_provenance": {k: p[k] for k in
                                       ("game", "seed", "timesteps", "n_envs", "n_steps")},
                       "arms": {arm: row}}, f, indent=2)
        done += 1
    print(f"pilot adopted as {done}/16 per-cell starpilot/s42 shards")


def queue_cells(spec):
    """All (game, seed, arm) cells not yet present in their per-cell shard."""
    todo = []
    for game in spec["games"]:
        for seed in SEEDS:
            for arm in spec["arms"]:
                if arm not in arms_in(shard_path(spec, game, seed, arm)):
                    todo.append((game, seed, arm))
    return todo


def status(spec):
    total = done = errors = 0
    for game in spec["games"]:
        for seed in SEEDS:
            got = errs = 0
            for arm in spec["arms"]:
                row = arms_in(shard_path(spec, game, seed, arm)).get(arm)
                if row is None:
                    continue
                got += 1
                errs += 1 if "error" in row else 0
            total += len(spec["arms"])
            done += got
            errors += errs
            print(f"{game:10s} s{seed}@{spec['timesteps'] // 1000}k: {got:2d}/{len(spec['arms'])}"
                  + (f"  ({errs} errors)" if errs else ""))
    print(f"total {done}/{total} cells, {errors} error rows")
    return done, total, errors


def run(spec, workers):
    # adopting the pilot is idempotent bookkeeping, but it is a 100k cell set: only
    # the default full grid may adopt it, or a budget sweep would import 100k rows
    if spec["timesteps"] == 100000 and spec["shards"] == SHARDS_DEFAULT \
            and set(spec["arms"]) == set(HEAVY_FIRST):
        adopt_pilot()
    set_keep_awake(True)
    try:
        todo = queue_cells(spec)
        print(f"{datetime.now().strftime('%H:%M:%S')} {len(todo)} cells to measure "
              f"({len(spec['games']) * len(spec['arms']) * len(SEEDS) - len(todo)} already in shards)",
              flush=True)
        running = []  # (proc, cell, started)
        while todo or running:
            while todo and len(running) < workers:
                game, seed, arm = todo.pop(0)
                cmd = [sys.executable, "-u", os.path.join(BASE, "audit_suite_validity.py"),
                       "--game", game, "--seed", str(seed), "--timesteps", str(spec["timesteps"]),
                       "--n-envs", "16", "--n-steps", "256",
                       "--configs", arm, "--out", shard_path(spec, game, seed, arm)]
                t0 = time.time()
                running.append((subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                                stderr=subprocess.STDOUT), (game, seed, arm), t0))
                print(f"{datetime.now().strftime('%H:%M:%S')} start {game}/s{seed}/{arm} "
                      f"({len(todo)} queued)", flush=True)
            time.sleep(5)
            for entry in running[:]:
                proc, (game, seed, arm), t0 = entry
                if proc.poll() is None:
                    continue
                running.remove(entry)
                got = arms_in(shard_path(spec, game, seed, arm)).get(arm)
                if got is None:
                    print(f"{datetime.now().strftime('%H:%M:%S')} MISSING {game}/s{seed}/{arm} "
                          f"after {(time.time() - t0) / 60:.1f} min (will need a re-run)", flush=True)
                else:
                    err = "error" in got
                    print(f"{datetime.now().strftime('%H:%M:%S')} done   {game}/s{seed}/{arm} "
                          f"ret={got.get('return_10eps', '--'):>5} TV={got.get('max_pairwise_tv', 0):.6f}"
                          f"{'  ERROR: ' + got['error'][:50] if err else ''} "
                          f"({(time.time() - t0) / 60:.1f} min, {len(todo)} queued)", flush=True)
        print(f"{datetime.now().strftime('%H:%M:%S')} sweep complete:", flush=True)
        status(spec)
    finally:
        set_keep_awake(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timesteps", type=int, default=100000,
                    help="budget per cell; the committed 3.6.7 grid is 100k")
    ap.add_argument("--games", nargs="*", default=list(GAMES))
    ap.add_argument("--arms", nargs="*", default=list(HEAVY_FIRST))
    ap.add_argument("--shards", default=None,
                    help="shard dir; default is the 3.6.7 grid's, a budget sweep names its own")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--adopt-pilot", action="store_true",
                    help="adopt the 3.6.6 pilot as the starpilot/s42 shard and exit (the "
                         "grid run adopts it automatically)")
    args = ap.parse_args()
    spec = {"timesteps": args.timesteps, "games": tuple(args.games), "arms": tuple(args.arms),
            "shards": args.shards or SHARDS_DEFAULT}
    if args.status:
        status(spec)
        return
    if args.adopt_pilot:
        adopt_pilot()
        return
    run(spec, args.workers)


if __name__ == "__main__":
    main()
