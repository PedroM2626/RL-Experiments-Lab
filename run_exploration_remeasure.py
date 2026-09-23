"""Run the section 3.6 exploration re-measurement (roadmap item 7) as four parallel groups.

compare_maze_heist.py takes ~17 min per model on this machine, so the 40-model grid serialises
to ~11h; splitting it by game and arm pair brings it inside ~3h without touching the protocol,
because each cell writes its own tensorboard run name and its own results entry.

The process also holds ES_SYSTEM_REQUIRED for the duration of the sweep: a 3h training run
must not be suspended by the laptop's idle-sleep timer. This is a per-process request that
Windows drops when this script exits, so the user's power plan is never modified.

Usage:
    py -3.10 run_exploration_remeasure.py                 # full grid
    py -3.10 run_exploration_remeasure.py --status        # what is measured so far
"""
import argparse
import ctypes
import glob
import json
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(BASE, "logs_maze_heist")
RUNLOGS = os.path.join(LOGS, "run_logs")
GROUPS = [(game, arms) for game in ("maze", "heist") for arms in (("ppo", "icm"), ("rnd", "ngu"))]
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def set_keep_awake(on):
    if os.name != "nt":
        return
    flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0)
    ctypes.windll.kernel32.SetThreadExecutionState(flags)


def cells_measured():
    """Every comparison_results.json written by any group, keyed by config."""
    out = {}
    for path in glob.glob(os.path.join(LOGS, "maze_heist_*", "comparison_results.json")):
        if os.path.abspath(os.path.dirname(path)) == os.path.abspath(RUNLOGS):
            continue
        with open(path, encoding="utf-8") as f:
            j = json.load(f)
        for k, v in j.items():
            if k.startswith("_"):
                continue
            for cell in v:
                if cell.get("mean_reward") is not None:
                    out.setdefault(k, {})[cell["seed"]] = cell
    return out


def status():
    cells = cells_measured()
    print(f"{'config':16s} {'seeds':>5s}  bonus-verified  mean")
    for game in ("maze", "heist"):
        for arm in ("ppo", "icm", "rnd", "ngu"):
            k = f"{game}_{arm}"
            got = cells.get(k, {})
            bonus = [c for c in got.values() if (c.get("bonus") or {}).get("bonus_applied")]
            means = [c["mean_reward"] for c in got.values()]
            mean = sum(means) / len(means) if means else float("nan")
            print(f"{k:16s} {len(got):5d}  {len(bonus):14d}  {mean:6.2f}"
                  + ("" if means else "   (no data yet)"))
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--status", action="store_true", help="report progress and exit")
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    args = parser.parse_args()
    os.makedirs(RUNLOGS, exist_ok=True)
    if args.status:
        status()
        return

    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    procs = []
    for game, arms in GROUPS:
        tag = f"{game}_{'_'.join(arms)}"
        log = open(os.path.join(RUNLOGS, f"{tag}.log"), "w", encoding="utf-8")
        cmd = [sys.executable, "-u", "compare_maze_heist.py", "--games", game, "--arms", *arms,
               "--seeds", *[str(s) for s in args.seeds], "--timesteps", str(args.timesteps),
               "--log_dir", LOGS, "--device", "cuda"]
        procs.append((tag, subprocess.Popen(cmd, cwd=BASE, stdout=log, stderr=subprocess.STDOUT, env=env), log))
        print(f"launched {tag}: {' '.join(cmd[2:])}")

    set_keep_awake(True)
    t0 = time.time()
    try:
        while any(p.poll() is None for _t, p, _l in procs):
            time.sleep(60)
            n = sum(len(v) for v in cells_measured().values())
            el = (time.time() - t0) / 3600
            rate = n / el if el > 0 else 0
            print(f"[{el:5.2f}h] {n}/40 cells measured, {rate:5.2f} cells/h", flush=True)
    finally:
        set_keep_awake(False)
    for tag, p, log in procs:
        log.close()
        print(f"{tag}: exit {p.returncode}")
    print("\n=== final ===")
    status()


if __name__ == "__main__":
    main()
