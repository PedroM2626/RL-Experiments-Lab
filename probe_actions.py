"""
Action space probing (15 discrete actions): for each action, applies it over several frames
and measures pixel variation and reward -- identifies which buttons move/jump/shoot, serving
as the baseline for the HRL skill library (README 15, fixed skills for jumper/plunder).

Importing this module used to run the whole sweep: the loop lived at module scope, so any
`from probe_actions import probe` paid two minutes of emulator steps and printed a table. It is
now a CLI, with --games defaulting to the two the HRL arm uses.

Caveat measured on 11/10/2026: with num_levels=1 a generated level can already be complete at
spawn (jumper, seed 42: prev_level_complete=1 and reward +10 on the FIRST step, pixel change
0.0), so the done% and reward columns of a single-level probe report "this level is trivial",
not "this action is jump". Pass --num-levels 200 to probe the distribution the study trains on.

Usage:
    py -3.10 probe_actions.py
    py -3.10 probe_actions.py --games maze --num-levels 200
"""
import argparse

import numpy as np

from procgen_wrapper import make_procgen_env


def probe(game, seeds=(42, 43, 44, 45, 46), frames=16, num_levels=1):
    """Return {action: {pixdiff, reward, done_pct, peak}} averaged over seeds, and print it."""
    print(f"\n=== {game} (mean of {len(seeds)} levels, {frames} frames/action, "
          f"num_levels={num_levels}, NEW env per action) ===")
    print(f"{'act':>4s} {'pixdiff':>10s} {'reward':>8s} {'done%':>7s} {'peakframe':>10s}")
    agg = {}
    for a in range(15):
        for seed in seeds:
            env = make_procgen_env(game, num_levels=num_levels, distribution_mode='easy',
                                   seed=seed, vector=False)
            obs0, _ = env.reset()
            total_r, diff, done = 0.0, 0.0, False
            peak = 0.0
            prev = obs0.astype(np.float32)
            for t in range(frames):
                o, r, term, trunc, _ = env.step(a)
                d = float(np.abs(o.astype(np.float32) - prev).mean())
                diff += d
                peak = max(peak, d)
                prev = o.astype(np.float32)
                total_r += r
                if term or trunc:
                    done = True
                    break
            acc = agg.setdefault(a, [0.0, 0.0, 0.0, 0.0])
            acc[0] += diff; acc[1] += total_r; acc[2] += done; acc[3] += peak
            env.close()
    rows = {}
    for a in range(15):
        d, r, dn, pk = agg[a]
        n = len(seeds)
        rows[a] = {"pixdiff": d / n, "reward": r / n, "done_pct": 100 * dn / n, "peak": pk / n}
        print(f"{a:4d} {rows[a]['pixdiff']:10.2f} {rows[a]['reward']:8.2f} "
              f"{rows[a]['done_pct']:6.0f}% {rows[a]['peak']:10.2f}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--games', nargs='+', default=['jumper', 'plunder'])
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    ap.add_argument('--frames', type=int, default=16)
    ap.add_argument('--num-levels', type=int, default=1,
                    help='1 probes one fixed level per seed; 200 probes the training distribution')
    args = ap.parse_args()
    for g in args.games:
        try:
            probe(g, seeds=tuple(args.seeds), frames=args.frames, num_levels=args.num_levels)
        except Exception as e:  # a missing native binary must not hide the other games
            print(f'{g}: ERROR {e}')


if __name__ == '__main__':
    main()
