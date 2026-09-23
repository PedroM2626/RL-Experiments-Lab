"""Temporal comparison 100k (easy) vs 500k (hard), per extractor and game.

For each (extractor, game): the 100k mean, the 500k mean, the delta, and whether the
500k CI95 covers the top-1 of the 100k ranking. Answers: with more budget, on the
games where memory matters, does the transformer improve?

Usage: python -m jax_port.analyze_temporal_hard [--results_root DIR]

Paths resolve from this file, not the working directory, and a missing grid is
reported instead of producing an empty comparison table.
"""
import argparse
import glob
import json
import os

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))


def collect(grade_dir, suite):
    cells = {}
    for f in sorted(glob.glob(os.path.join(grade_dir, suite, "*.json"))):
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        ev = (d.get("eval_unseen") or {}).get("mean")
        if ev is None:
            continue
        cfg = os.path.basename(f).split("__")[0]
        key = (cfg, d["game"])
        cells.setdefault(key, []).append(ev)
    return cells


def summarize(cells):
    out = {}
    for key, vals in cells.items():
        v = np.asarray(vals, float)
        n = len(v)
        sd = v.std(ddof=1) if n > 1 else 0.0
        se = sd / np.sqrt(n)
        out[key] = {"mean": float(v.mean()), "n": n, "sd": float(sd),
                    "ci95": (float(v.mean() - 1.96 * se),
                             float(v.mean() + 1.96 * se))}
    return out


def main(grade_dir):
    easy = summarize(collect(grade_dir, "temporal"))
    hard = summarize(collect(grade_dir, "temporal_hard"))
    if not easy and not hard:
        raise SystemExit(f"ERROR: no temporal/temporal_hard cells under {grade_dir}")

    for game in ("heist", "bossfight"):
        print(f"\n=== {game} (100k easy vs 500k hard) ===")
        ks_e = [k for k in easy if k[1] == game]
        ks_h = [k for k in hard if k[1] == game]
        rows = []
        for cfg in sorted(set(k[0] for k in ks_e + ks_h)):
            rows.append((cfg, easy.get((cfg, game)), hard.get((cfg, game))))
        # sort by the 500k-hard mean (descending)
        rows.sort(key=lambda r: -(r[2]["mean"] if r[2] else -1e9))
        print(f"{'cfg':<15} {'100k-easy':>10} {'CI95':>18} "
              f"{'500k-hard':>10} {'CI95':>18} {'delta':>7}")
        for cfg, e, h in rows:
            em = f"{e['mean']:.2f}" if e else "—"
            ec = (f"[{e['ci95'][0]:.2f},{e['ci95'][1]:.2f}]" if e else "—")
            hm = f"{h['mean']:.2f}" if h else "—"
            hc = (f"[{h['ci95'][0]:.2f},{h['ci95'][1]:.2f}]" if h else "—")
            dlt = f"{(h['mean'] - e['mean']):+.2f}" if (e and h) else "—"
            print(f"{cfg:<15} {em:>10} {ec:>18} {hm:>10} {hc:>18} {dlt:>7}")

    # the central question: do the transformer (and xl) leave the bottom with more budget?
    print("\n=== Does the transformer rise at 500k? ===")
    for cfg in ("transformer", "transformer_xl"):
        for game in ("heist", "bossfight"):
            e = easy.get((cfg, game))
            h = hard.get((cfg, game))
            if e and h:
                up = h["mean"] - e["mean"]
                sep = "separate CIs" if h["ci95"][0] > e["ci95"][1] else \
                    "overlapping CIs"
                print(f"  {cfg:>15} {game:>9}: {e['mean']:.2f} -> "
                      f"{h['mean']:.2f} ({up:+.2f}, {sep})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results_root", default=os.path.join(BASE, "results_grade"))
    main(parser.parse_args().results_root)
