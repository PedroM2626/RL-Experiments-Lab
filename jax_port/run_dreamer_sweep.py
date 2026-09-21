"""Dreamer tuning sweep (sequential, one process per cell).

Cells B/C/D over baseline A (raw/ent 3e-4, already measured ret 0.0):
  B: symlog + ent 3e-4   (isolates reward scale)
  C: symlog + ent 1e-3   (scale + exploration)
  D: raw    + ent 1e-3   (isolates entropy)
Usage (free GPU):
    wsl -e env PYTHONPATH=... /root/procgen-jax/bin/python \
      jax_port/run_dreamer_sweep.py --frames 1000000 --seed 42
Summary in jax_port/dreams/sweep.json.
"""

import argparse
import gc
import json
import subprocess
import sys

CELLS = [
    ("B", ["--reward-mode", "symlog", "--ent-coef", "3e-4"]),
    ("C", ["--reward-mode", "symlog", "--ent-coef", "1e-3"]),
    ("D", ["--reward-mode", "raw", "--ent-coef", "1e-3"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="coinrun")
    ap.add_argument("--frames", type=int, default=1000000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--eval-eps", type=int, default=0)
    ap.add_argument("--num-envs", type=int, default=16)
    args = ap.parse_args()
    seeds = args.seeds or [args.seed]
    summ = {}
    for seed in seeds:
        out = f"jax_port/dreams/dreamer_symlog_s{seed}.json"
        cmd = [sys.executable, "jax_port/train_dreamer.py", "--game", args.game,
               "--frames", str(args.frames), "--seed", str(seed),
               "--num-envs", str(args.num_envs),
               "--reward-mode", "symlog", "--ent-coef", "3e-4",
               "--eval-eps", str(args.eval_eps),
               "--eval-det-eps", str(args.eval_eps),
               "--out", out, "--out-dir", "jax_port/dreams"]
        print(f"[seed {seed}] symlog/ent 3e-4 eval {args.eval_eps}",
              flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(r.stdout[-500:] if r.stdout else "")
        if r.returncode != 0:
            print(r.stderr[-2000:])
            summ[f"s{seed}"] = {"ok": False}
            continue
        d = json.load(open(out))
        summ[f"s{seed}"] = {"ok": True,
                            "eval": d.get("eval_unseen"),
                            "eval_det": d.get("eval_unseen_det"),
                            "ret": d["train_ret_mean20"], "sps": d["sps"]}
        gc.collect()
    json.dump(summ, open("jax_port/dreams/sweep_seeds.json", "w"), indent=2)
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
