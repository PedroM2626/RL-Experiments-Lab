"""Comparacao temporal 100k (easy) vs 500k (hard) por extrator/jogo.

Para cada (extrator, jogo): media 100k, media 500k, delta, e se o CI95
do 500k nao cobre o 1o lugar do ranking 100k. Responde: com mais budget
e dificuldade onde a memoria importa, o transformer melhora?
"""
import glob
import json
import os

import numpy as np

B = "/mnt/c/Users/Acer/Downloads/MLE/jax_port/results_grade"


def collect(suite):
    cells = {}
    for f in glob.glob(os.path.join(B, suite, "*.json")):
        d = json.load(open(f))
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


easy = summarize(collect("temporal"))
hard = summarize(collect("temporal_hard"))

for game in ("heist", "bossfight"):
    print(f"\n=== {game} (100k easy vs 500k hard) ===")
    ks_e = [k for k in easy if k[1] == game]
    ks_h = [k for k in hard if k[1] == game]
    rows = []
    for cfg in sorted(set(k[0] for k in ks_e + ks_h)):
        e = easy.get((cfg, game))
        h = hard.get((cfg, game))
        rows.append((cfg, e, h))
    # ordena por media hard (desc)
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

# a pergunta central: transformer (e xl) saem do fundo com budget?
print("\n=== Transformer sobe com 500k? ===")
for cfg in ("transformer", "transformer_xl"):
    for game in ("heist", "bossfight"):
        e = easy.get((cfg, game))
        h = hard.get((cfg, game))
        if e and h:
            up = h["mean"] - e["mean"]
            sep = "CI separado" if h["ci95"][0] > e["ci95"][1] else \
                "ICs sobrepostos"
            print(f"  {cfg:>15} {game:>9}: {e['mean']:.2f} -> "
                  f"{h['mean']:.2f} ({up:+.2f}, {sep})")
