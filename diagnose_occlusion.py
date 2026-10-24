"""Occlusion attribution over the key-cell checkpoints: WHERE does a reader read?

The response measure (section 0.1) says whether a policy conditions on its
observation; it does not say where. This script slides a zeroed 16x16 patch over
the 64x64 frame (a 4x4 grid of occlusion sites; the vector arm occludes the
corresponding 4x4 blocks of its 16x16 downsample) on held-out random-walk frames,
and measures how much each occlusion site moves the action distribution against
the unoccluded baseline -- the attribution analogue of the response measure, run
on the eight committed key-cell checkpoints of section 4 (readers: classic and
mlp_scaled; non-readers: vit, impoola, and the raw mlp_vector cell the
input-scale rescue came from).

A reader should show a structured map (some sites matter much more than others)
and a non-reader a flat one, whatever its return; the pair is the instrument the
"reading is necessary but not sufficient" dissociation of section 3.6.13 asks for.

Usage:
    py -3.10 diagnose_occlusion.py               # all 8 committed cells -> figure + summary
    py -3.10 diagnose_occlusion.py --check      # fail if the summary is stale
"""
import argparse
import json
import os

import numpy as np
import torch
from stable_baselines3 import PPO

from diagnose_policy_saturation import cross_game_response, probe_frames
from response_guard import RESPONSIVE_TV

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, "results", "models")
OUT_JSON = os.path.join(BASE, "results", "occlusion_summary.json")
OUT_PNG = os.path.join(BASE, "results", "occlusion_maps.png")

# the eight key cells of the section 4 provenance table, in figure order
CELLS = [
    ("classic_bossfight", "bossfight", True),
    ("classic_dodgeball", "dodgeball", True),
    ("mlp_scaled_bossfight", "bossfight", True),
    ("mlp_scaled_dodgeball", "dodgeball", True),
    ("vit_bossfight", "bossfight", False),
    ("vit_dodgeball", "dodgeball", False),
    ("impoola_bossfight", "bossfight", False),
    ("mlp_dodgeball", "dodgeball", False),
]
GRID = 4          # 4x4 occlusion sites
PATCH = 16        # pixels per site on the 64x64 frame
N_FRAMES = 12     # held-out random-walk frames per cell


def action_probs(model, obs_np):
    """The policy's action distribution for one observation, via the same entry point
    the audit uses (extract_features -> policy_net -> distribution), so the
    attribution cannot drift from the measure it attributes."""
    pol = model.policy
    t = torch.as_tensor(np.expand_dims(obs_np, 0)).to(next(pol.parameters()).device)
    with torch.no_grad():
        f = pol.extract_features(t, pol.pi_features_extractor)
        net = pol.mlp_extractor.policy_net
        latent = net[3](net[2](net[1](net[0](f))))
        p = pol._get_action_dist_from_latent(latent).distribution.probs
    return p.detach().cpu().numpy()[0]


def occlude(obs, r, c, vector):
    """Zero one occlusion site of a single observation (CHW image or 256-D vector)."""
    o = obs.copy()
    if vector:
        g = o.reshape(16, 16)          # the wrapper's 16x16 downsample
        g[r * 4:(r + 1) * 4, c * 4:(c + 1) * 4] = 0
        return g.reshape(-1)
    o[:, r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH] = 0
    return o


def measure_cell(name, game, n_frames=N_FRAMES, device="cpu"):
    model = PPO.load(os.path.join(MODELS, f"{name}.zip"), device=device)
    vector = len(model.observation_space.shape) == 1
    scaled = model.observation_space.dtype == np.float32
    frames = probe_frames(game, seed=1042, n=n_frames, vector=vector)
    if scaled:  # the mlp_scaled cells were trained on [0,1] float32 observations
        frames = [f.astype(np.float32) / 255.0 for f in frames]
    base = [action_probs(model, f) for f in frames]
    patch = np.zeros((GRID, GRID))
    for r in range(GRID):
        for c in range(GRID):
            tvs = []
            for f, b in zip(frames, base):
                p = action_probs(model, occlude(f, r, c, vector))
                tvs.append(float(0.5 * np.abs(p - b).sum()))
            patch[r, c] = float(np.mean(tvs))
    resp = cross_game_response(model, vector=vector, vector_scale=scaled)
    return {
        "cell": name, "game": game, "frames": n_frames,
        "grid": GRID, "patch": PATCH,
        "patch_map": [[round(v, 6) for v in row] for row in patch],
        "occlusion_sensitivity": round(float(patch.mean()), 6),
        "occlusion_max_site": round(float(patch.max()), 6),
        "cross_game_tv": resp["max_pairwise_tv"],
        "responsive": bool(resp["max_pairwise_tv"] >= RESPONSIVE_TV),
    }


def draw(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vmax = max(0.05, max(x["occlusion_max_site"] for x in rows))
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.6))
    for ax, row in zip(axes.ravel(), rows):
        m = np.array(row["patch_map"])
        im = ax.imshow(m, cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(f'{row["cell"]}  TV {row["cross_game_tv"]:.3f}  '
                     f'sens {row["occlusion_sensitivity"]:.4f}', fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.8, label="mean TV vs baseline")
    fig.suptitle("Occlusion attribution over the key cells (24/10/2026): where a reader reads",
                 fontsize=12)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    rows = [measure_cell(name, game) for name, game, _ in CELLS]
    if args.check:
        with open(OUT_JSON, encoding="utf-8") as f:
            committed = json.load(f)
        if committed["cells"] != rows:
            print("the occlusion summary is stale: rerun diagnose_occlusion.py")
            return 1
        print("the occlusion summary matches the committed checkpoints")
        return 0
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"_provenance": {
            "produced_by": "diagnose_occlusion.py",
            "question": "where on the frame does a reader policy read; is a non-reader flat everywhere",
            "checkpoints": "results/models/*.zip (the section 4 key cells)",
            "measure": "mean TV of the action distribution between each occluded 16x16 site "
                       "and the unoccluded baseline, over 12 held-out random-walk frames",
        }, "cells": rows}, f, indent=1)
    draw(rows)
    for r in rows:
        print(f'{r["cell"]:24s} TV={r["cross_game_tv"]:.4f}  '
              f'sensitivity={r["occlusion_sensitivity"]:.5f}  max site={r["occlusion_max_site"]:.5f}')
    print(f"saved {os.path.relpath(OUT_JSON, BASE)} and the figure")


if __name__ == "__main__":
    raise SystemExit(main())
