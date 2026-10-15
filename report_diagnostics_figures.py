"""Figures for the dense-probe arc (README sections 3.6.4-3.6.10).

The study's 25 pre-existing figures cover the pre-diagnostic sections; the arc that
answered item 9 -- timeline, dose-response, replication, grid, budget, anatomy -- was
published as tables only. These five figures are drawn entirely from committed
artifacts (no training, no randomness), so they rebuild byte-comparably:

  dense_probe_timelines.png      3.6.4: the published arm dies, the wide arm never dips,
                                  coinrun ignites -- three timelines, one window
  dose_response_reliability.png   3.6.5: end-responsive per arm (and the levers)
  response_cliff.png              3.6.7: responsive cells per architecture, the cliff
  budget_response.png            3.6.8: TV at 100k vs 250k -- readers deepen
  anatomy_matrix.png             3.6.10: feature range vs action TV, the two loci

Usage: py -3.10 report_diagnostics_figures.py     (writes results/*.png)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results")


def load(name):
    with open(os.path.join(RESULTS, name), encoding="utf-8") as f:
        return json.load(f)


def savefig(fig, name):
    path = os.path.join(RESULTS, name)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"wrote {os.path.relpath(path, BASE)}")


def dense_probe_timelines():
    arms = [("maze, 1x256 (published recipe)", "policy_saturation_dense_nenv1_nsteps256.json"),
            ("maze, 16x16 (batch 256)", "policy_saturation_dense_nenv16_nsteps16.json"),
            ("coinrun, 1x256 (the survivor)", "coinrun_dense_curve.json")]
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, fn in arms:
        d = load(fn)
        xs = [r["steps"] for r in d["curve"]]
        ys = [r["cross_frame_tv_mean"] for r in d["curve"]]
        ax.plot(xs, ys, label=label, linewidth=1.4)
    ax.axhline(5e-5, color="grey", linewidth=0.8, linestyle="--")
    ax.text(50200, 6.3e-5, "dead threshold", color="grey", fontsize=8)
    ax.set_xlabel("environment steps")
    ax.set_ylabel("cross-frame TV (held-out probe)")
    ax.set_yscale("log")
    ax.set_title("One stream dies at 9.5k; sixteen never dip; coinrun ignites at 9k")
    ax.legend(fontsize=8, loc="upper right")
    savefig(fig, "dense_probe_timelines.png")


def dose_response_reliability():
    m = load("policy_saturation_dense_multiseed.json")
    by = m["by_arm"]
    arms = ["1x256", "2x128", "4x64", "8x32", "16x16", "16x256"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
    got = [by[a]["end_responsive"] for a in arms]
    a1.bar(range(len(arms)), got, color=["#b2182b" if g == 0 else "#2166ac" for g in got])
    a1.set_xticks(range(len(arms)))
    a1.set_xticklabels(arms, fontsize=8)
    a1.set_ylabel("seeds ending responsive (of 5)")
    a1.set_title("Rollout width -> reliability (the cliff is between 2 and 4 levels)")
    for i, g in enumerate(got):
        a1.text(i, g + 0.08, f"{g}/5", ha="center", fontsize=9)
    lev = [("repeat4", 0), ("entcoef001", 0), ("nsteps1024", 2)]
    a2.bar([l for l, _ in lev], [g for _, g in lev],
           color=["#b2182b" if g == 0 else "#2166ac" for _, g in lev])
    a2.set_ylabel("seeds ending responsive (of 2)")
    a2.set_title("Mechanism levers: what one update sees is the account")
    for i, (_, g) in enumerate(lev):
        a2.text(i, g + 0.03, f"{g}/2", ha="center", fontsize=9)
    savefig(fig, "dose_response_reliability.png")


def response_cliff():
    g = load("suite_validity_nenv16_grid.json")
    resp = g["responsiveness"]
    order = [a for a, _ in g["global_mean_ranking"]]
    order = sorted(order, key=lambda a: resp[a]["cells_tv_over_0p1"])
    vals = [resp[a]["cells_tv_over_0p1"] for a in order]
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#b2182b" if v <= 3 else ("#7f7f7f" if v < 15 else "#2166ac") for v in vals]
    ax.barh(range(len(order)), vals, color=colors)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=8)
    ax.set_xlabel("responsive cells (cross-game TV >= 0.1, of 15)")
    ax.set_title("The response cliff: what separates the front ends at this budget")
    savefig(fig, "response_cliff.png")


def budget_response():
    b = load("suite_validity_nenv16_budget250k.json")["arms"]
    labels, before, after = [], [], []
    for arm in ("classic", "cbam", "vit", "mlp_vector"):
        for game in ("starpilot", "dodgeball"):
            labels.append(f"{arm}\n{game[:6]}")
            before.append(b[arm][game]["tv_mean_100k"])
            after.append(b[arm][game]["tv_mean_250k"])
    import numpy as np
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - 0.2, before, 0.4, label="100k", color="#7f7f7f")
    ax.bar(x + 0.2, after, 0.4, label="250k", color="#2166ac")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("within-pair cross-game TV mean")
    ax.set_title("2.5x budget: readers deepen, non-readers do not")
    ax.legend()
    savefig(fig, "budget_response.png")


def anatomy_matrix():
    cells = [("classic", "bossfight"), ("classic", "dodgeball"), ("vit", "bossfight"),
             ("vit", "dodgeball"), ("impoola", "bossfight"), ("mlp", "bossfight"),
             ("mlp", "dodgeball")]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for a, g in cells:
        d = load(f"feature_vs_actor_{a}_{g}.json")
        w = d["within_game"]
        ax.scatter(w["features"]["feat_range_over_scale_mean"], max(w["action_tv"], 1e-7),
                   s=90, zorder=3)
        ax.annotate(f"{a}/{g[:4]}", (w["features"]["feat_range_over_scale_mean"],
                                     max(w["action_tv"], 1e-7)), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    # the multi-seed pass: lighter dots behind the seed-42 anchors
    for a, g in cells:
        for s in (43, 44, 45, 46):
            fn = os.path.join(RESULTS, f"feature_vs_actor_{a}_{g}_seed{s}.json")
            if not os.path.exists(fn):
                continue
            w = json.load(open(fn, encoding="utf-8"))["within_game"]
            ax.scatter(w["features"]["feat_range_over_scale_mean"],
                       max(w["action_tv"], 1e-7), s=25, alpha=0.45, color="#b2182b", zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("feature cross-frame range over activation scale (front end)")
    ax.set_ylabel("within-game action TV (head)")
    ax.set_title("Two anatomies: frontal collapse (bottom-left) vs head-side blindness (bottom-right)")
    ax.axvline(0.5, color="grey", linewidth=0.8, linestyle="--")
    ax.text(0.55, 1e-6, "features ~ frame-invariant", fontsize=7, color="grey")
    savefig(fig, "anatomy_matrix.png")


def main():
    dense_probe_timelines()
    dose_response_reliability()
    response_cliff()
    budget_response()
    anatomy_matrix()


if __name__ == "__main__":
    main()
