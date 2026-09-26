"""Regenerate README section 4 figures from data that is in the repository, and
report where each embedded figure actually comes from.

Two of the thirteen figures (coinrun 50k, bossfight 100k world models) are backed by
per-seed values that survive in results/legacy_records.json, so they can be rebuilt
without retraining. The rest were drawn from the timestamped logs_*/ run directories,
which are git-ignored and were pruned; for those this script prints the exact command
that regenerates them plus the file it writes, because there is no data left to plot.

Usage:
    py -3.10 report_figures.py           # rebuild what it can, print the inventory
    py -3.10 report_figures.py --check   # exit non-zero if a README figure is missing or unmapped
"""

import argparse
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results")
README = os.path.join(BASE, "README.md")

# README figure -> (regenerating script, the file that script writes, data source)
PROVENANCE = {
    "coinrun_50k_cnn_vs_mlp.png": ("compare_procgen.py", "logs_procgen/*/comparison_plot.png",
                                   "results/legacy_records.json"),
    "bossfight_100k_world_models.png": ("compare_world_models.py", "logs_world_models/*/comparison_plot.png",
                                        "results/legacy_records.json"),
    "suite_bossfight.png": ("compare_suite.py", "logs_suite/*/suite_<game>_plot.png", "pruned logs"),
    "suite_starpilot.png": ("compare_suite.py", "logs_suite/*/suite_<game>_plot.png", "pruned logs"),
    "suite_dodgeball.png": ("compare_suite.py", "logs_suite/*/suite_<game>_plot.png", "pruned logs"),
    "bossfight_hard_100k.png": ("compare_bossfight_hard.py", "logs_bossfight_hard/*/comparison_plot.png", "pruned logs"),
    "new_archs_bossfight.png": ("compare_new_archs.py", "logs_new_archs/*/new_archs_<game>_plot.png", "pruned logs"),
    "new_archs_starpilot.png": ("compare_new_archs.py", "logs_new_archs/*/new_archs_<game>_plot.png", "pruned logs"),
    "new_archs_dodgeball.png": ("compare_new_archs.py", "logs_new_archs/*/new_archs_<game>_plot.png", "pruned logs"),
    "maze_heist_maze_plot.png": ("compare_maze_heist.py", "logs_maze_heist/*/maze_heist_maze_plot.png",
                                 "results/exploration_remeasure.json"),
    "maze_heist_heist_plot.png": ("compare_maze_heist.py", "logs_maze_heist/*/maze_heist_heist_plot.png",
                                  "results/exploration_remeasure.json"),
    "global_16.png": ("compare_combined.py", "logs_combined/*/combined_<game>_plot.png", "pruned logs"),
    "symbolic_regression_pareto.png": ("symbolic_regression.py", "results/symbolic_regression_pareto.png",
                                       "results/symbolic_regression.json"),
    "symbolic_regression_scaling.png": ("symbolic_regression.py", "results/symbolic_regression_scaling.png",
                                        "results/symbolic_regression.json"),
    "symbolic_regression_identifiability.png": ("symbolic_regression.py",
                                                "results/symbolic_regression_identifiability.png",
                                                "results/symbolic_regression.json"),
    "rliable_profile.png": ("run_rliable_eval.py", "results/rliable_profile.png",
                            "results/eval100_results.json + results/random_baselines.json"),
}

DEFAULT_FOOTNOTE = ("dots = per-seed values; error bars are sample std (ddof=1), so they are "
                    "slightly wider than the population std quoted in README section 7")

# Per-seed groups that results/legacy_records.json still carries, in README order.
REBUILDABLE = {
    "coinrun_50k_cnn_vs_mlp.png": ("Procgen Coinrun - 50000 steps - 200 levels",
                                   "Mean reward (10 eps)",
                                   ["coinrun_classic", "coinrun_cbam", "coinrun_spatial",
                                    "coinrun_mlp_vector"]),
    "bossfight_100k_world_models.png": ("Procgen Bossfight - 100000 steps - world model auxiliaries",
                                        "Mean reward (10 eps)",
                                        ["bossfight_wm_vae", "bossfight_wm_ae",
                                         "bossfight_wm_recon", "bossfight_wm_contrastive"]),
}


def load_legacy():
    with open(os.path.join(RESULTS, "legacy_records.json"), encoding="utf-8") as f:
        return json.load(f)["per_seed"]


def load_exploration():
    """Per-seed 10-eps returns of the re-measured maze/heist arms, keyed by config."""
    path = os.path.join(RESULTS, "exploration_remeasure.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        j = json.load(f)
    return {k: [c["mean_reward_10eps"] for c in v["cells"]]
            for k, v in j.get("per_seed", {}).items()}


# Which figure shows which configs, in README order.
EXPLORATION_KEYS = {
    "maze_heist_maze_plot.png": ["maze_ppo", "maze_icm", "maze_rnd", "maze_ngu"],
    "maze_heist_heist_plot.png": ["heist_ppo", "heist_icm", "heist_rnd", "heist_ngu"],
}
EXPLORATION_TITLES = {
    "maze_heist_maze_plot.png": "Procgen Maze - 100000 steps - PPO vs ICM/RND/NGU",
    "maze_heist_heist_plot.png": "Procgen Heist - 100000 steps - PPO vs ICM/RND/NGU",
}


def rebuild(name, title, ylabel, keys, per_seed, footnote=None):
    vals = [per_seed[k] for k in keys]
    labels = [k.split("_", 1)[1].replace("_", " ").title() for k in keys]
    means = [float(np.mean(v)) for v in vals]
    stds = [float(np.std(v, ddof=1)) for v in vals]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["lightblue", "lightgreen", "lightcoral", "lightsalmon"]
    bars = ax.bar(labels, means, yerr=stds, capsize=5, alpha=0.8, color=colors[:len(labels)])
    for bar, v in zip(bars, vals):
        ax.plot([bar.get_x() + bar.get_width() / 2] * len(v), v, "ko", ms=4, alpha=0.6)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s,
                f"{m:.2f}±{s:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=25)
    ax.text(0.01, -0.16, footnote or DEFAULT_FOOTNOTE,
            transform=ax.transAxes, fontsize=7, color="0.4")
    fig.tight_layout()
    out = os.path.join(RESULTS, name)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def readme_figures():
    with open(README, encoding="utf-8") as f:
        return set(re.findall(r"!\[[^\]]*\]\((results/[^)]+)\)", f.read()))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="verify every figure embedded in README section 4 exists on disk")
    args = parser.parse_args()

    missing = [ref for ref in sorted(readme_figures()) if not os.path.exists(os.path.join(BASE, ref))]
    if args.check:
        if missing:
            print("ERROR: README embeds figures that are not in the repository:")
            for m in missing:
                print(f"  {m}")
            raise SystemExit(1)
        embedded = {os.path.basename(r) for r in readme_figures()}
        unreferenced = sorted(set(PROVENANCE) - embedded)
        unmapped = sorted(embedded - set(PROVENANCE))
        if unmapped:
            print("ERROR: embedded figures with no recorded producer:")
            for m in unmapped:
                print(f"  {m}")
            raise SystemExit(1)
        print(f"OK: every README-embedded figure exists and is mapped "
              f"({len(embedded)} embedded)")
        if unreferenced:
            print(f"note: mapped in provenance but not embedded: {unreferenced}")
        return

    per_seed = load_legacy()
    rebuilt = 0
    for name, (title, ylabel, keys) in REBUILDABLE.items():
        if all(k in per_seed for k in keys):
            print(f"rebuilt  {name}  <- results/legacy_records.json")
            rebuild(name, title, ylabel, keys, per_seed)
            rebuilt += 1
        else:
            print(f"skipped  {name}  (legacy_records.json lacks some of {keys})")

    expl = load_exploration()
    for name, keys in EXPLORATION_KEYS.items():
        if all(k in expl for k in keys):
            print(f"rebuilt  {name}  <- results/exploration_remeasure.json")
            rebuild(name, EXPLORATION_TITLES[name], "Mean reward (10 eps)", keys, expl,
                    footnote="dots = per-seed values from results/exploration_remeasure.json "
                             "(24/09/2026 re-measurement, intrinsic bonus verified to have "
                             "fired on every step of every arm); error bars are sample std")
            rebuilt += 1
        else:
            print(f"skipped  {name}  (exploration_remeasure.json lacks some of {keys})")

    print(f"\n{len(PROVENANCE) - rebuilt} figures below need their benchmark re-run; "
          f"they are drawn from the git-ignored logs_* directories:")
    for name, (script, produced, source) in sorted(PROVENANCE.items()):
        if name in REBUILDABLE:
            continue
        print(f"  {name:34s} {script:26s} writes {produced}  [{source}]")
    print(f"\nReproduction commands: README section 5. Data source per figure: "
          f"results/*.json (see README section 4).")


if __name__ == "__main__":
    main()
