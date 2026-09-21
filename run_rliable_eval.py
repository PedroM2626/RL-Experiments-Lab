"""Executes rigorous RLiable statistical evaluation on the 100-episode definitive evaluation dataset.

Loads results/eval100_results.json, applies per-game min-max normalization,
and calculates IQM, Trimmed Mean, Stratified Bootstrap 95% CIs, and
Probability of Improvement across architectural families.

Saves:
- results/rliable_scorecard.json
- results/rliable_profile.png
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

from rliable_metrics import (
    compute_iqm,
    compute_trimmed_mean,
    stratified_bootstrap_ci,
    compute_performance_profile,
    compute_probability_of_improvement,
)


def load_eval100_data(filepath="results/eval100_results.json"):
    with open(filepath, "r", encoding="utf-8") as f:
        raw = json.load(f)
    
    # game_arch -> list of seed returns
    game_arch_scores = {}
    for key, val in raw.items():
        if "error" in val:
            continue
        # key format: {game}_{arch}_seed{seed}
        parts = key.rsplit("_seed", 1)
        if len(parts) != 2:
            continue
        game_arch = parts[0]
        score = val.get("stoch_unseen", 0.0)
        game_arch_scores.setdefault(game_arch, []).append(score)
        
    return game_arch_scores


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    eval100_path = os.path.join(base_dir, "results", "eval100_results.json")
    game_arch_scores = load_eval100_data(eval100_path)

    # Decompose into games and architectures
    # Format: data[arch][game] = np.array(seed_scores)
    all_games = set()
    all_archs = set()
    data = {}

    for ga, scores in game_arch_scores.items():
        # First token is the game name (e.g. bossfight, starpilot, dodgeball, maze, heist)
        game, arch = ga.split("_", 1)
        all_games.add(game)
        all_archs.add(arch)
        data.setdefault(arch, {})[game] = np.array(scores, dtype=np.float64)

    all_games = sorted(all_games)
    all_archs = sorted(all_archs)

    # 1. Compute per-game empirical bounds (min and max across all architectures and seeds)
    game_min = {}
    game_max = {}
    for g in all_games:
        scores_g = []
        for a in all_archs:
            if g in data.get(a, {}):
                scores_g.extend(data[a][g])
        game_min[g] = float(np.min(scores_g))
        game_max[g] = float(np.max(scores_g))

    print("=" * 80)
    print("PER-GAME MIN-MAX NORMALIZATION BOUNDS")
    print("=" * 80)
    for g in all_games:
        print(f"  {g:12s} | min: {game_min[g]:6.2f} | max: {game_max[g]:6.2f} (span: {game_max[g]-game_min[g]:.2f})")

    # 2. Normalize scores per game: z = (score - min) / (max - min + 1e-8)
    norm_data = {}
    for a in all_archs:
        norm_data[a] = {}
        for g, scores in data[a].items():
            span = game_max[g] - game_min[g]
            span = span if span > 1e-6 else 1.0
            norm_data[a][g] = (scores - game_min[g]) / span

    # 3. Stratified Bootstrap Statistics across games
    results = {
        "normalization_bounds": {g: {"min": game_min[g], "max": game_max[g]} for g in all_games},
        "architectures": {},
        "probability_of_improvement": {},
    }

    # Focus on the 3-game suite architectures (evaluated across bossfight, starpilot, dodgeball)
    suite_archs = [a for a in all_archs if set(["bossfight", "starpilot", "dodgeball"]).issubset(data[a].keys())]

    print("\n" + "=" * 80)
    print("SUITE ARCHITECTURES — NORMALIZED METRICS (3 GAMES: B/S/D)")
    print("=" * 80)
    print(f"  {'architecture':22s} | {'IQM (95% CI)':22s} | {'Trimmed 5% (95% CI)':24s} | {'Mean Norm':9s}")
    print("-" * 80)

    suite_eval = []
    for a in suite_archs:
        # Subdict for the 3 suite games
        task_subset = {g: norm_data[a][g] for g in ["bossfight", "starpilot", "dodgeball"]}
        iqm_est, (iqm_lo, iqm_hi) = stratified_bootstrap_ci(task_subset, metric_fn=compute_iqm)
        trim_est, (trim_lo, trim_hi) = stratified_bootstrap_ci(task_subset, metric_fn=compute_trimmed_mean)
        all_norm = np.concatenate(list(task_subset.values()))
        mean_norm = float(np.mean(all_norm))

        results["architectures"][a] = {
            "iqm": round(iqm_est, 4),
            "iqm_ci95": [round(iqm_lo, 4), round(iqm_hi, 4)],
            "trimmed_mean": round(trim_est, 4),
            "trimmed_ci95": [round(trim_lo, 4), round(trim_hi, 4)],
            "mean_normalized": round(mean_norm, 4),
        }
        suite_eval.append((a, iqm_est, iqm_lo, iqm_hi, trim_est, trim_lo, trim_hi, mean_norm))

    # Sort by normalized IQM
    suite_eval.sort(key=lambda x: -x[1])

    for a, iqm, iqm_lo, iqm_hi, tr, tr_lo, tr_hi, mn in suite_eval:
        iqm_str = f"{iqm:.3f} [{iqm_lo:.3f}, {iqm_hi:.3f}]"
        tr_str = f"{tr:.3f} [{tr_lo:.3f}, {tr_hi:.3f}]"
        print(f"  {a:22s} | {iqm_str:22s} | {tr_str:24s} | {mn:6.3f}")

    # 4. Pairwise Probability of Improvement between top 5
    top5_archs = [x[0] for x in suite_eval[:5]]
    print("\n" + "=" * 80)
    print("PAIRWISE PROBABILITY OF IMPROVEMENT: P(Row > Column) with 95% Bootstrap CI")
    print("=" * 80)
    for a1 in top5_archs:
        for a2 in top5_archs:
            if a1 == a2:
                continue
            t_a1 = {g: norm_data[a1][g] for g in ["bossfight", "starpilot", "dodgeball"]}
            t_a2 = {g: norm_data[a2][g] for g in ["bossfight", "starpilot", "dodgeball"]}
            prob, (p_lo, p_hi) = compute_probability_of_improvement(t_a1, t_a2)
            key_pair = f"{a1}_vs_{a2}"
            results["probability_of_improvement"][key_pair] = {
                "p": round(prob, 3),
                "ci95": [round(p_lo, 3), round(p_hi, 3)],
            }
            print(f"  P({a1:18s} > {a2:18s}) = {prob:5.3f} [{p_lo:5.3f}, {p_hi:5.3f}]")

    # 5. Performance Profiles
    tau_grid = np.linspace(0.0, 1.0, 101)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # Plot IQM bar chart with bootstrap error bars
    names = [x[0] for x in suite_eval]
    iqms = [x[1] for x in suite_eval]
    yerr_lo = [x[1] - x[2] for x in suite_eval]
    yerr_hi = [x[3] - x[1] for x in suite_eval]

    y_pos = np.arange(len(names))
    ax1.barh(y_pos, iqms, xerr=[yerr_lo, yerr_hi], align="center", alpha=0.85, color="#1f77b4", ecolor="black", capsize=3)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=8)
    ax1.invert_yaxis()  # top-down
    ax1.set_xlabel("Normalized Interquartile Mean (IQM)")
    ax1.set_title("Aggregated Score (IQM with 95% Stratified Bootstrap CI)")
    ax1.grid(True, linestyle="--", alpha=0.4, axis="x")

    # Plot Performance Profiles for top-5
    for a in top5_archs:
        pooled = np.concatenate([norm_data[a][g] for g in ["bossfight", "starpilot", "dodgeball"]])
        taus, probs = compute_performance_profile(pooled, tau_grid)
        ax2.plot(taus, probs, label=a, linewidth=2)

    ax2.set_xlabel(r"Normalized Score Threshold ($\tau$)")
    ax2.set_ylabel(r"Fraction of Runs with Score $\geq \tau$")
    ax2.set_title("Performance Profiles (Top 5 Suite Architectures)")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    out_img = os.path.join(base_dir, "results", "rliable_profile.png")
    plt.savefig(out_img)
    plt.close()
    print(f"\nSaved visualization: {out_img}")

    out_json = os.path.join(base_dir, "results", "rliable_scorecard.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved JSON scorecard: {out_json}")


if __name__ == "__main__":
    main()
