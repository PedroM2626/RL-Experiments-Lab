"""
Offline scorecard analysis (Roadmap 6.1 items #3 and #4):
- 95% CI (Student's t, df=4) and Cohen's d over existing per-seed records
- AUC(reward, steps) from tensorboard logs of new_archs and maze_heist
No retraining. Output: results/scorecard.json

Per-seed data comes from two places: the training-run JSONs under the logs_*
directories (only present on the machine that ran the benchmarks), and
results/legacy_records.json for the runs whose raw logs were deleted. Re-running
this script without the logs therefore refuses to overwrite committed entries it
can no longer compute (see guard_no_data_loss).
"""
import argparse
import json
import os
import sys
import re
import glob
import numpy as np

T_CRIT_95_DF4 = 2.776  # two-tailed Student's t 95% critical value, n=5, df=4

LOG_SOURCES = [
    ('new_archs', 'logs_new_archs/*/comparison_results.json'),
    ('maze_heist', 'logs_maze_heist/*/comparison_results.json'),
]

def ci95(vals):
    vals = np.array(vals, dtype=float)
    m, s = vals.mean(), vals.std(ddof=1)
    half = T_CRIT_95_DF4 * s / np.sqrt(len(vals))
    return float(m), float(s), float(m - half), float(m + half)

def cohens_d(a, b):
    a, b = np.array(a, float), np.array(b, float)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na-1)*a.var(ddof=1) + (nb-1)*b.var(ddof=1)) / (na+nb-2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float('inf')

def load_legacy_records(base):
    """Per-seed returns for runs whose raw logs no longer exist (see README section 7)."""
    path = os.path.join(base, 'results', 'legacy_records.json')
    with open(path, encoding='utf-8') as f:
        return json.load(f)['per_seed']


def load_per_seed(base, logs_root):
    """Returns (per_seed_values, provenance_per_config)."""
    data, sources = {}, {}
    for tag, pattern in LOG_SOURCES:
        paths = sorted(glob.glob(os.path.join(logs_root, pattern)))
        if not paths:
            print(f"NOTE: no {tag} run JSONs under {logs_root} — the configs that came from "
                  f"them can only be reused from the committed results/scorecard.json")
            continue
        # a sweep split across processes writes one directory per subset, so cells are
        # merged by (config, seed) instead of one file replacing another wholesale.
        cells = {}
        for path in paths:
            with open(path, encoding='utf-8') as f:
                j = json.load(f)
            for k, v in j.items():
                if k.startswith('_'):
                    continue  # _protocol and friends, not a config
                for x in v:
                    if x.get('mean_reward') is None:
                        continue
                    prev = cells.get((k, x['seed']))
                    if prev is not None and abs(prev - x['mean_reward']) > 1e-9:
                        print(f"NOTE: {k} seed {x['seed']} measured twice "
                              f"({prev:.3f} then {x['mean_reward']:.3f}); keeping the later")
                    cells[(k, x['seed'])] = x['mean_reward']
                    sources[k] = f"{tag}: {os.path.relpath(path, logs_root)}"
        for (k, _seed) in sorted(cells):
            data.setdefault(k, []).append(cells[(k, _seed)])
    legacy = load_legacy_records(base)
    data.update({k: v for k, v in legacy.items() if k not in data})
    sources.update({k: 'results/legacy_records.json' for k in legacy if k in data})
    return data, sources


def guard_no_data_loss(out_path, results, force):
    """A rerun must not silently drop entries a published table depends on."""
    if force or not os.path.exists(out_path):
        return
    with open(out_path, encoding='utf-8') as f:
        committed = json.load(f)
    lost_ci = sorted(k for k in committed.get('ci_effect_size', {})
                     if not k.startswith('__') and k not in results['ci_effect_size'])
    lost_auc = sorted(k for k in committed.get('auc', {}) if k not in results['auc'])
    if lost_ci or lost_auc:
        if lost_ci:
            print(f"ERROR: rerunning would drop {len(lost_ci)} committed per-seed entries:\n  "
                  + "\n  ".join(lost_ci))
        if lost_auc:
            print(f"ERROR: rerunning would drop {len(lost_auc)} committed AUC entries "
                  f"(they need the tensorboard event files under logs_*).")
        print("README sections 3.8 and 3.9 cite these. Restore the logs_* directories "
              "(--logs_root) to recompute them, or pass --force to accept the reduced output.")
        sys.exit(1)

def tb_run_cells(log_dir, valid_keys):
    """Attribute each tensorboard run directory to a config key.

    Runs created by compare_maze_heist.py are named '<game>_<arm>_seed<n>[_<counter>]', so
    each curve carries its own config; SB3 appends the counter when a name repeats, which is
    what --resume does, so the highest counter for a cell is the run that actually finished.
    When no directory is named, this falls back to the legacy positional convention in which
    the k-th PPO_k directory was the k-th entry of the expected order — that assumption is
    wrong for a split or resumed sweep, which is why the naming exists.
    Returns ([(key, dir)], notes).
    """
    valid = set(valid_keys)
    named, unnamed = {}, []
    if not os.path.isdir(log_dir):
        return [], [f"{log_dir} does not exist"]
    for d in sorted(os.listdir(log_dir)):
        full = os.path.join(log_dir, d)
        if not os.path.isdir(full):
            continue
        m = re.fullmatch(r"(.+)_seed(\d+)(?:_(\d+))?", d)
        if m and m.group(1) in valid:
            counter = int(m.group(3) or 0)
            cell = (m.group(1), int(m.group(2)))
            if cell not in named or counter > named[cell][0]:
                named[cell] = (counter, full)
        else:
            unnamed.append(full)
    notes = []
    if named:
        if unnamed:
            notes.append(f"{len(unnamed)} tensorboard run(s) not attributable to a config and "
                         f"skipped: {[os.path.basename(u) for u in unnamed][:5]}")
        runs = [(key, full) for (key, _seed), (_c, full) in sorted(named.items())]
    else:
        notes.append("no named runs found, falling back to positional PPO_n attribution")
        runs = [(key, os.path.join(log_dir, f"PPO_{i}")) for i, key in enumerate(valid_keys, start=1)]
    return runs, notes


def auc_from_tb(log_dir, order, total=100000):
    """Tensorboard learning curves -> (key, auc_norm) pairs, normalised by the step budget."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    runs, notes = tb_run_cells(log_dir, order)
    for note in notes:
        print(f"NOTE: {log_dir}: {note}")
    out = []
    for key, d in runs:
        if not os.path.isdir(d): continue
        try:
            ea = EventAccumulator(d); ea.Reload()
            tags = ea.Tags().get('scalars', [])
            tag = 'rollout/ep_rew_mean' if 'rollout/ep_rew_mean' in tags else (tags[0] if tags else None)
            if tag is None: continue
            evs = ea.Scalars(tag)
            xs = [e.step for e in evs]; ys = [e.value for e in evs]
            auc = float(np.trapz(ys, xs))
            out.append((key, round(auc / total, 3)))
        except Exception as e:
            print(f"NOTE: AUC unavailable for {d}: {type(e).__name__}: {e}")
    return out

def main():
    parser = argparse.ArgumentParser(description='Rebuild results/scorecard.json from per-seed records')
    parser.add_argument('--logs_root', default=None,
                        help='directory holding logs_new_archs/ and logs_maze_heist/ (default: repository root)')
    parser.add_argument('--out', default='results/scorecard.json')
    parser.add_argument('--force', action='store_true',
                        help='accept an output that drops committed per-seed entries')
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    logs_root = args.logs_root or base
    out_path = args.out if os.path.isabs(args.out) else os.path.join(base, args.out)

    per_seed, sources = load_per_seed(base, logs_root)
    results = {'ci_effect_size': {}, 'auc': {}}

    # 95% CI per config
    for k, vals in per_seed.items():
        m, s, lo, hi = ci95(vals)
        results['ci_effect_size'][k] = {'mean': round(m, 3), 'std': round(s, 3), 'ci95': [round(lo, 3), round(hi, 3)]}

    # Cohen's d top-1 vs top-2 per game
    games = {}
    for k in per_seed:
        game, cfg = k.split('_', 1)
        games.setdefault(game, {})[cfg] = per_seed[k]
    for game, cfgs in games.items():
        ranked = sorted(cfgs.items(), key=lambda kv: -np.mean(kv[1]))
        if len(ranked) >= 2:
            (c1, v1), (c2, v2) = ranked[0], ranked[1]
            results['ci_effect_size'].setdefault(f"__toppair_{game}", {})
            results['ci_effect_size'][f"__toppair_{game}"] = {
                'top1': c1, 'top2': c2, 'cohens_d': round(cohens_d(v1, v2), 3),
                'ci_overlap': not (min(ci95(v1)[2:]) > max(ci95(v2)[2:]) or min(ci95(v2)[2:]) > max(ci95(v1)[2:]))
            }

    # AUC from tensorboard logs (new_archs and maze_heist)
    na_order = [f"{g}_{a}" for g in ['bossfight', 'starpilot', 'dodgeball']
                for a in ['impala', 'impoola', 'lstm_attention', 'vit', 'resnet18']
                for _ in range(5)]
    mh_order = [f"{g}_{w}" for g in ['maze', 'heist'] for w in ['ppo', 'icm', 'rnd', 'ngu'] for _ in range(5)]
    na = auc_from_tb(os.path.join(logs_root, 'logs_new_archs'), na_order)
    mh = auc_from_tb(os.path.join(logs_root, 'logs_maze_heist'), mh_order)
    # aggregate per config (5 seeds mean)
    for src in (na, mh):
        agg = {}
        for k, auc_norm in src:
            agg.setdefault(k, []).append(auc_norm)
        for k, lst in agg.items():
            results['auc'][k] = {'auc_norm_mean': round(float(np.mean(lst)), 3), 'auc_norm_std': round(float(np.std(lst, ddof=1)), 3) if len(lst) > 1 else 0.0, 'n_seeds': len(lst)}

    results['_meta'] = {
        'per_seed_provenance': sources,
        'logs_root': logs_root,
        'auc_source': 'tensorboard rollout/ep_rew_mean under logs_*',
        'auc_available': bool(results['auc']),
    }

    guard_no_data_loss(out_path, results, args.force)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}: {len(results['ci_effect_size'])} per-seed entries, "
          f"{len(results['auc'])} AUC entries")

if __name__ == '__main__':
    main()
