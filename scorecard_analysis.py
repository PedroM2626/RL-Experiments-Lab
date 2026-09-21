"""
Offline scorecard analysis (Roadmap 6.1 items #3 and #4):
- 95% CI (Student's t, df=4) and Cohen's d over existing per-seed records
- AUC(reward, steps) from tensorboard logs of new_archs and maze_heist
No retraining. Output: results/scorecard.json
"""
import os, json, glob
import numpy as np

T_CRIT_95_DF4 = 2.776  # two-tailed Student's t 95% critical value, n=5, df=4

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

def load_per_seed():
    data = {}
    base = os.path.dirname(os.path.abspath(__file__))
    # new benchmarks: JSON on disk
    for tag, p in [
        ('new_archs', 'logs_new_archs/new_archs_bossfight_starpilot_dodgeball_20260828_134545/comparison_results.json'),
        ('maze_heist', 'logs_maze_heist/maze_heist_maze_heist_20260829_014802/comparison_results.json'),
    ]:
        with open(os.path.join(base, p)) as f:
            j = json.load(f)
        for k, v in j.items():
            data[k] = [x['mean_reward'] for x in v if x['mean_reward'] is not None]
    # legacy benchmarks: per-seed preserved in README Section 7 (raw logs deleted)
    data.update({
        'coinrun_classic':        [7.0, 6.0, 7.0, 9.0, 9.0],
        'coinrun_cbam':           [8.0, 8.0, 8.0, 8.0, 8.0],
        'coinrun_spatial':        [6.0, 0.0, 7.0, 9.0, 9.0],
        'coinrun_mlp_vector':     [7.0, 7.0, 8.0, 9.0, 9.0],
        'bossfight_wm_vae':       [0.0, 0.0, 0.1, 0.4, 0.4],
        'bossfight_wm_ae':        [0.0, 0.0, 0.1, 0.1, 1.3],
        'bossfight_wm_recon':     [0.0, 0.0, 0.0, 0.0, 0.1],
        'bossfight_wm_contrastive':[0.0, 0.0, 0.0, 0.2, 1.6],
    })
    return data

def auc_from_tb(log_dir, order, total=100000):
    """order: list of keys in order of PPO_n creation; returns pairs (key, auc_norm)"""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    out = []
    for i, key in enumerate(order, start=1):
        d = os.path.join(log_dir, f"PPO_{i}")
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
        except Exception:
            pass
    return out

def main():
    base = os.path.dirname(os.path.abspath(__file__))
    per_seed = load_per_seed()
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
    na = auc_from_tb(os.path.join(base, 'logs_new_archs'), na_order)
    mh = auc_from_tb(os.path.join(base, 'logs_maze_heist'), mh_order)
    # aggregate per config (5 seeds mean)
    for src in (na, mh):
        agg = {}
        for k, auc_norm in src:
            agg.setdefault(k, []).append(auc_norm)
        for k, lst in agg.items():
            results['auc'][k] = {'auc_norm_mean': round(float(np.mean(lst)), 3), 'auc_norm_std': round(float(np.std(lst, ddof=1)), 3) if len(lst) > 1 else 0.0, 'n_seeds': len(lst)}

    os.makedirs(os.path.join(base, 'results'), exist_ok=True)
    with open(os.path.join(base, 'results', 'scorecard.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
