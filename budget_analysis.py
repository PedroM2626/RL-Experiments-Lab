"""Analysis of budget scaling (Section 3.13): curves 100k -> 250k -> 500k.
100k baseline comes from eval100_results.json (same seeds 42-44, identical protocol)."""
import json
import numpy as np

bud = json.load(open('results/budget_results.json', encoding='utf-8'))
ev100 = json.load(open('results/eval100_results.json', encoding='utf-8'))

SEEDS = [42, 43, 44]
# mapping: (game, scaling config) -> key prefix in eval100 (naming differs slightly)
MAP100 = {
    ('starpilot', 'resnet18'): 'starpilot_resnet18',
    ('starpilot', 'mlp_vector'): 'starpilot_cnn_mlp_vector',
    ('dodgeball', 'resnet18'): 'dodgeball_resnet18',
    ('dodgeball', 'mlp_vector'): 'dodgeball_cnn_mlp_vector',
}

def vals_from_budget(game, cfg, b, key='stoch_unseen'):
    return [bud[f'{game}_{cfg}_b{b//1000}k_seed{s}'][key] for s in SEEDS]

def vals_from_100(game, cfg, key='stoch_unseen'):
    return [ev100[f'{MAP100[(game, cfg)]}_seed{s}'][key] for s in SEEDS]

print(f"{'curve':30s} {'100k':>12s} {'250k':>12s} {'500k':>12s}   verdict")
print('-' * 84)
summary = {}
for game in ['starpilot', 'dodgeball']:
    for cfg in ['resnet18', 'mlp_vector']:
        v100 = vals_from_100(game, cfg)
        v250 = vals_from_budget(game, cfg, 250000)
        v500 = vals_from_budget(game, cfg, 500000)
        m = [np.mean(v100), np.mean(v250), np.mean(v500)]
        sd = [np.std(v100), np.std(v250), np.std(v500)]
        # simple verdict: gain 250k->500k vs combined noise
        gain = m[2] - m[1]
        noise = np.sqrt(sd[1]**2 + sd[2]**2)
        verdict = ('still rising' if gain > noise else
                   'stagnant' if abs(gain) <= noise else 'dropping')
        key = f'{game}_{cfg}'
        summary[key] = {b: {'mean': round(mm, 2), 'std': round(ss, 2)}
                        for b, mm, ss in zip([100000, 250000, 500000], m, sd)}
        summary[key]['verdict'] = verdict
        row = '  '.join(f'{mm:.2f}±{ss:.2f}' for mm, ss in zip(m, sd))
        print(f"{key:30s} {row}   {verdict}")

print('\nGEN GAP vs budget (stoch_train - stoch_unseen):')
for game in ['starpilot', 'dodgeball']:
    for cfg in ['resnet18', 'mlp_vector']:
        g100 = np.mean([ev100[f'{MAP100[(game, cfg)]}_seed{s}']['gen_gap'] for s in SEEDS])
        g250 = np.mean([bud[f'{game}_{cfg}_b250k_seed{s}']['gen_gap'] for s in SEEDS])
        g500 = np.mean([bud[f'{game}_{cfg}_b500k_seed{s}']['gen_gap'] for s in SEEDS])
        print(f"  {game}_{cfg:11s} {g100:+.2f} -> {g250:+.2f} -> {g500:+.2f}")

json.dump(summary, open('results/budget_analysis.json', 'w'), indent=2)
print('\nSaved: results/budget_analysis.json')
