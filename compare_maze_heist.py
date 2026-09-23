"""
Maze+Heist PPO vs ICM vs RND vs NGU — 2 games × 4 configs × 5 seeds × 100k
ICM: forward+inverse, RND: random target, NGU: RND+episodic
The bonus wrappers live in models/bonuses.py; each run asserts the mechanism actually
fired, so an arm cannot silently degrade to plain PPO.
The grid is ~7h, so comparison_results.json is rewritten after every seed and --resume
reuses an existing run directory: an interrupted sweep keeps its completed cells instead
of losing everything, and each cell records the bonus bookkeeping that proves the arm
explored (the 23/09/2026 defect was exactly a run that looked fine and had no bonus).
"""
import os, json, argparse, time
from datetime import datetime
import numpy as np, torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from procgen_wrapper import make_procgen_env
from models.sb3_extractors import ClassicCNNExtractor
from models.bonuses import ICMWrapper, RNDWrapper, NGUWrapper

WRAPPERS = {'icm': ICMWrapper, 'rnd': RNDWrapper, 'ngu': NGUWrapper}


def train_one(game, wrapper, timesteps, seed, log_dir, device, eval_eps, run_name):
    def make_env():
        env=make_procgen_env(game, num_levels=200, distribution_mode='easy', seed=seed, vector=False)
        if wrapper: env=WRAPPERS[wrapper](env)
        return Monitor(env)
    vec=DummyVecEnv([make_env])
    eval_env=DummyVecEnv([lambda: Monitor(make_procgen_env(game, num_levels=0, distribution_mode='easy', seed=seed+1000, vector=False))])
    model=PPO("CnnPolicy", vec, verbose=0, learning_rate=3e-4, n_steps=256, batch_size=64, n_epochs=3, gamma=0.99, gae_lambda=0.95, clip_range=0.2, seed=seed, device=device, policy_kwargs={"features_extractor_class": ClassicCNNExtractor, "features_extractor_kwargs": dict(features_dim=512)}, tensorboard_log=log_dir)
    t0=time.time()
    # tb_log_name carries the config because scorecard_analysis.py must attribute each
    # learning curve to the arm that produced it; the default "PPO_n" numbering only
    # identifies the config by creation order, which breaks on --resume and on
    # parallel processes and would silently mislabel the section 3.9 AUC table.
    model.learn(total_timesteps=timesteps, tb_log_name=run_name)
    bonus=None
    if wrapper:
        # env_method reaches the innermost wrapper through Monitor's attribute proxying
        st = vec.env_method('stats')[0]
        print(f"[{game}_{wrapper}] intrinsic bonus applied on {st['bonus_applied']}/{st['steps']} "
              f"steps, mean={st['intrinsic_mean']:.5f}")
        if st['bonus_applied'] == 0:
            raise RuntimeError(f"{wrapper} arm added no intrinsic reward at all — the run would "
                               f"be an unlabelled PPO run (stats={st})")
        bonus=st
    mean,std=evaluate_policy(model, eval_env, n_eval_episodes=eval_eps, deterministic=False)
    vec.close(); eval_env.close()
    return float(mean), float(std), model, bonus, round(time.time()-t0, 1)


def summarize(results):
    stats={}
    for k,v in results.items():
        rewards=[x['mean_reward'] for x in v if x.get('mean_reward') is not None]
        stats[k]={'mean':float(np.mean(rewards)),'std':float(np.std(rewards)),'min':float(np.min(rewards)),'max':float(np.max(rewards)),'n':len(rewards)} if rewards else None
    return stats


def save_run(comp_dir, results, protocol):
    os.makedirs(comp_dir, exist_ok=True)
    # a split or resumed run must describe itself, not the flags of whichever process wrote last
    meta=dict(protocol)
    meta['cells_present']={k: len([c for c in v if c.get('mean_reward') is not None]) for k, v in results.items()}
    payload=dict(results); payload['_protocol']=meta
    with open(os.path.join(comp_dir,'comparison_results.json'),'w') as f: json.dump(payload,f,indent=2)
    with open(os.path.join(comp_dir,'statistics.json'),'w') as f: json.dump(summarize(results),f,indent=2)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=100000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42,43,44,45,46])
    parser.add_argument('--games', type=str, nargs='+', default=['maze','heist'])
    parser.add_argument('--log_dir', type=str, default='./logs_maze_heist')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--eval_eps', type=int, default=10)
    parser.add_argument('--arms', type=str, nargs='+', default=['ppo','icm','rnd','ngu'],
                        choices=['ppo','icm','rnd','ngu'],
                        help='subset of arms to run, so the grid can be split across processes')
    parser.add_argument('--resume', type=str, default=None,
                        help='existing run directory to continue (its completed cells are kept and skipped)')
    args=parser.parse_args()
    for w in args.arms:
        if w not in WRAPPERS and w!='ppo':
            raise SystemExit(f"--arms: unknown arm {w}")
    device='cuda' if (args.device=='auto' and torch.cuda.is_available()) else args.device if args.device!='auto' else 'cpu'
    print(f"Device: {device} torch {torch.__version__} MAZE+HEIST ICM/RND/NGU")
    os.makedirs(args.log_dir, exist_ok=True)
    if args.resume:
        comp_dir=args.resume if os.path.isabs(args.resume) else os.path.join(args.log_dir, args.resume)
        if not os.path.isdir(comp_dir):
            raise SystemExit(f"--resume: {comp_dir} does not exist")
    else:
        ts=datetime.now().strftime("%Y%m%d_%H%M%S")
        comp_dir=os.path.join(args.log_dir, f"maze_heist_{'_'.join(args.games)}_{'_'.join(args.arms)}_{ts}")
    os.makedirs(comp_dir, exist_ok=True)
    protocol={'timesteps': args.timesteps, 'seeds': args.seeds, 'games': args.games,
              'arms': args.arms, 'eval_episodes': args.eval_eps,
              'eval_levels': 'unseen (num_levels=0, seed+1000)',
              'train_levels': 200, 'distribution_mode': 'easy',
              'intrinsic_beta': 0.01, 'torch': torch.__version__, 'device': device}
    results={}
    done=set()
    prior=os.path.join(comp_dir,'comparison_results.json')
    if os.path.exists(prior):
        with open(prior, encoding='utf-8') as f:
            results={k: v for k, v in json.load(f).items() if not k.startswith('_')}
        for gk, cells in results.items():
            for c in cells:
                if c.get('mean_reward') is not None:
                    done.add((gk, c['seed']))
        print(f"Resuming {comp_dir}: {len(done)} cells already measured")
    for game in args.games:
        for w in args.arms:
            gk=f"{game}_{w}"
            results.setdefault(gk, [])
            results[gk]=[c for c in results[gk] if c['seed'] in args.seeds]
            for seed in args.seeds:
                if (gk, seed) in done:
                    print(f"\n{gk} seed {seed}: already measured, skipped")
                    continue
                print(f"\n{'='*60}\n{gk} seed {seed} {game}\n{'='*60}")
                results[gk]=[c for c in results[gk] if c['seed']!=seed]
                try:
                    mean,std,model,bonus,secs=train_one(game, w if w!='ppo' else None, args.timesteps, seed, args.log_dir, device, args.eval_eps, f"{gk}_seed{seed}")
                    print(f"{gk} seed {seed}: {mean:.2f} +/- {std:.2f} in {secs}s")
                    results[gk].append({'seed': seed, 'mean_reward': mean, 'std_reward': std,
                                        'seconds': secs, 'bonus': bonus})
                    try: model.save(os.path.join(comp_dir, f"{gk}_seed{seed}.zip"))
                    except Exception as e: print(f"WARNING: checkpoint save failed for {gk}_seed{seed}: {e}")
                except Exception as e:
                    import traceback; traceback.print_exc()
                    results[gk].append({'seed': seed, 'mean_reward': None, 'error': str(e)})
                results[gk].sort(key=lambda c: c['seed'])
                save_run(comp_dir, results, protocol)
    stats=summarize(results)
    try:
        import matplotlib.pyplot as plt
        for game in args.games:
            keys=[k for k in stats if k.startswith(game+'_')]
            means=[stats[k]['mean'] if stats[k] else 0 for k in keys]
            stds=[stats[k]['std'] if stats[k] else 0 for k in keys]
            plt.figure(figsize=(10,6)); plt.bar(keys, means, yerr=stds, capsize=4, alpha=0.8)
            plt.xticks(rotation=20, ha='right'); plt.ylabel('Mean Reward'); plt.title(f"{game} - PPO vs ICM/RND/NGU - {args.timesteps} steps 5 seeds")
            plt.tight_layout(); plt.savefig(os.path.join(comp_dir, f"maze_heist_{game}_plot.png"), dpi=150, bbox_inches='tight'); plt.close()
        print(f"Plots saved in {comp_dir}")
    except Exception as e: print(f"Plot error: {e}")
    print(f"\nResults in {comp_dir}\n"+json.dumps(stats,indent=2))

if __name__=='__main__': main()
