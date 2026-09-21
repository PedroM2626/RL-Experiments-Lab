"""
INDEPENDENT Benchmark: Value-based vs Policy-based (Section 12 of README).
- Value: DQN + QR-DQN (off-policy, replay buffer)
- Policy: PPO + A2C (on-policy, policy gradient)
- 3 games × 4 algorithms × 5 seeds × 100k steps = 60 runs
- Identical architecture for all: CnnPolicy (NatureCNN, features_dim 512)
- Built-in definitive eval: 100 eps unseen stoch + 100 det (seed+1000) + 15 train
- Hyperparameters: PPO = study defaults; A2C/DQN/QR-DQN = SB3 defaults with documented
  adaptations for small budgets (100k buffer due to RAM, scaled learning_starts/exploration)
- Output: logs_algo/algo_zips + results/algo_families_results.json (incremental, error retry)
"""
import os, json, argparse
import numpy as np, torch
from stable_baselines3 import DQN, PPO, A2C
from sb3_contrib import QRDQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from procgen_wrapper import make_procgen_env

def make_algo(name, env, seed, device, log_dir, value_lr=1e-4):
    if name == 'ppo':   # identical hyperparameters to main study
        return PPO("CnnPolicy", env, verbose=0, learning_rate=3e-4, n_steps=256, batch_size=64,
                   n_epochs=3, gamma=0.99, gae_lambda=0.95, clip_range=0.2, seed=seed,
                   device=device, tensorboard_log=log_dir)
    if name == 'a2c':   # SB3 default with lr aligned to study
        return A2C("CnnPolicy", env, verbose=0, learning_rate=3e-4, n_steps=128, gamma=0.99,
                   seed=seed, device=device, tensorboard_log=log_dir)
    common = dict(verbose=0, buffer_size=100_000,        # 100k frames uint8 ~= 1.2 GB (default 1M exceeds RAM)
                  learning_starts=5000, exploration_fraction=0.25, batch_size=64,
                  gamma=0.99, train_freq=4, gradient_steps=1, target_update_interval=500,
                  seed=seed, device=device, tensorboard_log=log_dir)
    if name == 'dqn':
        return DQN("CnnPolicy", env, learning_rate=value_lr, **common)   # DQN default lr 1e-4
    if name == 'qrdqn':
        return QRDQN("CnnPolicy", env, learning_rate=value_lr, policy_kwargs=dict(n_quantiles=200), **common)
    raise ValueError(name)

def eval_model(model, game, seed, n_unseen=100, n_train=15):
    unseen = DummyVecEnv([lambda: Monitor(make_procgen_env(game, num_levels=0, distribution_mode='easy', seed=seed+1000, vector=False))])
    train = DummyVecEnv([lambda: Monitor(make_procgen_env(game, num_levels=200, distribution_mode='easy', seed=seed, vector=False))])
    m_st, _ = evaluate_policy(model, unseen, n_eval_episodes=n_unseen, deterministic=False)
    m_dt, _ = evaluate_policy(model, unseen, n_eval_episodes=n_unseen, deterministic=True)
    m_tr, _ = evaluate_policy(model, train, n_eval_episodes=n_train, deterministic=False)
    unseen.close(); train.close()
    return float(m_st), float(m_dt), float(m_tr)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', nargs='+', default=['starpilot', 'dodgeball', 'bossfight'])
    parser.add_argument('--algos', nargs='+', default=['ppo', 'a2c', 'dqn', 'qrdqn'])
    parser.add_argument('--timesteps', type=int, default=100000)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument('--log_dir', default='./logs_algo')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--out', default='results/algo_families_results.json')
    args = parser.parse_args()
    device = 'cuda' if (args.device == 'auto' and torch.cuda.is_available()) else ('cpu' if args.device == 'auto' else args.device)

    base = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(base, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    zip_dir = os.path.join(base, 'logs_algo', 'algo_zips')
    os.makedirs(zip_dir, exist_ok=True)
    results = {}
    if os.path.exists(out_path):
        with open(out_path) as f: results = json.load(f)

    jobs = [(g, a, s) for g in args.games for a in args.algos for s in args.seeds]
    print(f"{len(jobs)} jobs ({args.games} × {args.algos} × {args.seeds}), device={device}, {args.timesteps} steps")
    for i, (game, algo, seed) in enumerate(jobs):
        name = f"{game}_{algo}_seed{seed}"
        if name in results and 'error' not in results[name]: continue
        print(f"\n[{i+1}/{len(jobs)}] {name}")
        try:
            vec = DummyVecEnv([lambda: Monitor(make_procgen_env(game, num_levels=200, distribution_mode='easy', seed=seed, vector=False))])
            model = make_algo(algo, vec, seed, device, args.log_dir)
            model.learn(total_timesteps=args.timesteps)
            vec.close()
            m_st, m_dt, m_tr = eval_model(model, game, seed)
            try: model.save(os.path.join(zip_dir, f"{name}.zip"))
            except Exception as e: print(f"  zip save failed: {e}")
            results[name] = {'stoch_unseen': round(m_st, 3), 'det_unseen': round(m_dt, 3),
                             'stoch_train': round(m_tr, 3), 'gen_gap': round(m_tr - m_st, 3),
                             'n_unseen': 100, 'n_train': 15}
            print(f"  stoch={m_st:.2f} det={m_dt:.2f} train={m_tr:.2f} gap={m_tr-m_st:+.2f}")
        except Exception as e:
            import traceback; traceback.print_exc()
            results[name] = {'error': str(e)}
        with open(out_path, 'w') as f: json.dump(results, f, indent=2)
    print(f"Completed: {out_path}")

if __name__ == '__main__':
    main()
