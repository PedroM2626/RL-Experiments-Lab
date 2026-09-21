import os
import sys
import torch
import numpy as np
import argparse
from datetime import datetime
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

from env_setup import create_env
from sac_trainer import SACTrainer
from models import ClassicCNNActor, ClassicCritic, AttentionCNNActor, AttentionCritic


def train_experiment(
    architecture='classic',
    num_steps=50000,
    seed=42,
    log_dir='./logs',
    use_cbam=True,
    **sac_kwargs
):
    """
    Executes a training experiment.
    """
    # Configure seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create environment
    env = create_env(frame_stack=4, frame_size=(64, 64))
    
    # Create networks based on architecture
    if architecture == 'classic':
        print("Using Classic CNN")
        actor = ClassicCNNActor(action_dim=3, feature_dim=512)
        critic = ClassicCritic(action_dim=3, feature_dim=512)
        exp_name = 'classic_cnn'
    elif architecture == 'attention':
        print(f"Using CNN with Spatial Attention (CBAM={use_cbam})")
        actor = AttentionCNNActor(action_dim=3, feature_dim=512, use_cbam=use_cbam)
        critic = AttentionCritic(action_dim=3, feature_dim=512, use_cbam=use_cbam)
        exp_name = f'attention_cnn_cbam_{use_cbam}'
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_log_dir = os.path.join(log_dir, f"{exp_name}_{timestamp}")
    
    # Create trainer
    trainer = SACTrainer(
        actor=actor,
        critic=critic,
        env=env,
        log_dir=experiment_log_dir,
        **sac_kwargs
    )
    
    # Save configuration
    config = {
        'architecture': architecture,
        'use_cbam': use_cbam,
        'num_steps': num_steps,
        'seed': seed,
        'sac_kwargs': sac_kwargs,
        'timestamp': timestamp
    }
    
    config_path = os.path.join(experiment_log_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration saved to {config_path}")
    print(f"Logs will be saved to {experiment_log_dir}")
    
    # Train
    trainer.train(
        num_steps=num_steps,
        eval_frequency=5000,
        save_frequency=10000
    )
    
    # Final evaluation
    final_reward = trainer.evaluate(num_episodes=10, deterministic=True)
    print(f"\nFinal reward (10 episodes): {final_reward:.2f}")
    
    # Save final results (convert to native float)
    results = {
        'final_reward': _to_float(final_reward),
        'total_episodes': int(trainer.episode_count),
        'total_steps': int(trainer.step_count)
    }
    
    results_path = os.path.join(experiment_log_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    env.close()
    
    return experiment_log_dir, results


def run_comparison(
    num_steps=50000,
    seeds=[42, 43, 44],
    log_dir='./logs',
    **sac_kwargs
):
    """
    Runs comparison between architectures across multiple random seeds.
    """
    architectures = [
        {'name': 'classic', 'use_cbam': None},
        {'name': 'attention', 'use_cbam': True},
        {'name': 'attention', 'use_cbam': False}
    ]
    
    all_results = {}
    
    for arch_config in architectures:
        arch_name = arch_config['name']
        use_cbam = arch_config['use_cbam']
        
        if arch_name == 'classic':
            key = 'classic'
        else:
            key = f'attention_cbam_{use_cbam}'
        
        all_results[key] = []
        
        print(f"\n{'='*60}")
        print(f"Training {key}")
        print(f"{'='*60}")
        
        for seed in seeds:
            print(f"\nSeed {seed}")
            
            try:
                exp_dir, results = train_experiment(
                    architecture=arch_name,
                    num_steps=num_steps,
                    seed=seed,
                    log_dir=log_dir,
                    use_cbam=use_cbam,
                    **sac_kwargs
                )
                
                all_results[key].append({
                    'seed': seed,
                    'exp_dir': exp_dir,
                    'results': results
                })
                
            except Exception as e:
                print(f"Error in experiment {key} seed {seed}: {e}")
                all_results[key].append({
                    'seed': seed,
                    'exp_dir': None,
                    'results': None,
                    'error': str(e)
                })
    
    # Save aggregated results
    comparison_dir = os.path.join(log_dir, f'comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(comparison_dir, exist_ok=True)
    
    comparison_path = os.path.join(comparison_dir, 'comparison_results.json')
    # Convert numpy types to json serializable
    def convert(o):
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(str(type(o)))
    with open(comparison_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=convert)
    
    print(f"\nComparison results saved to {comparison_path}")
    
    # Generate report
    generate_comparison_report(all_results, comparison_dir)
    
    return all_results, comparison_dir


def _to_float(x):
    """Converts numpy types to native float for json serialization"""
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x

def generate_comparison_report(results, output_dir):
    """
    Generates visual report of the comparison.
    """
    # Compute statistics
    stats = {}
    
    for key, exp_list in results.items():
        rewards = [exp['results']['final_reward'] for exp in exp_list if exp['results'] is not None]
        
        if rewards:
            stats[key] = {
                'mean': _to_float(np.mean(rewards)),
                'std': _to_float(np.std(rewards)),
                'min': _to_float(np.min(rewards)),
                'max': _to_float(np.max(rewards)),
                'n': len(rewards)
            }
        else:
            stats[key] = None
    
    # Save statistics
    stats_path = os.path.join(output_dir, 'statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Create bar plot
    arch_names = []
    mean_rewards = []
    std_rewards = []
    
    for key, stat in stats.items():
        if stat is not None:
            arch_names.append(key.replace('_', ' ').title())
            mean_rewards.append(stat['mean'])
            std_rewards.append(stat['std'])
    
    if arch_names:
        plt.figure(figsize=(10, 6))
        bars = plt.bar(arch_names, mean_rewards, yerr=std_rewards, capsize=5, alpha=0.7)
        plt.ylabel('Final Reward')
        plt.title('Architecture Comparison - CarRacing')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Add labels to bars
        for bar, mean, std in zip(bars, mean_rewards, std_rewards):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{mean:.1f}±{std:.1f}',
                    ha='center', va='bottom')
        
        plot_path = os.path.join(output_dir, 'comparison_plot.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {plot_path}")
        plt.close()
    
    # Create text report
    report_path = os.path.join(output_dir, 'comparison_report.txt')
    with open(report_path, 'w') as f:
        f.write("Architecture Comparison Report - CarRacing\n")
        f.write("="*60 + "\n\n")
        
        for key, stat in stats.items():
            if stat is not None:
                f.write(f"{key.replace('_', ' ').title()}\n")
                f.write(f"  Mean: {stat['mean']:.2f}\n")
                f.write(f"  Std: {stat['std']:.2f}\n")
                f.write(f"  Min: {stat['min']:.2f}\n")
                f.write(f"  Max: {stat['max']:.2f}\n")
                f.write(f"  N: {stat['n']}\n\n")
            else:
                f.write(f"{key.replace('_', ' ').title()}\n")
                f.write("  No valid data\n\n")
    
    print(f"Report saved to {report_path}")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = argparse.ArgumentParser(description='CNN architecture comparison for SAC on CarRacing')
    
    parser.add_argument('--num_steps', type=int, default=50000,
                       help='Number of training steps (default: 50000)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                       help='Random seeds (default: [42, 43, 44])')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='Log directory (default: ./logs)')
    parser.add_argument('--architecture', type=str, choices=['classic', 'attention', 'both'],
                       default='both', help='Architecture to evaluate (default: both)')
    parser.add_argument('--use_cbam', type=str2bool, nargs='?', const=True, default=True,
                       help='Use full CBAM attention (default: True). Pass True/False')
    
    # SAC hyperparameters (aligned with sac_trainer.py and README: 3e-4)
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate (default: 3e-4)')
    parser.add_argument('--alpha', type=float, default=0.2,
                       help='Initial alpha (default: 0.2)')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005,
                       help='Target update rate (default: 0.005)')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size (default: 256)')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                       help='Warmup steps (default: 1000)')
    
    args = parser.parse_args()
    
    sac_kwargs = {
        'lr': args.lr,
        'alpha': args.alpha,
        'gamma': args.gamma,
        'tau': args.tau,
        'batch_size': args.batch_size,
        'warmup_steps': args.warmup_steps
    }
    
    if args.architecture == 'both':
        print("Running full architecture comparison")
        run_comparison(
            num_steps=args.num_steps,
            seeds=args.seeds,
            log_dir=args.log_dir,
            **sac_kwargs
        )
    else:
        print(f"Running single experiment: {args.architecture}")
        train_experiment(
            architecture=args.architecture,
            num_steps=args.num_steps,
            seed=args.seeds[0],
            log_dir=args.log_dir,
            use_cbam=args.use_cbam,
            **sac_kwargs
        )


if __name__ == "__main__":
    main()
