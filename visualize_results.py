import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import seaborn as sns
from pathlib import Path


def load_tensorboard_logs(log_dir):
    """
    Loads TensorBoard logs from a directory
    """
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    
    logs = {}
    
    # Load available scalars
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        logs[tag] = {
            'steps': [e.step for e in events],
            'values': [e.value for e in events]
        }
    
    return logs


def plot_training_curves(log_dirs, labels, output_dir):
    """
    Plots comparative training curves
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load logs
    all_logs = []
    for log_dir in log_dirs:
        if os.path.exists(log_dir):
            logs = load_tensorboard_logs(log_dir)
            all_logs.append(logs)
        else:
            print(f"Directory {log_dir} not found")
            all_logs.append(None)
    
    # Plot episode rewards
    plt.figure(figsize=(12, 6))
    
    for i, (logs, label) in enumerate(zip(all_logs, labels)):
        if logs and 'Reward/episode' in logs:
            steps = logs['Reward/episode']['steps']
            values = logs['Reward/episode']['values']
            
            # Smooth with moving average
            window = 10
            if len(values) >= window:
                smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
                smoothed_steps = steps[window-1:]
            else:
                smoothed = values
                smoothed_steps = steps
            
            plt.plot(smoothed_steps, smoothed, label=label, alpha=0.7)
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Episode Reward During Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'episode_rewards.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Reward plot saved in {output_path}")
    plt.close()
    
    # Plot losses
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    loss_types = ['Loss/critic', 'Loss/actor', 'Loss/alpha']
    titles = ['Critic Loss', 'Actor Loss', 'Alpha Loss']
    
    for ax, loss_type, title in zip(axes, loss_types, titles):
        for i, (logs, label) in enumerate(zip(all_logs, labels)):
            if logs and loss_type in logs:
                steps = logs[loss_type]['steps']
                values = logs[loss_type]['values']
                
                # Smooth
                window = 50
                if len(values) >= window:
                    smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
                    smoothed_steps = steps[window-1:]
                else:
                    smoothed = values
                    smoothed_steps = steps
                
                ax.plot(smoothed_steps, smoothed, label=label, alpha=0.7)
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'training_losses.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Loss plot saved in {output_path}")
    plt.close()
    
    # Plot evaluation rewards
    plt.figure(figsize=(12, 6))
    
    for i, (logs, label) in enumerate(zip(all_logs, labels)):
        if logs and 'Reward/eval' in logs:
            steps = logs['Reward/eval']['steps']
            values = logs['Reward/eval']['values']
            
            plt.plot(steps, values, 'o-', label=label, alpha=0.7)
    
    plt.xlabel('Training Step')
    plt.ylabel('Eval Reward')
    plt.title('Evaluation Reward During Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'eval_rewards.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Evaluation plot saved in {output_path}")
    plt.close()


def plot_attention_maps(model, state, output_dir, epoch=0):
    """
    Visualizes attention maps of a model with attention
    """
    if not hasattr(model, 'get_attention_maps'):
        print("Model does not support attention visualization")
        return
    
    import torch
    
    model.eval()
    with torch.no_grad():
        # Convert state to tensor
        if isinstance(state, np.ndarray):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
        else:
            state_tensor = state
        
        # Obtain attention maps
        att_maps = model.get_attention_maps(state_tensor)
    
    # Plot - handles case with only 1 map (axes is not array)
    n_maps = len(att_maps)
    fig, axes = plt.subplots(1, n_maps, figsize=(5*n_maps, 5))
    if n_maps == 1:
        axes = [axes]
    
    for i, att_map in enumerate(att_maps):
        # First channel of first batch
        att_np = att_map[0, 0].cpu().numpy() if hasattr(att_map, 'cpu') else np.asarray(att_map)[0, 0]
        
        axes[i].imshow(att_np, cmap='hot', interpolation='nearest')
        axes[i].set_title(f'Attention Layer {i+1}')
        axes[i].axis('off')
    
    plt.suptitle(f'Attention Maps - Epoch {epoch}')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'attention_maps_epoch_{epoch}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Attention maps saved in {output_path}")
    plt.close()


def plot_comparison_summary(comparison_results_path, output_dir):
    """
    Creates a visual summary of comparison results
    """
    with open(comparison_results_path, 'r') as f:
        results = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    arch_names = []
    final_rewards = []
    seeds_per_arch = []
    
    for arch_key, exp_list in results.items():
        rewards = [exp['results']['final_reward'] for exp in exp_list if exp['results'] is not None]
        seeds = [exp['seed'] for exp in exp_list if exp['results'] is not None]
        
        if rewards:
            arch_names.append(arch_key.replace('_', ' ').title())
            final_rewards.append(rewards)
            seeds_per_arch.append(seeds)
    
    # Box plot
    plt.figure(figsize=(10, 6))
    # labels deprecated -> tick_labels in matplotlib 3.9+, maintain compat
    try:
        bp = plt.boxplot(final_rewards, tick_labels=arch_names, patch_artist=True)
    except TypeError:
        bp = plt.boxplot(final_rewards, labels=arch_names, patch_artist=True)
    
    # Color box plots
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightsalmon', 'lightyellow']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.ylabel('Final Reward')
    plt.title('Comparison of CNN Architectures - Final Rewards')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'boxplot_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Box plot saved in {output_path}")
    plt.close()
    
    # Scatter plot with seeds
    plt.figure(figsize=(10, 6))
    
    for i, (rewards, seeds) in enumerate(zip(final_rewards, seeds_per_arch)):
        plt.scatter(seeds, rewards, label=arch_names[i], s=100, alpha=0.7)
    
    plt.xlabel('Seed')
    plt.ylabel('Final Reward')
    plt.title('Final Rewards by Seed')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'scatter_seeds.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Scatter plot saved in {output_path}")
    plt.close()
    
    # Statistics table
    stats = []
    for arch_key, exp_list in results.items():
        rewards = [exp['results']['final_reward'] for exp in exp_list if exp['results'] is not None]
        
        if rewards:
            stats.append({
                'Architecture': arch_key.replace('_', ' ').title(),
                'Mean': np.mean(rewards),
                'Std': np.std(rewards),
                'Min': np.min(rewards),
                'Max': np.max(rewards),
                'N': len(rewards)
            })
    
    # Create figure with table
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    for stat in stats:
        table_data.append([
            stat['Architecture'],
            f"{stat['Mean']:.2f}",
            f"{stat['Std']:.2f}",
            f"{stat['Min']:.2f}",
            f"{stat['Max']:.2f}",
            stat['N']
        ])
    
    table = ax.table(
        cellText=table_data,
        colLabels=['Architecture', 'Mean', 'Std', 'Min', 'Max', 'N'],
        cellLoc='center',
        loc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    plt.title('Statistics Summary', pad=20)
    
    output_path = os.path.join(output_dir, 'statistics_table.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Statistics table saved in {output_path}")
    plt.close()


def create_training_gif(log_dir, output_path, fps=10):
    """
    Creates GIF with training frames (if available)
    """
    try:
        from PIL import Image
        import glob
        
        # Look for saved frames
        frame_pattern = os.path.join(log_dir, 'frames', '*.png')
        frame_files = sorted(glob.glob(frame_pattern))
        
        if not frame_files:
            print("No frames found to create GIF")
            return
        
        # Load frames
        frames = []
        for frame_file in frame_files:
            img = Image.open(frame_file)
            frames.append(img)
        
        # Create GIF
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=1000//fps,
            loop=0
        )
        
        print(f"GIF saved in {output_path}")
        
    except ImportError:
        print("PIL not available to create GIF")
    except Exception as e:
        print(f"Error creating GIF: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize training results')
    # Supports both --log_dirs (new) and --log_dir (legacy, repeated) for README compatibility
    parser.add_argument('--log_dirs', type=str, nargs='+', default=None,
                       help='TensorBoard log directories')
    parser.add_argument('--log_dir', type=str, action='append', default=None,
                       help='Log directory (README compat, can repeat: --log_dir a --log_dir b)')
    parser.add_argument('--labels', type=str, nargs='+', required=True,
                       help='Labels for each directory')
    parser.add_argument('--output_dir', type=str, default='./visualizations',
                       help='Output directory (default: ./visualizations)')
    parser.add_argument('--comparison_results', type=str, default=None,
                       help='Path to comparison results JSON')
    parser.add_argument('--comparison_results_json', type=str, default=None,
                       help='Alias for --comparison_results')
    
    args = parser.parse_args()

    # Resolve log_dirs: prioritizes --log_dirs, otherwise uses --log_dir
    log_dirs = args.log_dirs if args.log_dirs is not None else args.log_dir
    if log_dirs is None:
        parser.error("one of the arguments --log_dirs/--log_dir is required")
    
    # Resolve comparison_results alias
    comp_path = args.comparison_results or args.comparison_results_json

    if len(log_dirs) != len(args.labels):
        print(f"Number of log_dirs ({len(log_dirs)}) must equal the number of labels ({len(args.labels)})")
        import sys
        sys.exit(1)
    
    plot_training_curves(log_dirs, args.labels, args.output_dir)
    
    if comp_path:
        plot_comparison_summary(comp_path, args.output_dir)
