#!/usr/bin/env python3
"""
plot_optimization_results.py - Visualize training speed optimization results

Creates plots showing steps/s vs each parameter for easy visualization
Saves all plots as high-quality JPG files.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_results():
    """Load both optimization result files"""
    results = []
    
    # Load first optimization results
    if Path("speed_optimization_results.json").exists():
        with open("speed_optimization_results.json", "r") as f:
            data1 = json.load(f)
            results.extend(data1["all_results"])
            print(f"Loaded {len(data1['all_results'])} results from speed_optimization_results.json")
    
    # Load extended optimization results
    if Path("extended_speed_optimization_results.json").exists():
        with open("extended_speed_optimization_results.json", "r") as f:
            data2 = json.load(f)
            # Filter out results with 0 steps/s (failed tests)
            valid_results = [r for r in data2["all_results"] if r["steps_per_sec"] > 0]
            results.extend(valid_results)
            print(f"Loaded {len(valid_results)} valid results from extended_speed_optimization_results.json")
    
    if not results:
        print("❌ No optimization results found!")
        return []
    
    print(f"📊 Total results loaded: {len(results)}")
    return results

def extract_parameter_data(results):
    """Extract parameter values and corresponding steps/s"""
    data = []
    
    for result in results:
        config = result["config"]
        steps_per_sec = result["steps_per_sec"]
        
        # Extract parameters
        row = {
            "steps_per_sec": steps_per_sec,
            "n_cores": config.get("n_cores", 6),
            "n_envs": config.get("n_envs", 6),
            "batch_size": config.get("batch_size", 256),
            "n_steps": config.get("n_steps", 1536),
            "n_epochs": config.get("n_epochs", 4),
            "config_name": config.get("name", "Unknown"),
            "net_arch_str": str(config.get("net_arch", [256, 256])),
            "net_arch_size": len(config.get("net_arch", [256, 256])),
            "net_arch_width": config.get("net_arch", [256, 256])[0] if config.get("net_arch") else 256
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    print(f"📊 Extracted data shape: {df.shape}")
    print(f"📊 Steps/s range: {df['steps_per_sec'].min():.2f} - {df['steps_per_sec'].max():.2f}")
    
    return df

def plot_parameter_vs_performance(df, param_name, param_label, filename):
    """Create a scatter plot of parameter vs performance"""
    plt.figure(figsize=(12, 8))
    
    # Get unique values and their performance
    param_performance = df.groupby(param_name)['steps_per_sec'].agg(['mean', 'std', 'count']).reset_index()
    param_performance = param_performance.sort_values(param_name)
    
    # Create scatter plot with error bars
    plt.errorbar(param_performance[param_name], param_performance['mean'], 
                yerr=param_performance['std'], fmt='o', capsize=5, capthick=2,
                markersize=8, linewidth=2, alpha=0.8)
    
    # Add individual points with some jitter for visibility
    for param_val in param_performance[param_name]:
        subset = df[df[param_name] == param_val]
        jitter = np.random.normal(0, 0.02, len(subset))  # Small jitter
        plt.scatter(subset[param_name] + jitter, subset['steps_per_sec'], 
                   alpha=0.5, s=30, color='red')
    
    # Highlight best performer
    best_idx = param_performance['mean'].idxmax()
    best_param = param_performance.loc[best_idx, param_name]
    best_perf = param_performance.loc[best_idx, 'mean']
    
    plt.scatter(best_param, best_perf, color='gold', s=200, marker='*', 
               edgecolor='black', linewidth=2, label=f'Best: {param_label}={best_param}')
    
    plt.xlabel(param_label, fontsize=14, fontweight='bold')
    plt.ylabel('Training Speed (steps/s)', fontsize=14, fontweight='bold')
    plt.title(f'Training Speed vs {param_label}\nOptimization Results', 
             fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Add statistics text box
    stats_text = f'Best: {best_param} → {best_perf:.2f} steps/s\n'
    stats_text += f'Range: {param_performance["mean"].min():.2f} - {param_performance["mean"].max():.2f}\n'
    stats_text += f'Configs tested: {len(param_performance)}'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
            verticalalignment='top', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', format='jpg')
    print(f"✅ Saved plot: {filename}")
    plt.close()

def plot_cores_vs_envs_heatmap(df, filename):
    """Create a heatmap showing cores vs envs performance"""
    plt.figure(figsize=(12, 8))
    
    # Create pivot table
    pivot_data = df.pivot_table(values='steps_per_sec', index='n_cores', 
                               columns='n_envs', aggfunc='mean')
    
    # Create heatmap
    sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdYlGn', 
                cbar_kws={'label': 'Steps/s'}, square=True)
    
    plt.title('Training Speed Heatmap: Cores vs Environments\n(Higher is Better)', 
             fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Number of Environments', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Cores', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', format='jpg')
    print(f"✅ Saved heatmap: {filename}")
    plt.close()

def plot_top_configurations(df, filename):
    """Plot top 10 configurations"""
    plt.figure(figsize=(14, 10))
    
    # Get top 15 configurations
    top_configs = df.nlargest(15, 'steps_per_sec')
    
    # Create horizontal bar plot
    y_pos = np.arange(len(top_configs))
    bars = plt.barh(y_pos, top_configs['steps_per_sec'], alpha=0.8)
    
    # Color code the bars
    colors = plt.cm.RdYlGn(np.linspace(0.3, 1.0, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Add configuration labels
    labels = []
    for _, row in top_configs.iterrows():
        label = f"{row['config_name']}\n({row['n_cores']}C/{row['n_envs']}E, B:{row['batch_size']}, S:{row['n_steps']})"
        labels.append(label)
    
    plt.yticks(y_pos, labels, fontsize=10)
    plt.xlabel('Training Speed (steps/s)', fontsize=14, fontweight='bold')
    plt.title('Top 15 Training Configurations\nRanked by Performance', 
             fontsize=16, fontweight='bold', pad=20)
    
    # Add value labels on bars
    for i, (_, row) in enumerate(top_configs.iterrows()):
        plt.text(row['steps_per_sec'] + 0.1, i, f'{row["steps_per_sec"]:.2f}', 
                va='center', fontweight='bold', fontsize=9)
    
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', format='jpg')
    print(f"✅ Saved top configs plot: {filename}")
    plt.close()

def plot_network_architecture_comparison(df, filename):
    """Compare network architectures"""
    plt.figure(figsize=(12, 8))
    
    # Group by network architecture
    net_performance = df.groupby('net_arch_str')['steps_per_sec'].agg(['mean', 'std', 'count']).reset_index()
    net_performance = net_performance.sort_values('mean', ascending=True)
    
    # Create horizontal bar plot with error bars
    y_pos = np.arange(len(net_performance))
    bars = plt.barh(y_pos, net_performance['mean'], xerr=net_performance['std'],
                   alpha=0.8, capsize=5)
    
    # Color bars based on performance
    colors = plt.cm.RdYlGn(np.linspace(0.3, 1.0, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.yticks(y_pos, net_performance['net_arch_str'], fontsize=11)
    plt.xlabel('Training Speed (steps/s)', fontsize=14, fontweight='bold')
    plt.title('Network Architecture Performance Comparison', 
             fontsize=16, fontweight='bold', pad=20)
    
    # Add value labels
    for i, row in net_performance.iterrows():
        plt.text(row['mean'] + 0.1, i, f'{row["mean"]:.2f}±{row["std"]:.2f}', 
                va='center', fontweight='bold', fontsize=10)
    
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', format='jpg')
    print(f"✅ Saved network comparison plot: {filename}")
    plt.close()

def create_summary_dashboard(df, filename):
    """Create a comprehensive dashboard"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Training Speed Optimization Dashboard\nApple Silicon MPS Performance Analysis', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # 1. Cores vs Performance
    ax1 = axes[0, 0]
    cores_perf = df.groupby('n_cores')['steps_per_sec'].mean().sort_index()
    ax1.plot(cores_perf.index, cores_perf.values, 'o-', linewidth=3, markersize=8)
    ax1.set_xlabel('Number of Cores')
    ax1.set_ylabel('Steps/s')
    ax1.set_title('Cores vs Performance')
    ax1.grid(True, alpha=0.3)
    
    # 2. Environments vs Performance  
    ax2 = axes[0, 1]
    envs_perf = df.groupby('n_envs')['steps_per_sec'].mean().sort_index()
    ax2.plot(envs_perf.index, envs_perf.values, 'o-', linewidth=3, markersize=8, color='orange')
    ax2.set_xlabel('Number of Environments')
    ax2.set_ylabel('Steps/s')
    ax2.set_title('Environments vs Performance')
    ax2.grid(True, alpha=0.3)
    
    # 3. Batch Size vs Performance
    ax3 = axes[0, 2]
    batch_perf = df.groupby('batch_size')['steps_per_sec'].mean().sort_index()
    ax3.plot(batch_perf.index, batch_perf.values, 'o-', linewidth=3, markersize=8, color='green')
    ax3.set_xlabel('Batch Size')
    ax3.set_ylabel('Steps/s')
    ax3.set_title('Batch Size vs Performance')
    ax3.grid(True, alpha=0.3)
    
    # 4. N_Steps vs Performance
    ax4 = axes[1, 0]
    steps_perf = df.groupby('n_steps')['steps_per_sec'].mean().sort_index()
    ax4.plot(steps_perf.index, steps_perf.values, 'o-', linewidth=3, markersize=8, color='red')
    ax4.set_xlabel('N_Steps (Rollout Buffer)')
    ax4.set_ylabel('Steps/s')
    ax4.set_title('N_Steps vs Performance')
    ax4.grid(True, alpha=0.3)
    
    # 5. Epochs vs Performance
    ax5 = axes[1, 1]
    epochs_perf = df.groupby('n_epochs')['steps_per_sec'].mean().sort_index()
    ax5.plot(epochs_perf.index, epochs_perf.values, 'o-', linewidth=3, markersize=8, color='purple')
    ax5.set_xlabel('Number of Epochs')
    ax5.set_ylabel('Steps/s')
    ax5.set_title('Epochs vs Performance')
    ax5.grid(True, alpha=0.3)
    
    # 6. Performance Distribution
    ax6 = axes[1, 2]
    ax6.hist(df['steps_per_sec'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax6.axvline(df['steps_per_sec'].mean(), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {df["steps_per_sec"].mean():.2f}')
    ax6.axvline(df['steps_per_sec'].max(), color='gold', linestyle='--', linewidth=2,
               label=f'Best: {df["steps_per_sec"].max():.2f}')
    ax6.set_xlabel('Steps/s')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Performance Distribution')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', format='jpg')
    print(f"✅ Saved dashboard: {filename}")
    plt.close()

def main():
    """Main function to create all plots"""
    print("📊 Creating Training Speed Optimization Plots")
    print("=" * 50)
    
    # Load data
    results = load_results()
    if not results:
        return
    
    # Extract parameter data
    df = extract_parameter_data(results)
    
    # Create plots directory
    Path("optimization_plots").mkdir(exist_ok=True)
    
    # Create individual parameter plots
    print("\n🎨 Creating parameter vs performance plots...")
    plot_parameter_vs_performance(df, 'n_cores', 'Number of Cores', 
                                 'optimization_plots/cores_vs_performance.jpg')
    
    plot_parameter_vs_performance(df, 'n_envs', 'Number of Environments', 
                                 'optimization_plots/envs_vs_performance.jpg')
    
    plot_parameter_vs_performance(df, 'batch_size', 'Batch Size', 
                                 'optimization_plots/batch_vs_performance.jpg')
    
    plot_parameter_vs_performance(df, 'n_steps', 'N_Steps (Rollout Buffer)', 
                                 'optimization_plots/steps_vs_performance.jpg')
    
    plot_parameter_vs_performance(df, 'n_epochs', 'Number of Epochs', 
                                 'optimization_plots/epochs_vs_performance.jpg')
    
    plot_parameter_vs_performance(df, 'net_arch_width', 'Network Width (First Layer)', 
                                 'optimization_plots/network_width_vs_performance.jpg')
    
    # Create specialized plots
    print("\n🎨 Creating specialized plots...")
    plot_cores_vs_envs_heatmap(df, 'optimization_plots/cores_envs_heatmap.jpg')
    plot_top_configurations(df, 'optimization_plots/top_configurations.jpg')
    plot_network_architecture_comparison(df, 'optimization_plots/network_architectures.jpg')
    
    # Create comprehensive dashboard
    print("\n🎨 Creating comprehensive dashboard...")
    create_summary_dashboard(df, 'optimization_plots/optimization_dashboard.jpg')
    
    print("\n" + "=" * 50)
    print("🎉 All plots created successfully!")
    print("📁 Plots saved in: optimization_plots/")
    print("\nPlot files created:")
    print("  • cores_vs_performance.jpg")
    print("  • envs_vs_performance.jpg") 
    print("  • batch_vs_performance.jpg")
    print("  • steps_vs_performance.jpg")
    print("  • epochs_vs_performance.jpg")
    print("  • network_width_vs_performance.jpg")
    print("  • cores_envs_heatmap.jpg")
    print("  • top_configurations.jpg")
    print("  • network_architectures.jpg")
    print("  • optimization_dashboard.jpg")
    
    # Print key insights
    print("\n📈 KEY INSIGHTS:")
    print(f"  🏆 Best performance: {df['steps_per_sec'].max():.2f} steps/s")
    best_config = df.loc[df['steps_per_sec'].idxmax()]
    print(f"  ⚙️  Best config: {best_config['config_name']}")
    print(f"  🔧 Optimal cores: {df.groupby('n_cores')['steps_per_sec'].mean().idxmax()}")
    print(f"  🔧 Optimal envs: {df.groupby('n_envs')['steps_per_sec'].mean().idxmax()}")
    print(f"  🔧 Optimal batch: {df.groupby('batch_size')['steps_per_sec'].mean().idxmax()}")
    print(f"  🔧 Optimal steps: {df.groupby('n_steps')['steps_per_sec'].mean().idxmax()}")

if __name__ == "__main__":
    main()