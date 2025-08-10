#!/usr/bin/env python3
"""
plot_focused_optimization.py - Visualize focused optimization results

Creates specialized plots for the 3-phase focused optimization:
1. Epochs optimization
2. Batch size optimization  
3. Network architecture optimization
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_focused_results():
    """Load focused optimization results"""
    if not Path("focused_optimization_results.json").exists():
        print("❌ focused_optimization_results.json not found!")
        return None
    
    with open("focused_optimization_results.json", "r") as f:
        data = json.load(f)
    
    print(f"📊 Loaded focused optimization results")
    print(f"   Fixed parameters: {data['fixed_parameters']}")
    print(f"   Best performance: {data['best_steps_per_sec']:.2f} steps/s")
    print(f"   Total tests: {len(data['all_results'])}")
    
    return data

def extract_phase_data(results, phase_name):
    """Extract data for a specific optimization phase"""
    phase_results = [r for r in results if r['phase'] == phase_name]
    if not phase_results:
        return pd.DataFrame()
    
    data = []
    for result in phase_results:
        row = {
            'value': result['value'],
            'steps_per_sec': result['steps_per_sec'],
            'timestamp': result['timestamp'],
            'value_name': result.get('value_name', str(result['value']))
        }
        data.append(row)
    
    return pd.DataFrame(data)

def plot_phase_optimization(df, phase_name, parameter_label, filename, is_network=False):
    """Create optimization plot for a specific phase"""
    plt.figure(figsize=(14, 8))
    
    if is_network:
        # For network architecture, use names instead of values
        x_values = df['value_name'].tolist()
        y_values = df['steps_per_sec'].tolist()
        
        # Sort by performance for better visualization
        sorted_data = sorted(zip(x_values, y_values), key=lambda x: x[1], reverse=True)
        x_sorted, y_sorted = zip(*sorted_data)
        
        # Create bar plot
        bars = plt.bar(range(len(x_sorted)), y_sorted, alpha=0.8)
        
        # Color bars based on performance
        colors = plt.cm.RdYlGn(np.linspace(0.3, 1.0, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.xticks(range(len(x_sorted)), x_sorted, rotation=45, ha='right')
        
        # Add value labels on bars
        for i, (name, value) in enumerate(zip(x_sorted, y_sorted)):
            plt.text(i, value + 0.1, f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.xlabel('Network Architecture', fontsize=14, fontweight='bold')
    else:
        # For numerical parameters
        x_values = df['value'].tolist()
        y_values = df['steps_per_sec'].tolist()
        
        # Create line plot with markers
        plt.plot(x_values, y_values, 'o-', linewidth=3, markersize=8, alpha=0.8)
        
        # Highlight best point
        best_idx = np.argmax(y_values)
        plt.scatter(x_values[best_idx], y_values[best_idx], color='gold', s=200, 
                   marker='*', edgecolor='black', linewidth=2, zorder=5,
                   label=f'Best: {parameter_label}={x_values[best_idx]}')
        
        # Add value labels on points
        for i, (x, y) in enumerate(zip(x_values, y_values)):
            plt.annotate(f'{y:.2f}', (x, y), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=10, fontweight='bold')
        
        plt.xlabel(parameter_label, fontsize=14, fontweight='bold')
        plt.legend(fontsize=12)
    
    plt.ylabel('Training Speed (steps/s)', fontsize=14, fontweight='bold')
    plt.title(f'{phase_name.title()} Optimization\nFixed Parameters: 1 Core, 1 Env, 3072 Steps', 
             fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    
    # Add performance statistics
    best_perf = max(y_values)
    worst_perf = min(y_values)
    avg_perf = np.mean(y_values)
    
    stats_text = f'Best: {best_perf:.2f} steps/s\n'
    stats_text += f'Worst: {worst_perf:.2f} steps/s\n'
    stats_text += f'Average: {avg_perf:.2f} steps/s\n'
    stats_text += f'Improvement: {((best_perf - worst_perf) / worst_perf * 100):.1f}%'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
            verticalalignment='top', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', format='jpg')
    print(f"✅ Saved plot: {filename}")
    plt.close()

def create_optimization_progression_plot(data, filename):
    """Create a plot showing the optimization progression through phases"""
    plt.figure(figsize=(16, 10))
    
    results = data['all_results']
    
    # Extract best performance from each phase
    phases = ['epochs', 'batch_size', 'network_arch']
    phase_labels = ['Phase 1: Epochs', 'Phase 2: Batch Size', 'Phase 3: Network Architecture']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Focused Optimization: 3-Phase Parameter Tuning\n'
                'Fixed: 1 Core, 1 Environment, 3072 Steps', 
                fontsize=18, fontweight='bold', y=0.98)
    
    for i, (phase, phase_label, color) in enumerate(zip(phases, phase_labels, colors)):
        ax = axes[i]
        phase_results = [r for r in results if r['phase'] == phase]
        
        if phase == 'network_arch':
            # Network architecture plot
            names = [r['value_name'] for r in phase_results]
            values = [r['steps_per_sec'] for r in phase_results]
            
            # Sort by performance
            sorted_data = sorted(zip(names, values), key=lambda x: x[1], reverse=True)
            names_sorted, values_sorted = zip(*sorted_data)
            
            bars = ax.bar(range(len(names_sorted)), values_sorted, color=color, alpha=0.8)
            ax.set_xticks(range(len(names_sorted)))
            ax.set_xticklabels(names_sorted, rotation=45, ha='right', fontsize=9)
            
            # Highlight best
            best_idx = 0  # Already sorted
            bars[best_idx].set_color('gold')
            bars[best_idx].set_edgecolor('black')
            bars[best_idx].set_linewidth(2)
            
        else:
            # Numerical parameter plot
            x_vals = [r['value'] for r in phase_results]
            y_vals = [r['steps_per_sec'] for r in phase_results]
            
            ax.plot(x_vals, y_vals, 'o-', color=color, linewidth=3, markersize=8, alpha=0.8)
            
            # Highlight best
            best_idx = np.argmax(y_vals)
            ax.scatter(x_vals[best_idx], y_vals[best_idx], color='gold', s=150, 
                      marker='*', edgecolor='black', linewidth=2, zorder=5)
        
        ax.set_ylabel('Steps/s', fontsize=12, fontweight='bold')
        ax.set_title(phase_label, fontsize=14, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3)
        
        # Add best value annotation
        best_val = max([r['steps_per_sec'] for r in phase_results])
        ax.text(0.02, 0.98, f'Best: {best_val:.2f}', transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
               verticalalignment='top', fontsize=11, fontweight='bold', color='white')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', format='jpg')
    print(f"✅ Saved progression plot: {filename}")
    plt.close()

def create_performance_summary_plot(data, filename):
    """Create a summary plot showing overall optimization results"""
    plt.figure(figsize=(14, 10))
    
    results = data['all_results']
    best_config = data['best_config']
    fixed_params = data['fixed_parameters']
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Focused Optimization Summary\nApple Silicon MPS Performance Tuning Results', 
                 fontsize=18, fontweight='bold', y=0.95)
    
    # 1. Performance progression over time
    timestamps = [r['timestamp'] for r in results]
    performances = [r['steps_per_sec'] for r in results]
    
    ax1.plot(range(len(performances)), performances, 'o-', alpha=0.7, linewidth=2)
    best_idx = np.argmax(performances)
    ax1.scatter(best_idx, performances[best_idx], color='gold', s=150, marker='*', 
               edgecolor='black', linewidth=2, zorder=5)
    ax1.set_xlabel('Test Number')
    ax1.set_ylabel('Steps/s')
    ax1.set_title('Performance Progression During Optimization')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=np.mean(performances), color='red', linestyle='--', alpha=0.7, 
               label=f'Average: {np.mean(performances):.2f}')
    ax1.legend()
    
    # 2. Performance distribution
    ax2.hist(performances, bins=15, alpha=0.7, edgecolor='black')
    ax2.axvline(x=np.mean(performances), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(performances):.2f}')
    ax2.axvline(x=max(performances), color='gold', linestyle='--', linewidth=2,
               label=f'Best: {max(performances):.2f}')
    ax2.set_xlabel('Steps/s')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Performance Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Phase comparison
    phases = ['epochs', 'batch_size', 'network_arch']
    phase_best = []
    phase_labels = ['Epochs', 'Batch Size', 'Network Arch']
    
    for phase in phases:
        phase_results = [r for r in results if r['phase'] == phase]
        if phase_results:
            phase_best.append(max([r['steps_per_sec'] for r in phase_results]))
        else:
            phase_best.append(0)
    
    bars = ax3.bar(phase_labels, phase_best, alpha=0.8)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax3.set_ylabel('Best Steps/s')
    ax3.set_title('Best Performance by Phase')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, phase_best)):
        ax3.text(i, value + 0.2, f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Configuration summary
    ax4.axis('off')
    
    config_text = "🏆 OPTIMAL CONFIGURATION\n\n"
    config_text += "Fixed Parameters (from previous optimization):\n"
    for key, value in fixed_params.items():
        config_text += f"  • {key}: {value}\n"
    
    config_text += "\nOptimized Parameters (this run):\n"
    config_text += f"  • n_epochs: {best_config['n_epochs']}\n"
    config_text += f"  • batch_size: {best_config['batch_size']}\n"
    config_text += f"  • net_arch: {best_config['net_arch']}\n"
    
    config_text += f"\n📈 PERFORMANCE RESULTS:\n"
    config_text += f"  • Best Speed: {data['best_steps_per_sec']:.2f} steps/s\n"
    config_text += f"  • Total Tests: {len(results)}\n"
    config_text += f"  • Performance Range: {min(performances):.2f} - {max(performances):.2f}\n"
    config_text += f"  • Optimization Gain: {((max(performances) - min(performances))/min(performances)*100):.1f}%"
    
    ax4.text(0.05, 0.95, config_text, transform=ax4.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8),
            verticalalignment='top', fontsize=11, fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', format='jpg')
    print(f"✅ Saved summary plot: {filename}")
    plt.close()

def main():
    """Main function to create focused optimization plots"""
    print("📊 Creating Focused Optimization Plots")
    print("=" * 50)
    
    # Load data
    data = load_focused_results()
    if not data:
        return
    
    # Create plots directory
    Path("focused_plots").mkdir(exist_ok=True)
    
    # Extract phase data
    epochs_df = extract_phase_data(data['all_results'], 'epochs')
    batch_df = extract_phase_data(data['all_results'], 'batch_size')
    network_df = extract_phase_data(data['all_results'], 'network_arch')
    
    print(f"\n🎨 Creating individual phase plots...")
    
    # Create individual phase plots
    if not epochs_df.empty:
        plot_phase_optimization(epochs_df, 'Epochs', 'Number of Epochs', 
                               'focused_plots/epochs_optimization.jpg')
    
    if not batch_df.empty:
        plot_phase_optimization(batch_df, 'Batch Size', 'Batch Size', 
                               'focused_plots/batch_size_optimization.jpg')
    
    if not network_df.empty:
        plot_phase_optimization(network_df, 'Network Architecture', 'Architecture', 
                               'focused_plots/network_architecture_optimization.jpg', is_network=True)
    
    print(f"\n🎨 Creating comprehensive plots...")
    
    # Create comprehensive plots
    create_optimization_progression_plot(data, 'focused_plots/optimization_progression.jpg')
    create_performance_summary_plot(data, 'focused_plots/performance_summary.jpg')
    
    print("\n" + "=" * 50)
    print("🎉 Focused optimization plots created successfully!")
    print("📁 Plots saved in: focused_plots/")
    print("\nPlot files created:")
    print("  • epochs_optimization.jpg - Phase 1: Epochs tuning")
    print("  • batch_size_optimization.jpg - Phase 2: Batch size tuning") 
    print("  • network_architecture_optimization.jpg - Phase 3: Network tuning")
    print("  • optimization_progression.jpg - All 3 phases combined")
    print("  • performance_summary.jpg - Complete optimization summary")
    
    # Print key insights
    results = data['all_results']
    performances = [r['steps_per_sec'] for r in results]
    
    print("\n📈 KEY INSIGHTS FROM FOCUSED OPTIMIZATION:")
    print(f"  🏆 Best performance: {max(performances):.2f} steps/s")
    print(f"  📊 Performance range: {min(performances):.2f} - {max(performances):.2f} steps/s")
    print(f"  🎯 Improvement: {((max(performances) - min(performances))/min(performances)*100):.1f}%")
    
    # Phase-specific insights
    epoch_results = [r for r in results if r['phase'] == 'epochs']
    batch_results = [r for r in results if r['phase'] == 'batch_size']
    network_results = [r for r in results if r['phase'] == 'network_arch']
    
    if epoch_results:
        best_epochs = max(epoch_results, key=lambda x: x['steps_per_sec'])
        print(f"  🔧 Best epochs: {best_epochs['value']} → {best_epochs['steps_per_sec']:.2f} steps/s")
    
    if batch_results:
        best_batch = max(batch_results, key=lambda x: x['steps_per_sec'])
        print(f"  🔧 Best batch size: {best_batch['value']} → {best_batch['steps_per_sec']:.2f} steps/s")
    
    if network_results:
        best_network = max(network_results, key=lambda x: x['steps_per_sec'])
        print(f"  🔧 Best network: {best_network['value_name']} → {best_network['steps_per_sec']:.2f} steps/s")
    
    print(f"\n🎊 FINAL OPTIMIZED CONFIGURATION APPLIED:")
    best_config = data['best_config']
    print(f"  • Cores: {best_config['n_cores']}")
    print(f"  • Environments: {best_config['n_envs']}")
    print(f"  • Steps: {best_config['n_steps']}")
    print(f"  • Epochs: {best_config['n_epochs']}")
    print(f"  • Batch Size: {best_config['batch_size']}")
    print(f"  • Network: {best_config['net_arch']}")

if __name__ == "__main__":
    main()