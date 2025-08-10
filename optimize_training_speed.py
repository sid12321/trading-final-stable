#!/usr/bin/env python3
"""
optimize_training_speed.py - Automated training speed optimization

Systematically tests different parameter combinations to maximize steps/s
on Apple Silicon MPS systems.

Usage: python optimize_training_speed.py
"""

import os
import sys
import time
import subprocess
import json
import signal
from datetime import datetime
from pathlib import Path

# Setup path
basepath = '/Users/skumar81/Desktop/Personal/trading-final-stable'
os.chdir(basepath)

class TrainingSpeedOptimizer:
    def __init__(self):
        self.results = []
        self.best_config = None
        self.best_steps_per_sec = 0
        self.test_duration = 25  # seconds to run each test
        self.results_file = "speed_optimization_results.json"
        
    def backup_parameters(self):
        """Backup original parameters.py"""
        import shutil
        shutil.copy("parameters.py", "parameters_backup.py")
        print("✅ Backed up original parameters.py")
    
    def restore_parameters(self):
        """Restore original parameters.py"""
        import shutil
        shutil.copy("parameters_backup.py", "parameters.py")
        print("✅ Restored original parameters.py")
    
    def modify_parameters(self, config):
        """Modify parameters.py with test configuration"""
        with open("parameters.py", "r") as f:
            content = f.read()
        
        # Apply modifications
        modifications = {
            'N_CORES': f'    N_CORES = min(mp.cpu_count(), {config["n_cores"]})  # Apple Silicon has performance/efficiency cores',
            'N_ENVS': f'    N_ENVS = min(mp.cpu_count(), {config["n_envs"]})  # Match core count for Metal',
            'BATCH_SIZE': f'BATCH_SIZE = {config["batch_size"]}   # Reasonable batch size for trading data #SID',
            'N_STEPS': f'N_STEPS = {config["n_steps"]}  # Reasonable rollout buffer #SID',
            'N_EPOCHS': f'N_EPOCHS = {config["n_epochs"]}  # Standard epochs for stability #SID'
        }
        
        for param, new_line in modifications.items():
            # Find the line containing the parameter and replace it
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if param in line and not line.strip().startswith('#') and '=' in line:
                    # Keep the indentation
                    indent = len(line) - len(line.lstrip())
                    if param in ['N_CORES', 'N_ENVS']:
                        # These are indented under elif DEVICE == "mps":
                        lines[i] = new_line
                    else:
                        # These are at the global level
                        lines[i] = new_line
                    break
            content = '\n'.join(lines)
        
        # Handle network architecture separately
        if 'net_arch' in config:
            net_arch = config['net_arch']
            new_policy_kwargs = f"""POLICY_KWARGS = {{
    'activation_fn': 'ReLU',  # Will be converted to torch.nn.ReLU in modeltrain
    'net_arch': {{
        'pi': {net_arch},  # Optimized network for trading
        'vf': {net_arch}   # Optimized network for trading
    }},
    'ortho_init': True  # Orthogonal initialization for better training
}}"""
            
            # Find and replace POLICY_KWARGS section
            lines = content.split('\n')
            start_idx = None
            end_idx = None
            
            for i, line in enumerate(lines):
                if 'POLICY_KWARGS = {' in line:
                    start_idx = i
                elif start_idx is not None and line.strip() == '}':
                    end_idx = i
                    break
            
            if start_idx is not None and end_idx is not None:
                lines[start_idx:end_idx+1] = new_policy_kwargs.split('\n')
                content = '\n'.join(lines)
        
        with open("parameters.py", "w") as f:
            f.write(content)
    
    def run_training_test(self, config):
        """Run a training test and measure steps/s"""
        print(f"\n🧪 Testing configuration:")
        for key, value in config.items():
            print(f"   {key}: {value}")
        
        # Modify parameters
        self.modify_parameters(config)
        
        # Start training process
        cmd = ["python", "train_only.py", "--symbols", "BPCL", "--no-posterior"]
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        start_time = time.time()
        steps_per_sec_readings = []
        
        try:
            while time.time() - start_time < self.test_duration:
                output = process.stdout.readline()
                if output:
                    # Look for steps/s in the output
                    if "steps/s" in output:
                        # Extract steps/s value
                        # Format: "Training BPCL:   0%|...| 1380/3000000 [00:13<8:03:28, 103.37steps/s]"
                        try:
                            parts = output.strip().split(',')
                            if len(parts) >= 2:
                                steps_part = parts[-1]  # Last part should contain steps/s
                                steps_value = float(steps_part.split('steps/s')[0].strip())
                                steps_per_sec_readings.append(steps_value)
                                print(f"   📊 Current: {steps_value:.2f} steps/s")
                        except (ValueError, IndexError):
                            continue
                
                # Check if process ended early
                if process.poll() is not None:
                    break
            
            # Terminate the process
            process.terminate()
            process.wait(timeout=5)
            
        except Exception as e:
            print(f"   ❌ Error during test: {e}")
            process.terminate()
            return 0
        
        # Calculate average steps/s (excluding first few readings for warmup)
        if len(steps_per_sec_readings) > 2:
            # Skip first 2 readings for warmup, average the rest
            avg_steps_per_sec = sum(steps_per_sec_readings[2:]) / len(steps_per_sec_readings[2:])
        elif len(steps_per_sec_readings) > 0:
            avg_steps_per_sec = sum(steps_per_sec_readings) / len(steps_per_sec_readings)
        else:
            avg_steps_per_sec = 0
        
        print(f"   📈 Average: {avg_steps_per_sec:.2f} steps/s ({len(steps_per_sec_readings)} readings)")
        return avg_steps_per_sec
    
    def save_results(self):
        """Save optimization results"""
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'best_config': self.best_config,
            'best_steps_per_sec': self.best_steps_per_sec,
            'all_results': self.results
        }
        
        with open(self.results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n💾 Results saved to {self.results_file}")
    
    def optimize(self):
        """Run optimization tests"""
        print("🚀 Starting Training Speed Optimization")
        print("=" * 60)
        print("This will test different parameter combinations to maximize steps/s")
        print(f"Each test runs for {self.test_duration} seconds")
        print("=" * 60)
        
        # Backup original parameters
        self.backup_parameters()
        
        try:
            # Define parameter search space (focusing on Apple Silicon MPS)
            test_configs = [
                # Current best (your finding)
                {"name": "Current Best", "n_cores": 6, "n_envs": 6, "batch_size": 256, "n_steps": 1536, "n_epochs": 4, "net_arch": [256, 256]},
                
                # Test fewer cores/envs
                {"name": "Minimal Parallel", "n_cores": 4, "n_envs": 4, "batch_size": 256, "n_steps": 1536, "n_epochs": 4, "net_arch": [256, 256]},
                {"name": "Ultra Minimal", "n_cores": 2, "n_envs": 2, "batch_size": 256, "n_steps": 1536, "n_epochs": 4, "net_arch": [256, 256]},
                {"name": "Single Thread", "n_cores": 1, "n_envs": 1, "batch_size": 256, "n_steps": 1536, "n_epochs": 4, "net_arch": [256, 256]},
                
                # Test larger batch sizes with fewer envs
                {"name": "Large Batch 4E", "n_cores": 4, "n_envs": 4, "batch_size": 384, "n_steps": 1536, "n_epochs": 4, "net_arch": [256, 256]},
                {"name": "Large Batch 6E", "n_cores": 6, "n_envs": 6, "batch_size": 384, "n_steps": 1536, "n_epochs": 4, "net_arch": [256, 256]},
                {"name": "XL Batch", "n_cores": 4, "n_envs": 4, "batch_size": 512, "n_steps": 1536, "n_epochs": 4, "net_arch": [256, 256]},
                
                # Test different n_steps
                {"name": "Larger Steps", "n_cores": 6, "n_envs": 6, "batch_size": 256, "n_steps": 2048, "n_epochs": 4, "net_arch": [256, 256]},
                {"name": "Smaller Steps", "n_cores": 6, "n_envs": 6, "batch_size": 256, "n_steps": 1024, "n_epochs": 4, "net_arch": [256, 256]},
                
                # Test different n_epochs
                {"name": "More Epochs", "n_cores": 6, "n_envs": 6, "batch_size": 256, "n_steps": 1536, "n_epochs": 8, "net_arch": [256, 256]},
                {"name": "Fewer Epochs", "n_cores": 6, "n_envs": 6, "batch_size": 256, "n_steps": 1536, "n_epochs": 2, "net_arch": [256, 256]},
                
                # Test different network sizes
                {"name": "Smaller Net", "n_cores": 6, "n_envs": 6, "batch_size": 256, "n_steps": 1536, "n_epochs": 4, "net_arch": [128, 128]},
                {"name": "Larger Net", "n_cores": 6, "n_envs": 6, "batch_size": 256, "n_steps": 1536, "n_epochs": 4, "net_arch": [384, 256]},
                {"name": "Deep Net", "n_cores": 6, "n_envs": 6, "batch_size": 256, "n_steps": 1536, "n_epochs": 4, "net_arch": [256, 256, 128]},
                
                # Test combinations
                {"name": "Fast Combo 1", "n_cores": 4, "n_envs": 4, "batch_size": 384, "n_steps": 1024, "n_epochs": 2, "net_arch": [128, 128]},
                {"name": "Fast Combo 2", "n_cores": 2, "n_envs": 2, "batch_size": 512, "n_steps": 2048, "n_epochs": 4, "net_arch": [256, 256]},
            ]
            
            print(f"🔬 Running {len(test_configs)} optimization tests...")
            
            for i, config in enumerate(test_configs):
                print(f"\n[{i+1}/{len(test_configs)}] {config['name']}")
                
                steps_per_sec = self.run_training_test(config)
                
                result = {
                    'config': config,
                    'steps_per_sec': steps_per_sec,
                    'timestamp': datetime.now().isoformat()
                }
                
                self.results.append(result)
                
                if steps_per_sec > self.best_steps_per_sec:
                    self.best_steps_per_sec = steps_per_sec
                    self.best_config = config
                    print(f"   🏆 NEW BEST! {steps_per_sec:.2f} steps/s")
                
                # Save intermediate results
                self.save_results()
                
                # Brief pause between tests
                time.sleep(2)
            
            print("\n" + "=" * 60)
            print("🏁 OPTIMIZATION COMPLETE!")
            print("=" * 60)
            
            if self.best_config:
                print(f"🏆 Best Configuration: {self.best_config['name']}")
                print(f"📈 Best Speed: {self.best_steps_per_sec:.2f} steps/s")
                print("\nOptimal parameters:")
                for key, value in self.best_config.items():
                    if key != 'name':
                        print(f"  {key}: {value}")
                
                # Apply best configuration
                print(f"\n🔄 Applying best configuration to parameters.py...")
                self.modify_parameters(self.best_config)
                print("✅ Best configuration applied!")
                
            print("\n📊 All Results (sorted by speed):")
            sorted_results = sorted(self.results, key=lambda x: x['steps_per_sec'], reverse=True)
            for result in sorted_results:
                config = result['config']
                print(f"  {result['steps_per_sec']:6.2f} steps/s - {config['name']} "
                      f"(cores:{config['n_cores']}, envs:{config['n_envs']}, "
                      f"batch:{config['batch_size']}, steps:{config['n_steps']})")
                
        except KeyboardInterrupt:
            print("\n\n⏸️  Optimization interrupted by user")
        except Exception as e:
            print(f"\n❌ Optimization failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Always save results and optionally restore backup
            if self.results:
                self.save_results()
            
            response = input("\n🔄 Keep optimized parameters? [Y/n]: ").strip().lower()
            if response in ['n', 'no']:
                self.restore_parameters()
                print("✅ Original parameters restored")

if __name__ == "__main__":
    optimizer = TrainingSpeedOptimizer()
    optimizer.optimize()