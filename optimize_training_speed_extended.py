#!/usr/bin/env python3
"""
optimize_training_speed_extended.py - Extended training speed optimization

Tests both lower AND higher core/environment counts to find the absolute optimum.
Includes comprehensive testing across the full parameter space.

Usage: python optimize_training_speed_extended.py
"""

import os
import sys
import time
import subprocess
import json
import multiprocessing as mp
from datetime import datetime
from pathlib import Path

# Setup path
basepath = '/Users/skumar81/Desktop/Personal/trading-final-stable'
os.chdir(basepath)

class ExtendedTrainingSpeedOptimizer:
    def __init__(self):
        self.results = []
        self.best_config = None
        self.best_steps_per_sec = 0
        self.test_duration = 30  # seconds to run each test (increased for stability)
        self.results_file = "extended_speed_optimization_results.json"
        self.max_cores = mp.cpu_count()  # Get actual core count
        
    def backup_parameters(self):
        """Backup original parameters.py"""
        import shutil
        shutil.copy("parameters.py", "parameters_backup_extended.py")
        print("✅ Backed up original parameters.py")
    
    def restore_parameters(self):
        """Restore original parameters.py"""
        import shutil
        if os.path.exists("parameters_backup_extended.py"):
            shutil.copy("parameters_backup_extended.py", "parameters.py")
            print("✅ Restored original parameters.py")
    
    def modify_parameters(self, config):
        """Modify parameters.py with test configuration"""
        with open("parameters.py", "r") as f:
            content = f.read()
        
        # Apply modifications with more robust pattern matching
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            # Handle MPS core settings
            if 'N_CORES = min(mp.cpu_count(),' in line and 'elif DEVICE == "mps":' in content[content.find('\n'.join(lines[:i])):content.find('\n'.join(lines[:i+10]))]:
                lines[i] = f'    N_CORES = min(mp.cpu_count(), {config["n_cores"]})  # Apple Silicon has performance/efficiency cores'
            elif 'N_ENVS = min(mp.cpu_count(),' in line and 'elif DEVICE == "mps":' in content[content.find('\n'.join(lines[:i])):content.find('\n'.join(lines[:i+10]))]:
                lines[i] = f'    N_ENVS = min(mp.cpu_count(), {config["n_envs"]})  # Match core count for Metal'
            # Handle global parameters
            elif line.strip().startswith('BATCH_SIZE ='):
                lines[i] = f'BATCH_SIZE = {config["batch_size"]}   # Optimized batch size for trading data'
            elif line.strip().startswith('N_STEPS ='):
                lines[i] = f'N_STEPS = {config["n_steps"]}  # Optimized rollout buffer'
            elif line.strip().startswith('N_EPOCHS ='):
                lines[i] = f'N_EPOCHS = {config["n_epochs"]}  # Optimized epochs for stability'
        
        content = '\n'.join(lines)
        
        # Handle network architecture
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
                elif start_idx is not None and line.strip() == '}' and 'ortho_init' in lines[i-1]:
                    end_idx = i
                    break
            
            if start_idx is not None and end_idx is not None:
                lines[start_idx:end_idx+1] = new_policy_kwargs.split('\n')
                content = '\n'.join(lines)
        
        with open("parameters.py", "w") as f:
            f.write(content)
    
    def run_training_test(self, config):
        """Run a training test and measure steps/s"""
        print(f"\n🧪 Testing configuration: {config['name']}")
        print(f"   cores: {config['n_cores']}, envs: {config['n_envs']}, "
              f"batch: {config['batch_size']}, steps: {config['n_steps']}, "
              f"epochs: {config['n_epochs']}, net: {config.get('net_arch', 'default')}")
        
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
        training_started = False
        
        try:
            while time.time() - start_time < self.test_duration:
                output = process.stdout.readline()
                if output:
                    print(f"   {output.strip()}")
                    
                    # Wait for actual training to start (skip data loading)
                    if "Training BPCL:" in output:
                        training_started = True
                    
                    # Look for steps/s in the output
                    if training_started and "steps/s" in output:
                        # Extract steps/s value
                        # Format: "Training BPCL:   0%|...| 1380/3000000 [00:13<8:03:28, 103.37steps/s]"
                        try:
                            parts = output.strip().split(',')
                            if len(parts) >= 2:
                                steps_part = parts[-1]  # Last part should contain steps/s
                                if 'steps/s]' in steps_part:
                                    steps_value = float(steps_part.split('steps/s')[0].strip())
                                    steps_per_sec_readings.append(steps_value)
                                    print(f"   📊 Measured: {steps_value:.2f} steps/s")
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
        if len(steps_per_sec_readings) >= 3:
            # Skip first 2 readings for warmup, average the rest
            stable_readings = steps_per_sec_readings[2:]
            avg_steps_per_sec = sum(stable_readings) / len(stable_readings)
            print(f"   📈 Average (stable): {avg_steps_per_sec:.2f} steps/s (from {len(stable_readings)} readings)")
        elif len(steps_per_sec_readings) > 0:
            avg_steps_per_sec = sum(steps_per_sec_readings) / len(steps_per_sec_readings)
            print(f"   📈 Average (all): {avg_steps_per_sec:.2f} steps/s (from {len(steps_per_sec_readings)} readings)")
        else:
            avg_steps_per_sec = 0
            print(f"   ❌ No valid readings captured")
        
        return avg_steps_per_sec
    
    def save_results(self):
        """Save optimization results"""
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'system_cores': self.max_cores,
            'best_config': self.best_config,
            'best_steps_per_sec': self.best_steps_per_sec,
            'all_results': self.results
        }
        
        with open(self.results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n💾 Results saved to {self.results_file}")
    
    def generate_test_configs(self):
        """Generate comprehensive test configurations"""
        configs = []
        
        print(f"System has {self.max_cores} CPU cores")
        
        # Test a wide range of core/env combinations
        core_env_tests = [
            # Very low (your current finding that 6 > 16)
            (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6),
            # Medium range 
            (8, 8), (10, 10), (12, 12), (14, 14), (16, 16),
            # Higher range (test if more is actually better)
            (20, 20), (24, 24), (28, 28), (32, 32),
            # Asymmetric tests
            (4, 8), (6, 12), (8, 16), (12, 24)
        ]
        
        # Filter to valid ranges for the system
        valid_core_env_tests = [(c, e) for c, e in core_env_tests if c <= self.max_cores and e <= self.max_cores * 2]
        
        # Base configuration
        base_config = {
            "batch_size": 256, 
            "n_steps": 1536, 
            "n_epochs": 4, 
            "net_arch": [256, 256]
        }
        
        # 1. Test core/env scaling with base config
        for n_cores, n_envs in valid_core_env_tests:
            configs.append({
                "name": f"Cores{n_cores}_Envs{n_envs}",
                "n_cores": n_cores,
                "n_envs": n_envs,
                **base_config
            })
        
        # 2. Test your current best config variations
        current_best = {"n_cores": 6, "n_envs": 6, "batch_size": 256, "n_steps": 1536, "n_epochs": 4}
        
        # Batch size variations
        for batch_size in [128, 192, 256, 320, 384, 512, 768]:
            configs.append({
                "name": f"Batch{batch_size}_6C6E",
                "n_cores": 6,
                "n_envs": 6,
                "batch_size": batch_size,
                "n_steps": 1536,
                "n_epochs": 4,
                "net_arch": [256, 256]
            })
        
        # N_steps variations
        for n_steps in [512, 1024, 1536, 2048, 3072]:
            configs.append({
                "name": f"Steps{n_steps}_6C6E",
                "n_cores": 6,
                "n_envs": 6,
                "batch_size": 256,
                "n_steps": n_steps,
                "n_epochs": 4,
                "net_arch": [256, 256]
            })
        
        # N_epochs variations  
        for n_epochs in [1, 2, 4, 6, 8]:
            configs.append({
                "name": f"Epochs{n_epochs}_6C6E",
                "n_cores": 6,
                "n_envs": 6,
                "batch_size": 256,
                "n_steps": 1536,
                "n_epochs": n_epochs,
                "net_arch": [256, 256]
            })
        
        # Network architecture variations
        net_configs = [
            ([128, 128], "128x128"),
            ([256, 256], "256x256"),
            ([384, 256], "384x256"),
            ([512, 256], "512x256"),
            ([256, 256, 128], "256x256x128"),
            ([128, 64], "128x64")
        ]
        
        for net_arch, name in net_configs:
            configs.append({
                "name": f"Net{name}_6C6E",
                "n_cores": 6,
                "n_envs": 6,
                "batch_size": 256,
                "n_steps": 1536,
                "n_epochs": 4,
                "net_arch": net_arch
            })
        
        # 3. Test some high-performance combinations for higher core counts
        if self.max_cores >= 16:
            high_perf_configs = [
                {"name": "HighPerf_16C_LargeBatch", "n_cores": 16, "n_envs": 16, "batch_size": 512, "n_steps": 2048, "n_epochs": 2, "net_arch": [256, 256]},
                {"name": "HighPerf_24C_XLBatch", "n_cores": 24, "n_envs": 24, "batch_size": 768, "n_steps": 1024, "n_epochs": 2, "net_arch": [384, 256]},
                {"name": "HighPerf_32C_Massive", "n_cores": 32, "n_envs": 32, "batch_size": 1024, "n_steps": 512, "n_epochs": 1, "net_arch": [512, 256]},
            ]
            
            for config in high_perf_configs:
                if config["n_cores"] <= self.max_cores:
                    configs.append(config)
        
        print(f"Generated {len(configs)} test configurations")
        return configs
    
    def optimize(self):
        """Run extended optimization tests"""
        print("🚀 Starting EXTENDED Training Speed Optimization")
        print("=" * 70)
        print(f"System: {self.max_cores} CPU cores")
        print("Testing BOTH low AND high core/environment counts")
        print(f"Each test runs for {self.test_duration} seconds")
        print("=" * 70)
        
        # Backup original parameters
        self.backup_parameters()
        
        try:
            # Generate all test configurations
            test_configs = self.generate_test_configs()
            
            print(f"🔬 Running {len(test_configs)} optimization tests...")
            
            for i, config in enumerate(test_configs):
                print(f"\n[{i+1}/{len(test_configs)}] Testing: {config['name']}")
                
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
                    print(f"       Config: {config['name']}")
                
                # Save intermediate results
                self.save_results()
                
                # Brief pause between tests
                time.sleep(3)
            
            print("\n" + "=" * 70)
            print("🏁 EXTENDED OPTIMIZATION COMPLETE!")
            print("=" * 70)
            
            if self.best_config:
                print(f"🏆 Best Configuration: {self.best_config['name']}")
                print(f"📈 Best Speed: {self.best_steps_per_sec:.2f} steps/s")
                print("\nOptimal parameters:")
                for key, value in self.best_config.items():
                    if key != 'name':
                        print(f"  {key}: {value}")
            
            # Show top 10 results
            print(f"\n📊 Top 10 Results:")
            sorted_results = sorted(self.results, key=lambda x: x['steps_per_sec'], reverse=True)
            for i, result in enumerate(sorted_results[:10]):
                config = result['config']
                print(f"  {i+1:2d}. {result['steps_per_sec']:6.2f} steps/s - {config['name']} "
                      f"(C:{config['n_cores']}, E:{config['n_envs']}, "
                      f"B:{config['batch_size']}, S:{config['n_steps']})")
            
            # Apply best configuration
            if self.best_config:
                apply = input(f"\n🔄 Apply best configuration ({self.best_config['name']})? [Y/n]: ").strip().lower()
                if apply not in ['n', 'no']:
                    print(f"🔄 Applying best configuration...")
                    self.modify_parameters(self.best_config)
                    print("✅ Best configuration applied!")
                
        except KeyboardInterrupt:
            print("\n\n⏸️  Optimization interrupted by user")
        except Exception as e:
            print(f"\n❌ Optimization failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Always save results
            if self.results:
                self.save_results()
            
            # Option to restore backup
            if not hasattr(self, 'applied_best') or input("\n🔄 Keep current parameters? [Y/n]: ").strip().lower() in ['n', 'no']:
                self.restore_parameters()

if __name__ == "__main__":
    optimizer = ExtendedTrainingSpeedOptimizer()
    optimizer.optimize()