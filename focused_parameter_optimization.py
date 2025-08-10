#!/usr/bin/env python3
"""
focused_parameter_optimization.py - Focused optimization on 3 key parameters

Based on previous optimization results, fix the optimal parameters:
- N_CORES = 1 (optimal from plots)
- N_ENVS = 1 (optimal from plots) 
- N_STEPS = 3072 (optimal from plots)

Then systematically optimize:
1. EPOCHS (1, 2, 3, 4, 6, 8, 10, 12)
2. BATCH_SIZE (128, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024)
3. NETWORK_ARCH (various architectures)

One parameter at a time to find the absolute optimum.
"""

import os
import sys
import time
import subprocess
import json
import numpy as np
from datetime import datetime
from pathlib import Path

# Setup path
basepath = '/Users/skumar81/Desktop/Personal/trading-final-stable'
os.chdir(basepath)

class FocusedOptimizer:
    def __init__(self):
        self.results = []
        self.best_config = None
        self.best_steps_per_sec = 0
        self.test_duration = 30  # seconds per test
        self.results_file = "focused_optimization_results.json"
        
        # FIXED OPTIMAL PARAMETERS (from previous optimization)
        self.FIXED_PARAMS = {
            "n_cores": 1,      # Optimal from plots
            "n_envs": 1,       # Optimal from plots  
            "n_steps": 3072,   # Optimal from plots
        }
        
        # BASE CONFIGURATION (will be varied)
        self.BASE_CONFIG = {
            "n_epochs": 4,           # Will vary this
            "batch_size": 256,       # Will vary this
            "net_arch": [256, 256],  # Will vary this
            **self.FIXED_PARAMS
        }
        
    def backup_parameters(self):
        """Backup original parameters.py"""
        import shutil
        shutil.copy("parameters.py", "parameters_backup_focused.py")
        print("✅ Backed up original parameters.py")
    
    def restore_parameters(self):
        """Restore original parameters.py"""
        import shutil
        if os.path.exists("parameters_backup_focused.py"):
            shutil.copy("parameters_backup_focused.py", "parameters.py")
            print("✅ Restored original parameters.py")
    
    def modify_parameters(self, config):
        """Modify parameters.py with test configuration"""
        with open("parameters.py", "r") as f:
            content = f.read()
        
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            # Handle MPS core settings
            if 'N_CORES = min(mp.cpu_count(),' in line and 'elif DEVICE == "mps":' in content[content.find('\n'.join(lines[:i])):content.find('\n'.join(lines[:i+10]))]:
                lines[i] = f'    N_CORES = min(mp.cpu_count(), {config["n_cores"]})  # OPTIMIZED: Single core best for Apple Silicon'
            elif 'N_ENVS = min(mp.cpu_count(),' in line and 'elif DEVICE == "mps":' in content[content.find('\n'.join(lines[:i])):content.find('\n'.join(lines[:i+10]))]:
                lines[i] = f'    N_ENVS = min(mp.cpu_count(), {config["n_envs"]})  # OPTIMIZED: Single env best for Apple Silicon'
            # Handle global parameters
            elif line.strip().startswith('BATCH_SIZE ='):
                lines[i] = f'BATCH_SIZE = {config["batch_size"]}   # OPTIMIZED batch size'
            elif line.strip().startswith('N_STEPS ='):
                lines[i] = f'N_STEPS = {config["n_steps"]}  # OPTIMIZED rollout buffer (3072 is optimal)'
            elif line.strip().startswith('N_EPOCHS ='):
                lines[i] = f'N_EPOCHS = {config["n_epochs"]}  # OPTIMIZED epochs'
        
        content = '\n'.join(lines)
        
        # Handle network architecture
        if 'net_arch' in config:
            net_arch = config['net_arch']
            new_policy_kwargs = f"""POLICY_KWARGS = {{
    'activation_fn': 'ReLU',  # Will be converted to torch.nn.ReLU in modeltrain
    'net_arch': {{
        'pi': {net_arch},  # OPTIMIZED network architecture
        'vf': {net_arch}   # OPTIMIZED network architecture
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
    
    def run_training_test(self, config, test_name):
        """Run a training test and measure steps/s"""
        print(f"\n🧪 {test_name}")
        print(f"   Config: cores={config['n_cores']}, envs={config['n_envs']}, "
              f"batch={config['batch_size']}, steps={config['n_steps']}, "
              f"epochs={config['n_epochs']}, net={config.get('net_arch', 'default')}")
        
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
                    # Wait for actual training to start
                    if "Training BPCL:" in output:
                        training_started = True
                    
                    # Look for steps/s in the output
                    if training_started and "steps/s" in output:
                        try:
                            parts = output.strip().split(',')
                            if len(parts) >= 2:
                                steps_part = parts[-1]
                                if 'steps/s]' in steps_part:
                                    steps_value = float(steps_part.split('steps/s')[0].strip())
                                    steps_per_sec_readings.append(steps_value)
                                    print(f"   📊 {steps_value:.2f} steps/s")
                        except (ValueError, IndexError):
                            continue
                
                if process.poll() is not None:
                    break
            
            process.terminate()
            process.wait(timeout=5)
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            process.terminate()
            return 0
        
        # Calculate stable average (skip first 2 readings for warmup)
        if len(steps_per_sec_readings) >= 3:
            stable_readings = steps_per_sec_readings[2:]
            avg_steps_per_sec = sum(stable_readings) / len(stable_readings)
            print(f"   📈 Average: {avg_steps_per_sec:.2f} steps/s (from {len(stable_readings)} stable readings)")
        elif len(steps_per_sec_readings) > 0:
            avg_steps_per_sec = sum(steps_per_sec_readings) / len(steps_per_sec_readings)
            print(f"   📈 Average: {avg_steps_per_sec:.2f} steps/s (from {len(steps_per_sec_readings)} readings)")
        else:
            avg_steps_per_sec = 0
            print(f"   ❌ No valid readings captured")
        
        return avg_steps_per_sec
    
    def save_results(self):
        """Save optimization results"""
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'fixed_parameters': self.FIXED_PARAMS,
            'best_config': self.best_config,
            'best_steps_per_sec': self.best_steps_per_sec,
            'all_results': self.results
        }
        
        with open(self.results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n💾 Results saved to {self.results_file}")
    
    def optimize_epochs(self):
        """Optimize number of epochs"""
        print(f"\n🔍 PHASE 1: Optimizing EPOCHS")
        print("=" * 50)
        
        epoch_values = [1, 2, 3, 4, 5, 6, 8, 10, 12]  # Comprehensive range
        
        best_epochs = self.BASE_CONFIG["n_epochs"]
        best_epoch_performance = 0
        
        for epochs in epoch_values:
            config = {**self.BASE_CONFIG, "n_epochs": epochs}
            
            steps_per_sec = self.run_training_test(config, f"EPOCHS TEST: {epochs} epochs")
            
            result = {
                'phase': 'epochs',
                'parameter': 'n_epochs',
                'value': epochs,
                'config': config,
                'steps_per_sec': steps_per_sec,
                'timestamp': datetime.now().isoformat()
            }
            
            self.results.append(result)
            
            if steps_per_sec > best_epoch_performance:
                best_epoch_performance = steps_per_sec
                best_epochs = epochs
                print(f"   🏆 NEW BEST EPOCHS: {epochs} → {steps_per_sec:.2f} steps/s")
            
            if steps_per_sec > self.best_steps_per_sec:
                self.best_steps_per_sec = steps_per_sec
                self.best_config = config.copy()
            
            self.save_results()
        
        # Update base config with best epochs
        self.BASE_CONFIG["n_epochs"] = best_epochs
        print(f"\n✅ EPOCHS OPTIMIZATION COMPLETE")
        print(f"   🏆 Best epochs: {best_epochs} → {best_epoch_performance:.2f} steps/s")
        return best_epochs, best_epoch_performance
    
    def optimize_batch_size(self):
        """Optimize batch size"""
        print(f"\n🔍 PHASE 2: Optimizing BATCH SIZE")
        print("=" * 50)
        
        batch_values = [128, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024]  # Comprehensive range
        
        best_batch = self.BASE_CONFIG["batch_size"] 
        best_batch_performance = 0
        
        for batch_size in batch_values:
            config = {**self.BASE_CONFIG, "batch_size": batch_size}
            
            steps_per_sec = self.run_training_test(config, f"BATCH TEST: {batch_size} batch size")
            
            result = {
                'phase': 'batch_size',
                'parameter': 'batch_size', 
                'value': batch_size,
                'config': config,
                'steps_per_sec': steps_per_sec,
                'timestamp': datetime.now().isoformat()
            }
            
            self.results.append(result)
            
            if steps_per_sec > best_batch_performance:
                best_batch_performance = steps_per_sec
                best_batch = batch_size
                print(f"   🏆 NEW BEST BATCH: {batch_size} → {steps_per_sec:.2f} steps/s")
            
            if steps_per_sec > self.best_steps_per_sec:
                self.best_steps_per_sec = steps_per_sec
                self.best_config = config.copy()
            
            self.save_results()
        
        # Update base config with best batch size
        self.BASE_CONFIG["batch_size"] = best_batch
        print(f"\n✅ BATCH SIZE OPTIMIZATION COMPLETE")
        print(f"   🏆 Best batch size: {best_batch} → {best_batch_performance:.2f} steps/s")
        return best_batch, best_batch_performance
    
    def optimize_network_architecture(self):
        """Optimize network architecture"""
        print(f"\n🔍 PHASE 3: Optimizing NETWORK ARCHITECTURE")
        print("=" * 50)
        
        # Network architectures to test
        network_configs = [
            # Small networks (fast)
            ([64, 64], "64x64_Tiny"),
            ([96, 64], "96x64_Small"), 
            ([128, 64], "128x64_Compact"),
            ([128, 128], "128x128_Small"),
            
            # Medium networks (balanced)
            ([192, 128], "192x128_Medium"),
            ([256, 128], "256x128_MediumWide"),
            ([256, 256], "256x256_Standard"),
            ([320, 256], "320x256_Wide"),
            
            # Large networks (powerful but slower)
            ([384, 256], "384x256_Large"),
            ([512, 256], "512x256_XLarge"),
            ([256, 256, 128], "256x256x128_Deep"),
            ([384, 256, 128], "384x256x128_DeepWide"),
            
            # Asymmetric networks
            ([512, 128], "512x128_Asymmetric"),
            ([400, 200], "400x200_Asymmetric2"),
        ]
        
        best_arch = self.BASE_CONFIG["net_arch"]
        best_arch_performance = 0
        best_arch_name = "Unknown"
        
        for net_arch, name in network_configs:
            config = {**self.BASE_CONFIG, "net_arch": net_arch}
            
            steps_per_sec = self.run_training_test(config, f"NETWORK TEST: {name} {net_arch}")
            
            result = {
                'phase': 'network_arch',
                'parameter': 'net_arch',
                'value': net_arch,
                'value_name': name,
                'config': config,
                'steps_per_sec': steps_per_sec,
                'timestamp': datetime.now().isoformat()
            }
            
            self.results.append(result)
            
            if steps_per_sec > best_arch_performance:
                best_arch_performance = steps_per_sec
                best_arch = net_arch
                best_arch_name = name
                print(f"   🏆 NEW BEST NETWORK: {name} {net_arch} → {steps_per_sec:.2f} steps/s")
            
            if steps_per_sec > self.best_steps_per_sec:
                self.best_steps_per_sec = steps_per_sec
                self.best_config = config.copy()
            
            self.save_results()
        
        # Update base config with best network
        self.BASE_CONFIG["net_arch"] = best_arch
        print(f"\n✅ NETWORK ARCHITECTURE OPTIMIZATION COMPLETE")
        print(f"   🏆 Best network: {best_arch_name} {best_arch} → {best_arch_performance:.2f} steps/s")
        return best_arch, best_arch_performance, best_arch_name
    
    def optimize(self):
        """Run focused optimization"""
        print("🚀 FOCUSED PARAMETER OPTIMIZATION")
        print("=" * 60)
        print("Fixed optimal parameters:")
        for key, value in self.FIXED_PARAMS.items():
            print(f"  {key}: {value}")
        print("\nOptimizing: epochs, batch_size, network_architecture")
        print("=" * 60)
        
        self.backup_parameters()
        
        try:
            # Phase 1: Optimize epochs
            best_epochs, epoch_perf = self.optimize_epochs()
            
            # Phase 2: Optimize batch size (with optimal epochs)
            best_batch, batch_perf = self.optimize_batch_size()
            
            # Phase 3: Optimize network architecture (with optimal epochs + batch)
            best_arch, arch_perf, arch_name = self.optimize_network_architecture()
            
            print("\n" + "=" * 60)
            print("🏁 FOCUSED OPTIMIZATION COMPLETE!")
            print("=" * 60)
            
            print(f"\n🎯 FINAL OPTIMAL CONFIGURATION:")
            print(f"  Fixed parameters:")
            for key, value in self.FIXED_PARAMS.items():
                print(f"    {key}: {value}")
            print(f"  Optimized parameters:")
            print(f"    n_epochs: {best_epochs}")
            print(f"    batch_size: {best_batch}")
            print(f"    net_arch: {best_arch} ({arch_name})")
            
            print(f"\n📈 PERFORMANCE PROGRESSION:")
            print(f"  Phase 1 (epochs): {epoch_perf:.2f} steps/s")
            print(f"  Phase 2 (+ batch): {batch_perf:.2f} steps/s")  
            print(f"  Phase 3 (+ network): {arch_perf:.2f} steps/s")
            print(f"  🏆 FINAL BEST: {self.best_steps_per_sec:.2f} steps/s")
            
            # Apply final optimal configuration
            if self.best_config:
                apply = input(f"\n🔄 Apply final optimal configuration? [Y/n]: ").strip().lower()
                if apply not in ['n', 'no']:
                    self.modify_parameters(self.best_config)
                    print("✅ Final optimal configuration applied!")
                    
                    print(f"\n🎊 OPTIMIZATION SUCCESS!")
                    print(f"Your training speed should now be: {self.best_steps_per_sec:.2f} steps/s")
                
        except KeyboardInterrupt:
            print("\n\n⏸️  Optimization interrupted by user")
        except Exception as e:
            print(f"\n❌ Optimization failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.results:
                self.save_results()
            
            # Option to restore backup
            if input("\n🔄 Keep optimized parameters? [Y/n]: ").strip().lower() in ['n', 'no']:
                self.restore_parameters()

if __name__ == "__main__":
    optimizer = FocusedOptimizer()
    optimizer.optimize()