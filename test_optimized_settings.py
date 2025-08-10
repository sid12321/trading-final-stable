#!/usr/bin/env python3
"""
test_optimized_settings.py - Test the memory-optimized GPU settings

This script tests the new balanced approach for GPU utilization.
"""

import os
import subprocess
import time

# Setup path
basepath = '/Users/skumar81/Desktop/Personal/trading-final-stable'
os.chdir(basepath)

def get_gpu_info():
    """Get detailed GPU information"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,pstate', 
                               '--format=csv,noheader,nounits'], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            line = result.stdout.strip().split('\n')[0]
            parts = line.split(', ')
            return {
                'utilization': int(parts[0]),
                'memory_used': int(parts[1]),
                'memory_total': int(parts[2]),
                'temperature': int(parts[3]),
                'power': float(parts[4]),
                'pstate': parts[5]
            }
    except Exception:
        pass
    return None

def count_gpu_processes():
    """Count the number of GPU processes"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            lines = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
            return len(lines)
    except Exception:
        pass
    return 0

def show_optimized_settings():
    """Display the current optimized settings"""
    try:
        import parameters
        print("OPTIMIZED SETTINGS SUMMARY")
        print("=" * 50)
        print(f"Environment Settings:")
        print(f"  N_ENVS: {parameters.N_ENVS} (reduced from 128 to prevent memory fragmentation)")
        print(f"  N_CORES: {parameters.N_CORES}")
        print(f"  DEVICE: {parameters.DEVICE}")
        print()
        print(f"Training Parameters:")
        print(f"  BATCH_SIZE: {parameters.BATCH_SIZE} (increased for better GPU utilization)")
        print(f"  N_STEPS: {parameters.N_STEPS} (increased to compensate for fewer environments)")
        print(f"  N_EPOCHS: {parameters.N_EPOCHS}")
        print(f"  BASEMODELITERATIONS: {parameters.BASEMODELITERATIONS}")
        print()
        print(f"Network Architecture:")
        print(f"  Policy network: {parameters.POLICY_KWARGS['net_arch']['pi']}")
        print(f"  Value network: {parameters.POLICY_KWARGS['net_arch']['vf']}")
        print()
        print(f"GPU Optimization:")
        print(f"  USE_MIXED_PRECISION: {getattr(parameters, 'USE_MIXED_PRECISION', False)}")
        print(f"  Expected total timesteps per rollout: {parameters.N_STEPS * parameters.N_ENVS}")
        
        return True
    except Exception as e:
        print(f"Error loading parameters: {e}")
        return False

def monitor_quick_training():
    """Monitor a quick training session"""
    print("\nSTARTING MEMORY-OPTIMIZED TRAINING TEST")
    print("=" * 50)
    
    # Get baseline
    baseline = get_gpu_info()
    baseline_processes = count_gpu_processes()
    
    if baseline:
        print(f"Baseline GPU: {baseline['utilization']}% | Memory: {baseline['memory_used']}MB | Processes: {baseline_processes}")
    
    print("\nStarting training with optimized settings...")
    print("Monitoring for 90 seconds...")
    print()
    print("Time     | GPU% | Memory      | Procs | Temp | Power | Status")
    print("-" * 65)
    
    # Start training process
    training_process = subprocess.Popen(['python', 'model_trainer.py'], 
                                      stdout=subprocess.PIPE, 
                                      stderr=subprocess.PIPE)
    
    max_util = 0
    max_memory = 0
    max_processes = 0
    sample_count = 0
    
    start_time = time.time()
    
    try:
        while time.time() - start_time < 90 and training_process.poll() is None:
            gpu_info = get_gpu_info()
            gpu_processes = count_gpu_processes()
            
            if gpu_info:
                util = gpu_info['utilization']
                memory = gpu_info['memory_used']
                temp = gpu_info['temperature']
                power = gpu_info['power']
                pstate = gpu_info['pstate']
                
                max_util = max(max_util, util)
                max_memory = max(max_memory, memory)
                max_processes = max(max_processes, gpu_processes)
                sample_count += 1
                
                # Status based on utilization
                if util > 50:
                    status = "🔥 HIGH"
                elif util > 20:
                    status = "✅ GOOD"
                elif util > 5:
                    status = "⚠️  MOD"
                else:
                    status = "❌ LOW"
                
                elapsed = int(time.time() - start_time)
                print(f"{elapsed:3d}s     | {util:3d}% | {memory:4d}MB/{gpu_info['memory_total']:4d}MB | {gpu_processes:3d}   | {temp:2d}°C | {power:5.1f}W | {status}")
            else:
                elapsed = int(time.time() - start_time)
                print(f"{elapsed:3d}s     | N/A  | N/A         | N/A   | N/A  | N/A   | ❌ ERROR")
            
            time.sleep(3)
        
        # Terminate training if still running
        if training_process.poll() is None:
            training_process.terminate()
            training_process.wait(timeout=10)
    
    except KeyboardInterrupt:
        print("\nMonitoring interrupted by user")
        training_process.terminate()
    
    except Exception as e:
        print(f"\nError during monitoring: {e}")
        training_process.terminate()
    
    print("-" * 65)
    print("MONITORING SUMMARY")
    print("-" * 65)
    print(f"Maximum GPU utilization: {max_util}%")
    print(f"Maximum memory usage: {max_memory}MB")
    print(f"Maximum GPU processes: {max_processes}")
    print(f"Total monitoring samples: {sample_count}")
    
    if baseline:
        util_improvement = max_util - baseline['utilization']
        memory_increase = max_memory - baseline['memory_used']
        process_increase = max_processes - baseline_processes
        
        print(f"\nCOMPARISON TO BASELINE:")
        print(f"GPU utilization improvement: +{util_improvement}%")
        print(f"Memory usage increase: +{memory_increase}MB")
        print(f"Process count increase: +{process_increase}")
        
        # Assessment
        if max_processes < 50 and max_util > 30:
            print("\n✅ EXCELLENT: Good utilization with controlled process count!")
        elif max_processes < 50 and max_util > 15:
            print("\n✅ GOOD: Reasonable utilization with controlled process count!")
        elif max_processes > 40:
            print("\n⚠️  WARNING: Too many processes may cause memory fragmentation")
        else:
            print("\n❌ NEEDS IMPROVEMENT: Low utilization")

if __name__ == "__main__":
    print("MEMORY-OPTIMIZED GPU UTILIZATION TEST")
    print("=" * 50)
    
    # Show current settings
    if show_optimized_settings():
        print("\n✅ Settings loaded successfully")
    else:
        print("\n❌ Error loading settings")
        exit(1)
    
    # Monitor training
    monitor_quick_training()
    
    print(f"\n" + "=" * 50)
    print("RECOMMENDATIONS")
    print("=" * 50)
    print("1. If processes < 50 and GPU util > 30%: Settings are optimal!")
    print("2. If processes > 40: Reduce N_ENVS further")
    print("3. If GPU util < 15%: Increase BATCH_SIZE or network size")
    print("4. Monitor full training with: nvidia-smi -l 2")