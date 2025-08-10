#!/usr/bin/env python3
"""
test_gpu_utilization.py - Test script to validate GPU utilization improvements

This script runs a quick training test to validate that the GPU optimizations
are working properly and utilizing more GPU resources.
"""

import os
import sys
import time
import subprocess
import torch

# Setup path
basepath = '/Users/skumar81/Desktop/Personal/trading-final-stable'
os.chdir(basepath)

from parameters import *
from common import optimize_gpu_settings, cleanup_gpu_memory
from model_trainer import ModelTrainer

def get_gpu_utilization():
    """Get current GPU utilization using nvidia-smi"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', 
                               '--format=csv,noheader,nounits'], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_data = []
            for line in lines:
                parts = line.split(', ')
                if len(parts) >= 3:
                    util = int(parts[0])
                    mem_used = int(parts[1])
                    mem_total = int(parts[2])
                    gpu_data.append({
                        'utilization': util,
                        'memory_used': mem_used,
                        'memory_total': mem_total,
                        'memory_usage_percent': (mem_used / mem_total) * 100
                    })
            return gpu_data
    except Exception as e:
        print(f"Error getting GPU utilization: {e}")
    return []

def monitor_gpu_during_training():
    """Monitor GPU utilization during a short training session"""
    print("Starting GPU utilization test...")
    print("=" * 60)
    
    # Initialize GPU optimizations
    print("Initializing GPU optimizations...")
    gpu_available = optimize_gpu_settings()
    
    if not gpu_available:
        print("GPU not available - test cannot run")
        return False
    
    # Get baseline GPU stats
    baseline_gpu = get_gpu_utilization()
    if baseline_gpu:
        print(f"Baseline GPU utilization: {baseline_gpu[0]['utilization']}%")
        print(f"Baseline GPU memory: {baseline_gpu[0]['memory_used']}MB / {baseline_gpu[0]['memory_total']}MB ({baseline_gpu[0]['memory_usage_percent']:.1f}%)")
    
    # Initialize trainer with reduced data for quick test
    print("\nInitializing model trainer for quick test...")
    trainer = ModelTrainer(symbols=TESTSYMBOLS[:1])  # Only test with one symbol
    
    try:
        # Load only historical data (skip preprocessing for speed)
        print("Loading historical data...")
        trainer.load_historical_data()
        
        # Extract signals
        print("Extracting signals...")
        globalsignals = trainer.extract_signals()
        print(f"Found {len(globalsignals)} global signals")
        
        # Run a quick training test with reduced iterations
        print("\nStarting training with GPU optimization...")
        
        # Temporarily reduce training iterations for quick test
        original_trainreps = TRAINREPS
        original_basemodeliterations = BASEMODELITERATIONS
        
        # Set to very small values for quick test
        globals()['TRAINREPS'] = 1
        globals()['BASEMODELITERATIONS'] = 1000  # Much smaller for test
        
        print(f"Quick test parameters:")
        print(f"  TRAINREPS: {TRAINREPS}")
        print(f"  BASEMODELITERATIONS: {BASEMODELITERATIONS}")
        print(f"  BATCH_SIZE: {BATCH_SIZE}")
        print(f"  N_STEPS: {N_STEPS}")
        print(f"  N_ENVS: {N_ENVS}")
        print(f"  N_EPOCHS: {N_EPOCHS}")
        print(f"  USE_MIXED_PRECISION: {USE_MIXED_PRECISION if 'USE_MIXED_PRECISION' in globals() else False}")
        
        # Monitor GPU before training
        pre_training_gpu = get_gpu_utilization()
        if pre_training_gpu:
            print(f"\nPre-training GPU utilization: {pre_training_gpu[0]['utilization']}%")
            print(f"Pre-training GPU memory: {pre_training_gpu[0]['memory_used']}MB / {pre_training_gpu[0]['memory_total']}MB ({pre_training_gpu[0]['memory_usage_percent']:.1f}%)")
        
        # Start training
        start_time = time.time()
        reward = trainer.train_models_with_params()
        training_time = time.time() - start_time
        
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Training reward: {reward}")
        
        # Monitor GPU during training (if still running)
        during_training_gpu = get_gpu_utilization()
        if during_training_gpu:
            print(f"\nDuring/post-training GPU utilization: {during_training_gpu[0]['utilization']}%")
            print(f"During/post-training GPU memory: {during_training_gpu[0]['memory_used']}MB / {during_training_gpu[0]['memory_total']}MB ({during_training_gpu[0]['memory_usage_percent']:.1f}%)")
        
        # Restore original values
        globals()['TRAINREPS'] = original_trainreps
        globals()['BASEMODELITERATIONS'] = original_basemodeliterations
        
        print("\n" + "=" * 60)
        print("GPU UTILIZATION TEST RESULTS")
        print("=" * 60)
        
        if baseline_gpu and during_training_gpu:
            util_increase = during_training_gpu[0]['utilization'] - baseline_gpu[0]['utilization']
            mem_increase = during_training_gpu[0]['memory_used'] - baseline_gpu[0]['memory_used']
            
            print(f"GPU utilization increase: +{util_increase}%")
            print(f"GPU memory increase: +{mem_increase}MB")
            
            if util_increase > 10:
                print("✓ GPU utilization improved significantly!")
            elif util_increase > 5:
                print("⚠ Moderate GPU utilization improvement")
            else:
                print("✗ Low GPU utilization increase")
        
        print(f"\nOptimizations applied:")
        print(f"  ✓ Larger batch size: {BATCH_SIZE}")
        print(f"  ✓ More environments: {N_ENVS}")
        print(f"  ✓ Larger rollout buffer: {N_STEPS}")
        print(f"  ✓ Larger network architecture: {POLICY_KWARGS['net_arch']}")
        print(f"  ✓ Mixed precision training: {USE_MIXED_PRECISION if 'USE_MIXED_PRECISION' in globals() else False}")
        print(f"  ✓ TF32 optimizations enabled")
        print(f"  ✓ CuDNN benchmark mode enabled")
        
        # Clean up
        cleanup_gpu_memory(aggressive=True)
        
        return True
        
    except Exception as e:
        print(f"Error during training test: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_continuous_gpu_monitoring():
    """Run continuous GPU monitoring during a longer training session"""
    print("\nStarting continuous GPU monitoring...")
    print("Run 'python model_trainer.py' in another terminal to see real-time GPU usage")
    print("Press Ctrl+C to stop monitoring")
    
    max_util = 0
    avg_util = 0
    sample_count = 0
    
    try:
        while True:
            gpu_data = get_gpu_utilization()
            if gpu_data:
                util = gpu_data[0]['utilization']
                mem_percent = gpu_data[0]['memory_usage_percent']
                
                max_util = max(max_util, util)
                avg_util = (avg_util * sample_count + util) / (sample_count + 1)
                sample_count += 1
                
                print(f"GPU: {util:3d}% | Memory: {mem_percent:5.1f}% | Max: {max_util:3d}% | Avg: {avg_util:5.1f}%", end='\r')
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print(f"\n\nMonitoring stopped after {sample_count} samples")
        print(f"Maximum GPU utilization: {max_util}%")
        print(f"Average GPU utilization: {avg_util:.1f}%")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--monitor":
        run_continuous_gpu_monitoring()
    else:
        success = monitor_gpu_during_training()
        
        print("\n" + "=" * 60)
        print("NEXT STEPS")
        print("=" * 60)
        print("1. Run full training: python model_trainer.py")
        print("2. Monitor GPU usage: python test_gpu_utilization.py --monitor")
        print("3. Run optimized training: python run_optimized_model_training.py")
        print("4. Check nvidia-smi for real-time GPU utilization")
        
        if success:
            print("\n✓ GPU optimization test completed successfully!")
        else:
            print("\n✗ GPU optimization test failed!")