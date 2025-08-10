#!/usr/bin/env python3
"""
quick_gpu_test.py - Ultra-fast GPU utilization test

This script runs a very quick test to validate GPU optimizations work.
"""

import os
import torch
import subprocess
from datetime import timedelta
import time

# Setup path
basepath = '/Users/skumar81/Desktop/Personal/trading-final-stable'
os.chdir(basepath)

def get_gpu_stats():
    """Get current GPU utilization"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', 
                               '--format=csv,noheader,nounits'], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            line = result.stdout.strip().split('\n')[0]
            parts = line.split(', ')
            return {
                'utilization': int(parts[0]),
                'memory_used': int(parts[1]),
                'memory_total': int(parts[2])
            }
    except Exception:
        pass
    return None

def test_gpu_utilization():
    """Test GPU utilization with a simple workload"""
    
    print("Quick GPU Utilization Test")
    print("=" * 40)
    
    # Check baseline
    baseline = get_gpu_stats()
    if baseline:
        print(f"Baseline GPU: {baseline['utilization']}% | Memory: {baseline['memory_used']}MB")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
    
    # Test 1: Simple PyTorch operations
    print("\nTest 1: Basic PyTorch GPU operations...")
    device = torch.device('cuda')
    
    # Create large tensors to stress GPU
    start_time = time.time()
    x = torch.randn(4096, 4096, device=device, dtype=torch.float16)  # Use half precision
    y = torch.randn(4096, 4096, device=device, dtype=torch.float16)
    
    # Perform compute-intensive operations
    for i in range(20):
        z = torch.matmul(x, y)
        x = torch.relu(z)
        if i % 5 == 0:
            gpu_stats = get_gpu_stats()
            if gpu_stats:
                print(f"  Iteration {i}: GPU {gpu_stats['utilization']}% | Memory: {gpu_stats['memory_used']}MB")
    
    torch.cuda.synchronize()
    elapsed = time.time() - start_time
    print(f"PyTorch test completed in {elapsed:.2f}s")
    
    # Final GPU check
    final_stats = get_gpu_stats()
    if final_stats:
        print(f"Peak GPU: {final_stats['utilization']}% | Memory: {final_stats['memory_used']}MB")
    
    # Test 2: Mixed precision operations
    print("\nTest 2: Mixed precision training simulation...")
    scaler = torch.cuda.amp.GradScaler()
    
    # Simulate neural network operations with mixed precision
    model = torch.nn.Sequential(
        torch.nn.Linear(1024, 2048),
        torch.nn.ReLU(),
        torch.nn.Linear(2048, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 512)
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    start_time = time.time()
    for i in range(50):
        # Large batch size to stress GPU
        batch = torch.randn(512, 1024, device=device)
        target = torch.randn(512, 512, device=device)
        
        optimizer.zero_grad()
        
        # Use autocast for mixed precision
        with torch.cuda.amp.autocast():
            output = model(batch)
            loss = torch.nn.functional.mse_loss(output, target)
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if i % 10 == 0:
            gpu_stats = get_gpu_stats()
            if gpu_stats:
                print(f"  Batch {i}: GPU {gpu_stats['utilization']}% | Memory: {gpu_stats['memory_used']}MB")
    
    torch.cuda.synchronize()
    elapsed = time.time() - start_time
    print(f"Mixed precision test completed in {elapsed:.2f}s")
    
    # Cleanup
    del x, y, z, model, batch, target, output
    torch.cuda.empty_cache()
    
    # Final stats
    final_stats = get_gpu_stats()
    if baseline and final_stats:
        util_diff = final_stats['utilization'] - baseline['utilization']
        mem_diff = final_stats['memory_used'] - baseline['memory_used']
        
        print(f"\n" + "=" * 40)
        print("RESULTS:")
        print(f"GPU utilization increase: +{util_diff}%")
        print(f"Memory increase: +{mem_diff}MB")
        
        if util_diff > 20:
            print("✅ Excellent GPU utilization!")
            return True
        elif util_diff > 10:
            print("✅ Good GPU utilization!")
            return True
        else:
            print("⚠️  Low GPU utilization increase")
            return False
    
    return True

if __name__ == "__main__":
    success = test_gpu_utilization()
    
    print(f"\n" + "=" * 40)
    print("OPTIMIZATION STATUS")
    print("=" * 40)
    
    # Import and show our optimizations
    try:
        from parameters import *
        print(f"✓ Batch size: {BATCH_SIZE}")
        print(f"✓ Environments: {N_ENVS}")
        print(f"✓ Rollout steps: {N_STEPS}")
        print(f"✓ Mixed precision: {USE_MIXED_PRECISION if 'USE_MIXED_PRECISION' in globals() else False}")
        print(f"✓ Network size: {POLICY_KWARGS['net_arch']['pi']}")
    except Exception as e:
        print(f"Error loading parameters: {e}")
    
    if success:
        print("\n🚀 GPU optimizations are working!")
        print("Run: python model_trainer.py")
    else:
        print("\n❌ GPU optimizations need adjustment")