#!/usr/bin/env python3
"""
intensive_gpu_test.py - Intensive GPU stress test

This script creates a massive workload to truly stress test the RTX 4080.
"""

import os
import torch
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

# Setup path
basepath = '/Users/skumar81/Desktop/Personal/trading-final-stable'
os.chdir(basepath)

def get_gpu_stats():
    """Get current GPU utilization"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', 
                               '--format=csv,noheader,nounits'], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            line = result.stdout.strip().split('\n')[0]
            parts = line.split(', ')
            return {
                'utilization': int(parts[0]),
                'memory_used': int(parts[1]),
                'memory_total': int(parts[2]),
                'temperature': int(parts[3])
            }
    except Exception:
        pass
    return None

def massive_matrix_workload(device, duration=10):
    """Create massive matrix operations to stress GPU"""
    print(f"Starting massive matrix workload on {device} for {duration}s...")
    
    # Create very large tensors that will use significant GPU memory
    sizes = [8192, 4096, 2048]  # Different matrix sizes
    tensors = []
    
    for size in sizes:
        # Create multiple large tensors in FP16 for better throughput
        for _ in range(4):  # 4 tensors per size
            tensor = torch.randn(size, size, device=device, dtype=torch.float16, requires_grad=True)
            tensors.append(tensor)
    
    print(f"Created {len(tensors)} large tensors")
    
    start_time = time.time()
    iteration = 0
    
    while time.time() - start_time < duration:
        # Perform computationally intensive operations
        for i in range(0, len(tensors), 2):
            if i + 1 < len(tensors):
                # Matrix multiplication between consecutive tensors
                result = torch.matmul(tensors[i], tensors[i + 1])
                
                # Apply multiple operations to keep GPU busy
                result = torch.relu(result)
                result = torch.tanh(result)
                
                # Compute gradients to stress GPU further
                loss = result.sum()
                loss.backward(retain_graph=True)
                
                # Update the tensors with gradients (simulating training)
                with torch.no_grad():
                    tensors[i] -= 0.001 * tensors[i].grad
                    tensors[i + 1] -= 0.001 * tensors[i + 1].grad
                    
                # Clear gradients
                tensors[i].grad.zero_()
                tensors[i + 1].grad.zero_()
        
        iteration += 1
        if iteration % 5 == 0:
            gpu_stats = get_gpu_stats()
            if gpu_stats:
                print(f"  Iteration {iteration}: GPU {gpu_stats['utilization']}% | Memory: {gpu_stats['memory_used']}MB | Temp: {gpu_stats['temperature']}°C")
    
    # Cleanup
    del tensors
    torch.cuda.empty_cache()
    
    final_stats = get_gpu_stats()
    if final_stats:
        print(f"Final: GPU {final_stats['utilization']}% | Memory: {final_stats['memory_used']}MB | Temp: {final_stats['temperature']}°C")

def parallel_neural_networks(device, duration=10):
    """Run multiple neural networks in parallel"""
    print(f"Starting parallel neural networks on {device} for {duration}s...")
    
    # Create multiple large neural networks
    networks = []
    optimizers = []
    
    for i in range(8):  # 8 parallel networks
        net = torch.nn.Sequential(
            torch.nn.Linear(2048, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 1024)
        ).to(device)
        
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        
        networks.append(net)
        optimizers.append(optimizer)
    
    print(f"Created {len(networks)} parallel neural networks")
    
    start_time = time.time()
    iteration = 0
    
    while time.time() - start_time < duration:
        for i, (net, optimizer) in enumerate(zip(networks, optimizers)):
            # Large batch size to stress GPU
            batch_size = 1024
            input_data = torch.randn(batch_size, 2048, device=device, dtype=torch.float16)
            target = torch.randn(batch_size, 1024, device=device, dtype=torch.float16)
            
            optimizer.zero_grad()
            
            # Use mixed precision
            with torch.amp.autocast('cuda'):
                output = net(input_data)
                loss = torch.nn.functional.mse_loss(output, target)
            
            loss.backward()
            optimizer.step()
        
        iteration += 1
        if iteration % 2 == 0:
            gpu_stats = get_gpu_stats()
            if gpu_stats:
                print(f"  Iteration {iteration}: GPU {gpu_stats['utilization']}% | Memory: {gpu_stats['memory_used']}MB | Temp: {gpu_stats['temperature']}°C")
    
    # Cleanup
    del networks, optimizers
    torch.cuda.empty_cache()

def run_intensive_gpu_test():
    """Run intensive GPU stress test"""
    
    print("INTENSIVE GPU STRESS TEST FOR RTX 4080")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    device = torch.device('cuda')
    print(f"✓ GPU: {torch.cuda.get_device_name(device)}")
    
    # Get baseline
    baseline = get_gpu_stats()
    if baseline:
        print(f"Baseline: {baseline['utilization']}% | {baseline['memory_used']}MB | {baseline['temperature']}°C")
    
    # Test 1: Massive matrix operations
    print(f"\n{'='*50}")
    print("TEST 1: MASSIVE MATRIX OPERATIONS")
    print(f"{'='*50}")
    massive_matrix_workload(device, duration=15)
    
    # Cool down
    print("\nCooling down for 3 seconds...")
    time.sleep(3)
    torch.cuda.empty_cache()
    
    # Test 2: Parallel neural networks
    print(f"\n{'='*50}")
    print("TEST 2: PARALLEL NEURAL NETWORKS")
    print(f"{'='*50}")
    parallel_neural_networks(device, duration=15)
    
    # Cool down
    print("\nCooling down for 3 seconds...")
    time.sleep(3)
    torch.cuda.empty_cache()
    
    # Test 3: Combined workload
    print(f"\n{'='*50}")
    print("TEST 3: COMBINED WORKLOAD")
    print(f"{'='*50}")
    
    def run_combined():
        massive_matrix_workload(device, duration=10)
        parallel_neural_networks(device, duration=10)
    
    run_combined()
    
    # Final stats
    final_stats = get_gpu_stats()
    if baseline and final_stats:
        util_diff = final_stats['utilization'] - baseline['utilization']
        mem_diff = final_stats['memory_used'] - baseline['memory_used']
        temp_diff = final_stats['temperature'] - baseline['temperature']
        
        print(f"\n{'='*50}")
        print("STRESS TEST RESULTS")
        print(f"{'='*50}")
        print(f"GPU utilization increase: +{util_diff}%")
        print(f"Memory increase: +{mem_diff}MB")
        print(f"Temperature increase: +{temp_diff}°C")
        
        if util_diff > 50:
            print("🔥 EXCELLENT! GPU is being fully utilized!")
            return True
        elif util_diff > 30:
            print("✅ GOOD! GPU utilization significantly improved!")
            return True
        elif util_diff > 10:
            print("⚠️  MODERATE GPU utilization improvement")
            return False
        else:
            print("❌ LOW GPU utilization - needs more optimization")
            return False
    
    return False

if __name__ == "__main__":
    print("This test will stress your RTX 4080 to maximum capacity.")
    print("Make sure you have adequate cooling!")
    print("Starting test in 2 seconds...")
    time.sleep(2)
    
    success = run_intensive_gpu_test()
    
    if success:
        print("\n🚀 Your GPU can handle intensive workloads!")
        print("The trading system optimizations should work well.")
    else:
        print("\n🔧 Need more aggressive optimizations for this GPU.")
        
    print("\nCheck nvidia-smi for real-time monitoring during actual training.")