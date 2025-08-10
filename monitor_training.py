#!/usr/bin/env python3
"""
monitor_training.py - Monitor GPU utilization during training

This script runs training and monitors GPU utilization in real-time.
"""

import os
import subprocess
import time
import threading
from datetime import datetime

def get_gpu_stats():
    """Get current GPU utilization"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw', 
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
                'power': float(parts[4])
            }
    except Exception:
        pass
    return None

def monitor_gpu(duration=300):  # Monitor for 5 minutes
    """Monitor GPU utilization"""
    print("Starting GPU monitoring...")
    print("Time       | GPU% | Memory    | Temp | Power | Status")
    print("-" * 60)
    
    max_util = 0
    avg_util = 0
    sample_count = 0
    start_time = time.time()
    
    while time.time() - start_time < duration:
        gpu_stats = get_gpu_stats()
        current_time = datetime.now().strftime("%H:%M:%S")
        
        if gpu_stats:
            util = gpu_stats['utilization']
            mem_used = gpu_stats['memory_used']
            mem_total = gpu_stats['memory_total']
            temp = gpu_stats['temperature']
            power = gpu_stats['power']
            
            mem_percent = (mem_used / mem_total) * 100
            
            max_util = max(max_util, util)
            avg_util = (avg_util * sample_count + util) / (sample_count + 1)
            sample_count += 1
            
            # Determine status
            if util > 80:
                status = "🔥 HIGH"
            elif util > 50:
                status = "✅ GOOD"
            elif util > 20:
                status = "⚠️  MOD"
            else:
                status = "❌ LOW"
            
            print(f"{current_time} | {util:3d}% | {mem_used:4d}MB/{mem_total:4d}MB | {temp:2d}°C | {power:5.1f}W | {status}")
        else:
            print(f"{current_time} | N/A  | N/A        | N/A  | N/A   | ❌ ERROR")
        
        time.sleep(2)  # Update every 2 seconds
    
    print("-" * 60)
    print(f"SUMMARY after {duration}s:")
    print(f"Max GPU utilization: {max_util}%")
    print(f"Average GPU utilization: {avg_util:.1f}%")
    print(f"Total samples: {sample_count}")

def run_training_with_monitoring():
    """Run training while monitoring GPU"""
    print("GPU UTILIZATION MONITORING DURING TRAINING")
    print("=" * 60)
    
    # Get baseline
    baseline = get_gpu_stats()
    if baseline:
        print(f"Baseline: {baseline['utilization']}% GPU, {baseline['memory_used']}MB, {baseline['temperature']}°C")
    
    print("\nStarting training with optimized parameters:")
    print("- Environments: 128")
    print("- Batch size: 512")
    print("- Rollout steps: 2048")
    print("- Network size: [1024, 512, 256]")
    print("- Mixed precision: Enabled")
    print()
    
    # Start monitoring in a separate thread
    monitor_thread = threading.Thread(target=monitor_gpu, args=(180,))  # 3 minutes
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Run the training
    try:
        # Run a quick training session
        result = subprocess.run(['python', 'model_trainer.py'], 
                              capture_output=True, text=True, timeout=180)
        
        if result.returncode == 0:
            print("\n✅ Training completed successfully!")
        else:
            print(f"\n❌ Training failed with return code {result.returncode}")
            if result.stderr:
                print("Error output:")
                print(result.stderr[-1000:])  # Last 1000 chars of error
    
    except subprocess.TimeoutExpired:
        print("\n⏰ Training timed out after 3 minutes (this is expected for monitoring)")
    
    except Exception as e:
        print(f"\n❌ Error running training: {e}")
    
    # Wait for monitoring to complete
    monitor_thread.join()
    
    # Get final stats
    final_stats = get_gpu_stats()
    if baseline and final_stats:
        util_diff = final_stats['utilization'] - baseline['utilization']
        mem_diff = final_stats['memory_used'] - baseline['memory_used']
        
        print(f"\nFINAL COMPARISON:")
        print(f"GPU utilization change: {util_diff:+d}%")
        print(f"Memory usage change: {mem_diff:+d}MB")
        
        if util_diff > 30:
            print("🎉 EXCELLENT GPU utilization improvement!")
        elif util_diff > 15:
            print("✅ Good GPU utilization improvement!")
        elif util_diff > 5:
            print("⚠️  Moderate improvement")
        else:
            print("❌ Low improvement - may need more optimization")

if __name__ == "__main__":
    run_training_with_monitoring()