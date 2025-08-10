#!/usr/bin/env python3
"""
Performance monitoring script for M4 Max optimization

Monitors CPU, GPU, memory, and training throughput during model training.
Run this in a separate terminal while training to track resource utilization.

Usage:
    python monitor_performance.py
    python monitor_performance.py --interval 5  # Update every 5 seconds
"""

import time
import psutil
import argparse
import subprocess
import sys
from datetime import datetime
import os

def get_system_info():
    """Get basic system information"""
    try:
        # Get chip info
        chip_info = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                 capture_output=True, text=True).stdout.strip()
        
        # Get memory info
        mem_bytes = int(subprocess.run(['sysctl', '-n', 'hw.memsize'], 
                                      capture_output=True, text=True).stdout.strip())
        total_memory_gb = mem_bytes / (1024**3)
        
        return {
            'chip': chip_info,
            'total_memory_gb': total_memory_gb,
            'cpu_count': psutil.cpu_count(),
            'cpu_count_logical': psutil.cpu_count(logical=True)
        }
    except:
        return {
            'chip': 'Unknown',
            'total_memory_gb': 0,
            'cpu_count': psutil.cpu_count(),
            'cpu_count_logical': psutil.cpu_count(logical=True)
        }

def get_training_processes():
    """Find Python training processes"""
    training_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info']):
        try:
            if proc.info['name'] == 'python' and proc.info['cmdline']:
                cmdline = ' '.join(proc.info['cmdline'])
                if any(keyword in cmdline for keyword in ['model_trainer', 'train_only', 'train.py']):
                    training_processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return training_processes

def get_gpu_utilization():
    """Get Metal GPU utilization (approximate using system resources)"""
    try:
        # On macOS, we can check GPU pressure through system stats
        result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], 
                               capture_output=True, text=True, timeout=2)
        # This is a placeholder - Metal doesn't expose detailed utilization like nvidia-smi
        return "Metal GPU Active (detailed metrics not available)"
    except:
        return "GPU metrics unavailable"

def monitor_performance(interval=3):
    """Monitor system performance during training"""
    
    # Get system info
    sys_info = get_system_info()
    
    print("🔥 M4 Max Performance Monitor")
    print("=" * 60)
    print(f"Chip: {sys_info['chip']}")
    print(f"CPU Cores: {sys_info['cpu_count']} physical, {sys_info['cpu_count_logical']} logical")
    print(f"Total Memory: {sys_info['total_memory_gb']:.1f} GB")
    print(f"Update Interval: {interval}s")
    print("=" * 60)
    print()
    
    try:
        iteration = 0
        while True:
            iteration += 1
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # Get system stats
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            memory_used_gb = memory.used / (1024**3)
            memory_percent = memory.percent
            
            # Get per-core CPU usage
            cpu_per_core = psutil.cpu_percent(percpu=True, interval=0.1)
            avg_cpu_per_core = sum(cpu_per_core) / len(cpu_per_core)
            
            # Get training processes
            training_procs = get_training_processes()
            
            # Clear screen on each update (except first)
            if iteration > 1:
                os.system('clear' if os.name == 'posix' else 'cls')
                print("🔥 M4 Max Performance Monitor")
                print("=" * 60)
                print(f"Chip: {sys_info['chip']}")
                print(f"CPU Cores: {sys_info['cpu_count']} physical, {sys_info['cpu_count_logical']} logical")
                print(f"Total Memory: {sys_info['total_memory_gb']:.1f} GB")
                print(f"Update Interval: {interval}s")
                print("=" * 60)
                print()
            
            print(f"⏰ {timestamp} - Update #{iteration}")
            print("-" * 40)
            
            # Overall system utilization
            print(f"🔥 CPU Utilization: {cpu_percent:.1f}% (avg per core: {avg_cpu_per_core:.1f}%)")
            print(f"🧠 Memory Usage: {memory_used_gb:.1f}GB / {sys_info['total_memory_gb']:.1f}GB ({memory_percent:.1f}%)")
            print(f"⚡ GPU Status: {get_gpu_utilization()}")
            
            # Per-core breakdown
            print(f"📊 Per-Core CPU Usage:")
            cores_per_line = 8
            for i in range(0, len(cpu_per_core), cores_per_line):
                core_group = cpu_per_core[i:i+cores_per_line]
                core_strs = [f"C{i+j}:{usage:.0f}%" for j, usage in enumerate(core_group)]
                print(f"   {' | '.join(core_strs)}")
            
            # Training process details
            if training_procs:
                print(f"\n🎯 Training Processes ({len(training_procs)} found):")
                for proc in training_procs:
                    try:
                        cpu_usage = proc.cpu_percent()
                        memory_mb = proc.memory_info().rss / (1024*1024)
                        cmdline = ' '.join(proc.cmdline())
                        script_name = cmdline.split('/')[-1].split()[0]
                        print(f"   PID {proc.pid}: {script_name} | CPU: {cpu_usage:.1f}% | RAM: {memory_mb:.0f}MB")
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
            else:
                print(f"\n🎯 No training processes detected")
                print("   (Looking for: model_trainer.py, train_only.py, train.py)")
            
            # Resource utilization analysis
            print(f"\n📈 Resource Utilization Analysis:")
            if cpu_percent > 85:
                print("   🟢 Excellent CPU utilization!")
            elif cpu_percent > 70:
                print("   🟡 Good CPU utilization")
            elif cpu_percent > 50:
                print("   🟠 Moderate CPU utilization - room for improvement")
            else:
                print("   🔴 Low CPU utilization - consider optimization")
            
            if memory_percent > 70:
                print("   🟢 Good memory utilization!")
            elif memory_percent > 50:
                print("   🟡 Moderate memory utilization")
            else:
                print("   🟠 Low memory utilization - can use larger batches/models")
            
            print("=" * 60)
            print(f"💡 Press Ctrl+C to stop monitoring")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print(f"\n\n⏸️  Monitoring stopped")
        print(f"📊 Final stats after {iteration} updates:")
        print(f"   Last CPU: {cpu_percent:.1f}%")
        print(f"   Last Memory: {memory_used_gb:.1f}GB ({memory_percent:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description='Monitor M4 Max performance during training')
    parser.add_argument('--interval', type=int, default=3,
                       help='Update interval in seconds (default: 3)')
    
    args = parser.parse_args()
    
    try:
        monitor_performance(args.interval)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()