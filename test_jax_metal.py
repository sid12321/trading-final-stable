#!/usr/bin/env python3
"""
Test JAX Metal backend configuration
"""

import os
import sys

# Set JAX to use Metal before importing jax
os.environ['JAX_PLATFORMS'] = 'metal'
os.environ['JAX_METAL_DEVICE_ID'] = '0'

import jax
import jax.numpy as jnp
from jax import jit, vmap
import time
import numpy as np

def check_jax_backend():
    """Check which backend JAX is using"""
    print("=" * 60)
    print("🔍 JAX Configuration Check")
    print("=" * 60)
    
    # Check JAX version
    print(f"JAX version: {jax.__version__}")
    
    # Check available backends
    print(f"Available backends: {jax.lib.xla_bridge.get_backend().platform}")
    
    # Check default backend
    backend = jax.default_backend()
    print(f"Current backend: {backend}")
    
    # Check devices
    devices = jax.devices()
    print(f"Available devices: {devices}")
    for i, device in enumerate(devices):
        print(f"  Device {i}: {device}")
        print(f"    Platform: {device.platform}")
        print(f"    Device kind: {device.device_kind}")
    
    # Check if Metal is available
    try:
        from jax._src.lib import xla_client
        print(f"\nXLA backends available: {xla_client.get_plugin_names()}")
    except:
        print("\nCould not check XLA backends")
    
    print("\n" + "=" * 60)
    return backend

def benchmark_jax_operations():
    """Benchmark JAX operations on current backend"""
    print("🚀 Benchmarking JAX Operations")
    print("=" * 60)
    
    # Test matrix multiplication
    size = 2000
    key = jax.random.PRNGKey(0)
    
    # Create random matrices
    A = jax.random.normal(key, (size, size))
    B = jax.random.normal(key, (size, size))
    
    # JIT compile the operation
    @jit
    def matmul(A, B):
        return jnp.dot(A, B)
    
    # Warm-up
    _ = matmul(A, B).block_until_ready()
    
    # Benchmark
    start = time.time()
    for _ in range(10):
        result = matmul(A, B).block_until_ready()
    elapsed = time.time() - start
    
    print(f"Matrix multiplication ({size}x{size}):")
    print(f"  Time for 10 iterations: {elapsed:.3f} seconds")
    print(f"  Average per iteration: {elapsed/10:.3f} seconds")
    
    # Test vectorized operations (like environment steps)
    @jit
    @vmap
    def simple_env_step(obs, action):
        # Simulate environment computation
        reward = jnp.sum(obs * action)
        next_obs = obs + action * 0.1
        return next_obs, reward
    
    batch_size = 1000
    obs_dim = 100
    
    obs = jax.random.normal(key, (batch_size, obs_dim))
    actions = jax.random.normal(key, (batch_size, obs_dim))
    
    # Warm-up
    _ = simple_env_step(obs, actions)
    
    # Benchmark
    start = time.time()
    for _ in range(100):
        next_obs, rewards = simple_env_step(obs, actions)
        rewards.block_until_ready()
    elapsed = time.time() - start
    
    print(f"\nVectorized environment steps (batch={batch_size}):")
    print(f"  Time for 100 iterations: {elapsed:.3f} seconds")
    print(f"  Steps per second: {(100 * batch_size) / elapsed:.0f}")
    
    print("=" * 60)

def test_metal_specific_operations():
    """Test operations that should be fast on Metal"""
    print("\n🔧 Testing Metal-Optimized Operations")
    print("=" * 60)
    
    # These operations are optimized for Metal GPUs
    key = jax.random.PRNGKey(42)
    
    # 1. Convolution (common in feature extraction)
    @jit
    def conv_operation(x):
        # Simulate 1D convolution for time series
        kernel = jnp.ones((5, 1, 1))
        return jax.lax.conv_general_dilated(
            x[None, :, None],  # Add batch and channel dims
            kernel,
            window_strides=[1],
            padding='SAME'
        )[0, :, 0]
    
    data = jax.random.normal(key, (10000,))
    
    start = time.time()
    for _ in range(100):
        result = conv_operation(data).block_until_ready()
    conv_time = time.time() - start
    
    print(f"1D Convolution (length=10000):")
    print(f"  Time for 100 iterations: {conv_time:.3f} seconds")
    
    # 2. Batch normalization (used in PPO networks)
    @jit
    def batch_norm(x):
        mean = jnp.mean(x, axis=0)
        std = jnp.std(x, axis=0) + 1e-5
        return (x - mean) / std
    
    batch_data = jax.random.normal(key, (1000, 100))
    
    start = time.time()
    for _ in range(1000):
        result = batch_norm(batch_data).block_until_ready()
    bn_time = time.time() - start
    
    print(f"\nBatch Normalization (1000x100):")
    print(f"  Time for 1000 iterations: {bn_time:.3f} seconds")
    
    print("=" * 60)

if __name__ == "__main__":
    # Check current backend
    backend = check_jax_backend()
    
    # Run benchmarks
    benchmark_jax_operations()
    
    # Test Metal-specific operations
    test_metal_specific_operations()
    
    # Provide recommendations
    print("\n📊 RECOMMENDATIONS")
    print("=" * 60)
    
    if "metal" in backend.lower():
        print("✅ JAX is using Metal backend - GPU acceleration enabled!")
        print("   Your M4 Max GPU is being utilized for JAX operations.")
    elif "cpu" in backend.lower():
        print("⚠️  JAX is using CPU backend - GPU acceleration NOT enabled!")
        print("\n📝 To enable Metal acceleration, you need to:")
        print("1. Install JAX with Metal support:")
        print("   pip uninstall jax jaxlib")
        print("   pip install jax-metal")
        print("\n2. Set environment variables before importing JAX:")
        print("   export JAX_PLATFORMS=metal")
        print("\n3. Or in Python before importing jax:")
        print("   os.environ['JAX_PLATFORMS'] = 'metal'")
        print("\n4. Verify with: jax.default_backend()")
    else:
        print(f"🤔 JAX is using {backend} backend")
        print("   Consider switching to Metal for GPU acceleration.")
    
    print("\n💡 For your trading system:")
    print("- Metal acceleration would significantly speed up JAX operations")
    print("- Especially beneficial for vectorized environment steps")
    print("- Could provide 5-10x additional speedup over CPU JAX")
    print("=" * 60)