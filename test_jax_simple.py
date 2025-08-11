#!/usr/bin/env python3
"""
Simple JAX optimization test - CPU performance improvements
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np
import time

print("=" * 60)
print("🚀 JAX CPU Optimizations for Trading")
print("=" * 60)
print(f"JAX version: {jax.__version__}")
print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")
print()

# 1. Simple vectorized reward calculation
@jit
def calculate_rewards_jax(prices, positions, actions):
    """Calculate trading rewards using JAX"""
    # Actions: [buy_amount, sell_amount] for each step
    buys = actions[:, 0]
    sells = actions[:, 1]
    
    # Update positions
    new_positions = positions + buys - sells
    
    # Calculate P&L
    price_changes = jnp.diff(prices, prepend=prices[0])
    pnl = new_positions * price_changes
    
    # Transaction costs
    costs = (jnp.abs(buys) + jnp.abs(sells)) * prices * 0.001
    
    # Net rewards
    rewards = pnl - costs
    return rewards

def calculate_rewards_numpy(prices, positions, actions):
    """Same calculation using NumPy"""
    buys = actions[:, 0]
    sells = actions[:, 1]
    new_positions = positions + buys - sells
    price_changes = np.diff(prices, prepend=prices[0])
    pnl = new_positions * price_changes
    costs = (np.abs(buys) + np.abs(sells)) * prices * 0.001
    rewards = pnl - costs
    return rewards

# 2. Batch environment step
@jit
def env_step_batch_jax(states, actions):
    """Process multiple environment steps in parallel"""
    # States: [cash, position] for each env
    # Actions: [action_type, amount] for each env
    
    cash = states[:, 0]
    positions = states[:, 1]
    
    action_types = actions[:, 0]
    amounts = actions[:, 1]
    
    # Simple trading logic
    # Buy when action_type > 0, sell when < 0
    buy_mask = action_types > 0
    sell_mask = action_types < 0
    
    # Calculate new positions
    buy_amounts = jnp.where(buy_mask, amounts * 100, 0)
    sell_amounts = jnp.where(sell_mask, jnp.minimum(positions, amounts * 100), 0)
    
    new_positions = positions + buy_amounts - sell_amounts
    new_cash = cash - buy_amounts * 100 + sell_amounts * 100
    
    # Simple reward: change in total value
    old_value = cash + positions * 100
    new_value = new_cash + new_positions * 100
    rewards = (new_value - old_value) / old_value
    
    new_states = jnp.stack([new_cash, new_positions], axis=1)
    return new_states, rewards

# 3. Technical indicator calculation
@jit
def calculate_indicators_jax(prices):
    """Calculate simple technical indicators"""
    # Returns
    returns = jnp.diff(prices) / prices[:-1]
    
    # Simple moving averages (fixed windows)
    def sma(arr, window):
        cumsum = jnp.cumsum(jnp.concatenate([jnp.zeros(1), arr]))
        return (cumsum[window:] - cumsum[:-window]) / window
    
    sma_5 = sma(prices, 5)
    sma_10 = sma(prices, 10)
    
    # RSI simplified
    gains = jnp.maximum(0, jnp.diff(prices))
    losses = jnp.maximum(0, -jnp.diff(prices))
    avg_gain = sma(gains, 14)
    avg_loss = sma(losses, 14)
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return returns, sma_5, sma_10, rsi

# Benchmark
print("📊 Benchmarking JAX vs NumPy")
print("-" * 40)

# Test data
n_steps = 10000
n_envs = 1000

# Generate random data
np.random.seed(42)
prices = np.abs(np.random.randn(n_steps)) * 10 + 100
positions = np.random.randn(n_steps) * 10
actions = np.random.randn(n_steps, 2)

# Convert to JAX
prices_jax = jnp.array(prices)
positions_jax = jnp.array(positions)
actions_jax = jnp.array(actions)

# Test 1: Reward calculation
print("\n1️⃣ Reward Calculation (10,000 steps):")

# NumPy
start = time.time()
for _ in range(100):
    rewards_np = calculate_rewards_numpy(prices, positions, actions)
numpy_time = time.time() - start

# JAX (with JIT compilation)
# Warm-up to compile
_ = calculate_rewards_jax(prices_jax, positions_jax, actions_jax)

start = time.time()
for _ in range(100):
    rewards_jax = calculate_rewards_jax(prices_jax, positions_jax, actions_jax).block_until_ready()
jax_time = time.time() - start

print(f"   NumPy: {numpy_time:.3f}s")
print(f"   JAX:   {jax_time:.3f}s")
print(f"   Speedup: {numpy_time/jax_time:.2f}x")

# Test 2: Batch environment steps
print("\n2️⃣ Batch Environment Steps (1,000 envs):")

states = np.random.randn(n_envs, 2) * 1000 + 10000
actions = np.random.randn(n_envs, 2)
states_jax = jnp.array(states)
actions_jax = jnp.array(actions)

# Warm-up
_ = env_step_batch_jax(states_jax, actions_jax)

start = time.time()
for _ in range(1000):
    new_states, rewards = env_step_batch_jax(states_jax, actions_jax)
    rewards.block_until_ready()
jax_batch_time = time.time() - start

steps_per_sec = (1000 * n_envs) / jax_batch_time
print(f"   JAX: {jax_batch_time:.3f}s")
print(f"   Steps/second: {steps_per_sec:,.0f}")

# Test 3: Technical indicators
print("\n3️⃣ Technical Indicators (10,000 prices):")

# Warm-up
_ = calculate_indicators_jax(prices_jax)

start = time.time()
for _ in range(100):
    indicators = calculate_indicators_jax(prices_jax)
    # Force computation
    indicators[0].block_until_ready()
jax_indicator_time = time.time() - start

print(f"   JAX: {jax_indicator_time:.3f}s for 100 iterations")
print(f"   Time per calculation: {jax_indicator_time/100*1000:.2f}ms")

print("\n" + "=" * 60)
print("📈 Summary")
print("=" * 60)
print("JAX on CPU provides significant speedups through:")
print(f"✅ JIT compilation: {numpy_time/jax_time:.1f}x faster for rewards")
print(f"✅ Vectorization: {steps_per_sec:,.0f} env steps/second")
print(f"✅ Efficient ops: {jax_indicator_time/100*1000:.1f}ms per indicator calc")
print("\n💡 Key Insights:")
print("- JAX works great on CPU (no Metal/GPU needed)")
print("- JIT compilation eliminates Python overhead")
print("- Best for operations called many times (like env.step)")
print("- Could provide 2-5x overall training speedup")
print("\n🎯 Next Steps:")
print("1. Implement JAX version of environment step")
print("2. Use JAX for technical indicator calculations")
print("3. Consider JAX-based PPO implementation")
print("=" * 60)